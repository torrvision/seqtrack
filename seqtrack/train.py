from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
import time
import os
import pprint
import random
import re
import subprocess
import threading
from functools import partial
from itertools import chain
from six import string_types

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import draw
from seqtrack import track
from seqtrack import geom
from seqtrack import graph
from seqtrack import helpers
from seqtrack import motion
from seqtrack import pipeline
from seqtrack import sample
from seqtrack.models.itermodel import ModelFromIterModel
from seqtrack.models.siamfc import SiamFC


# Pickling a function in one module to load in another module does not work if
# the function is defined in the same module.
# Therefore provide a worker function here that takes tmp_dir from the mapper.
def train_worker(context, *args, **kwargs):
    # The mapper context provides a temporary directory.
    return train(*args, tmp_dir=context.tmp_dir(), **kwargs)


def train(
        dir,
        model_params,
        seed,
        only_evaluate_existing=False,
        tmp_dir=None,
        # Dataset:
        train_dataset=None,
        val_dataset=None,
        eval_datasets=None,
        pool_datasets=None,
        pool_split=None,
        untar=None,
        data_dir=None,
        tar_dir=None,
        use_tmp_data_dir=None,
        preproc_id=None,
        data_cache_dir=None,
        # Sampling:
        sampler_params=None,
        augment_motion=False,
        motion_params=None,
        # Evaluation:
        eval_samplers=None,
        max_eval_videos=None,
        # Training process:
        ntimesteps=None,  # Needed for constructing sampler.
        **kwargs):
    '''
    Args:
        kwargs: For train_model_data.

    Specify either `data_dir` or `use_tmp_data_dir`.
    The option `use_tmp_data_dir` implies `untar`.
    It is possible to specify `untar` without `use_tmp_data_dir` to untar to `data_dir`.

    Side effects:
        Resets global random seed.
    '''
    tf.reset_default_graph()
    _set_global_seed(seed)  # Caution: Global side effects!

    if only_evaluate_existing:
        train_dataset = None
        val_dataset = None

    if eval_samplers is None:
        eval_samplers = ['full']

    if use_tmp_data_dir:
        untar = True
        data_dir = os.path.join(tmp_dir, 'data')

    # TODO: How to get datasets from train_dataset and eval_datasets?
    dataset_names = list(set(chain(_datasets_in_sampler(train_dataset),
                                   _datasets_in_sampler(val_dataset),
                                   eval_datasets)))
    # TODO: Flag to enable/disable.
    datasets = setup_data(
        dataset_names=dataset_names,
        pool_datasets=pool_datasets,
        pool_split=pool_split,
        untar=untar,
        data_dir=data_dir,
        tar_dir=tar_dir,
        preproc_id=preproc_id,
        data_cache_dir=data_cache_dir)

    siamese = SiamFC(**model_params)
    model_properties = siamese.derived_properties()
    model = ModelFromIterModel(siamese)

    streams, eval_sample_fns = make_samplers(
        datasets,
        seed=seed,
        ntimesteps=ntimesteps,
        sampler_params=sampler_params,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        augment_motion=augment_motion,
        motion_params=motion_params,
        eval_datasets=eval_datasets,
        eval_samplers=eval_samplers,
        max_eval_videos=max_eval_videos)

    train_series, track_series = train_model_data(
        dir, model, streams, eval_sample_fns,
        only_evaluate_existing=only_evaluate_existing,
        ntimesteps=ntimesteps,
        **kwargs)

    return make_train_result(
        model_properties=model_properties,
        train_series=train_series,
        track_series=track_series,
    )


def make_train_result(model_properties, train_series, track_series):
    '''Returns a dictionary that represents the result of a single training procedure.

    This includes various metrics as time-series during training.

    Args:
        model_properties: Derived properties of model.
        train_series: Time-series of training metrics.
        track_series: Time-series of tracking metrics.
    '''
    return dict(
        model_properties=model_properties,
        train_series=train_series,
        track_series=track_series,
    )


def setup_data(dataset_names, pool_datasets, pool_split,
               untar, data_dir, tar_dir, preproc_id, data_cache_dir):
    logger.info('load datasets: %s', helpers.quote_list(dataset_names))

    # If 'pool_train' or 'pool_val' are in dataset_names, replace them with the pool datasets.
    use_pool = any(x.startswith('pool') for x in dataset_names)
    if use_pool:
        dataset_names = [x for x in dataset_names if not x.startswith('pool')]
        dataset_names += pool_datasets

    if untar:
        datasets = data.untar_and_load_all(
            tar_dir, data_dir, preproc_id, dataset_names,
            cache_dir=data_cache_dir)
    else:
        datasets = {
            name: data.load(data_dir, preproc_id, name,
                            cache=True, cache_dir=data_cache_dir) for name in dataset_names}

    if use_pool:
        # Add pool_train and pool_val to datasets.
        # Split the datasets into train and val.
        pool_splits = {
            name: data.split_dataset(datasets[name], [pool_split, 1 - pool_split], seed=0)
            for name in pool_datasets}
        # Concat the individual train and val datasets.
        datasets['pool_train'] = data.Concat({x: pool_splits[x][0] for x in pool_datasets})
        datasets['pool_val'] = data.Concat({x: pool_splits[x][1] for x in pool_datasets})

    return datasets


# def _make_model_spec(args):
#     if args.model == 'SiamFC':
#         model = ModelFromIterModel(SiamFC(**args.model_params))
#     else:
#         raise ValueError('unknown model: {}'.format(args.model))
#     return model


def _set_global_seed(seed):
    # TODO: Not sure if TensorFlow initialisation is deterministic given global seed?
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_samplers(datasets, seed, ntimesteps, sampler_params, train_dataset, val_dataset,
                  augment_motion, motion_params,
                  eval_datasets, eval_samplers, max_eval_videos):
    frame_sampler_presets = {
        'full': partial(sample.FrameSampler, kind='full'),
        'train': partial(sample.FrameSampler, ntimesteps=ntimesteps, **sampler_params)}

    # Create infinite example streams for train and val.
    # Use a separate random number generator for each sampler as they may run in parallel.
    sampler_specs = {'train': train_dataset, 'val': val_dataset}
    streams = {}
    for i, mode in enumerate(['train', 'val']):
        if sampler_specs[mode] is None:
            continue
        # Use a different seed for train and val, in particular for augmentation!
        postproc_fn = (
            None if not augment_motion else
            partial(motion.augment, rand=np.random.RandomState(seed + i), **motion_params))
        streams[mode] = sample.sample(
            _make_video_sampler(sampler_specs[mode], datasets), frame_sampler_presets['train'](),
            postproc_fn=postproc_fn, rand=np.random.RandomState(seed + i), infinite=True)
    # Create functions to sample finite sets for evaluation.
    eval_sample_fns = {
        # Give each dataset its own random stream.
        (d + '-' + s): partial(
            sample.sample, sample.EpochSampler(datasets[d]), frame_sampler_presets[s](),
            rand=np.random.RandomState(seed),
            infinite=False, max_num=max_eval_videos)
        for d in eval_datasets for s in eval_samplers}
    return streams, eval_sample_fns


def _datasets_in_sampler(names):
    if names is None:
        return []
    if isinstance(names, string_types):
        return [names]
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, string_types) for name in names):
            return names
        elif all(_is_pair(elem) for elem in names):
            return [name for _, name in names]
    raise ValueError('invalid structure: {}'.format(repr(names)))


def _make_video_sampler(names, datasets):
    '''
    Args:
        names: string (EpochSampler), list of strings (EpochSampler of Concat) or
            list of float-string-pairs (MixtureSampler).
    '''
    if isinstance(names, string_types):
        return sample.EpochSampler(datasets[names])
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, string_types) for name in names):
            concat = data.Concat({name: datasets[name] for name in names})
            return sample.EpochSampler(concat)
        elif all(_is_pair(elem) for elem in names):
            samplers = {name: sample.EpochSampler(datasets[name]) for _, name in names}
            weights = {name: weight for weight, name in names}
            return sample.MixtureSampler(samplers, weights)
    raise ValueError('invalid structure: {}'.format(repr(names)))


# TODO: Look at resume and seeds.

def train_model_data(
        dir, model, sequences, eval_sets,
        override_ckpt_dir=None,
        # Args that affect the operation but not the result.
        only_evaluate_existing=False,
        resume=False,
        tfdb=False,
        use_queues=True,
        summary_dir=None,
        summary_name=None,
        nosave=False,
        metrics_resolution=1000,
        period_ckpt=10000,
        period_assess=40000,
        extra_assess=None,
        period_skip=0,  # TODO: Change name since not periodic?
        period_summary=10,
        period_preview=100,
        verbose_train=False,
        visualize=False,
        keep_frames=False,
        session_config_kwargs=None,
        # Training args:
        ntimesteps=None,
        batchsz=None,
        imwidth=None,
        imheight=None,
        lr_schedule='constant',
        lr_init=1e-3,
        lr_params=None,
        optimizer=None,
        optimizer_params=None,
        grad_clip=False,
        grad_clip_params=None,
        siamese_pretrain=None,
        siamese_model_file=None,
        num_steps=None,
        use_gt_train=True,
        gt_decay_rate=1,
        min_gt_ratio=0,
        # Evaluation args:
        eval_tre_num=None):
    '''Trains a network.

    Args:
        create_model: Function that takes as input a dictionary of tensors and
            returns a model object.
        datasets: Dictionary of datasets with keys 'train' and 'val'.
        eval_sets: A dictionary of sampling functions which return collections
            of sequences on which to evaluate the tracker.

    Returns:
        Tuple of (train_series, track_series).
        train_series[step][metric]
        track_series[step][dataset][metric]

    The method model.instantiate() constructs the model in the tf graph.
    The model is expected to use variables scopes.

    The input dictionary has fields::

        'x0'         # First image in sequence, shape [b, h, w, 3]
        'y0'         # Position of target in first image, shape [b, 4]
        'x'          # Input images, shape [b, n, h, w, 3]
        'y_is_valid' # Booleans indicating presence of frame, shape [b, n]
        'aspect'     # Aspect ratio (width/height) of original image, shape [b]

    and the output dictionary has fields::

        'y'       # (optional) Predicted position of target in each frame, shape [b, n, 4]
        'heatmap' # (optional) Score for pixel belonging to target, shape [b, n, h, w, 1]

    The images provided to the model are already normalized (e.g. dataset mean subtracted).

    Each sampler in the eval_sets dictionary is a function that
    returns a collection of sequences or is a finite generator of sequences.
    '''
    extra_assess = extra_assess or []
    session_config_kwargs = session_config_kwargs or {}
    optimizer_params = optimizer_params or {}
    grad_clip_params = grad_clip_params or {}

    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    path_ckpt = override_ckpt_dir or os.path.join(dir, 'ckpt')
    path_output = os.path.join(dir, 'output')
    if summary_dir:
        assert summary_name is not None
        path_summary = os.path.join(summary_dir, summary_name)
    else:
        path_summary = os.path.join(dir, 'summary')

    # How should we compute training and validation error with pipelines?
    # Option 1 is to have multiple copies of the network with shared parameters.
    # However, this makes it difficult to plot training and validation error on the same axes
    # since there are separate summary ops for training and validation (with different names).
    # Option 2 is to use FIFOQueue.from_list()

    modes = ['train', 'val']

    feed_loop = {}  # Each value is a function to call in a thread.
    with tf.name_scope('input'):
        from_queue = None
        if use_queues:
            queues = []
            for mode in modes:
                # Create a queue each for training and validation data.
                queue, feed_loop[mode] = _make_input_pipeline(
                    ntimesteps=ntimesteps, batchsz=batchsz,
                    im_size=(imheight, imwidth),
                    num_load_threads=1, num_batch_threads=1, name='pipeline_' + mode)
                queues.append(queue)
            queue_index, from_queue = pipeline.make_multiplexer(queues, capacity=4, num_threads=1)
        example, run_opts = graph.make_placeholders(
            ntimesteps, (imheight, imwidth), default=from_queue)

    # example = _perform_color_augmentation(example, args)
    metric_vars = {}

    example_input = graph.whiten(example)
    with tf.variable_scope('model'):
        outputs, losses, init_state, final_state = model.instantiate(
            example_input, run_opts, enable_loss=True,
            image_summaries_collections=['IMAGE_SUMMARIES'])
    metric_vars.update(losses)
    _loss_summary(losses)
    # _image_summary(example, outputs)
    # loss_var = _add_losses(losses, args.loss_coeffs)
    loss_var = _add_losses(losses, {})

    with tf.name_scope('diagnostic'):
        iou = geom.rect_iou(outputs['y'], example_input['y'])
        mean_iou = tf.reduce_mean(tf.boolean_mask(iou, example_input['y_is_valid']))
        metric_vars['iou'] = mean_iou
        tf.summary.scalar('iou', mean_iou)

        dist = tf.norm(geom.rect_center(outputs['y']) - geom.rect_center(example_input['y']), axis=-1)
        dist = dist / tf.reduce_mean(geom.rect_size(example_input['y']), axis=-1)
        mean_dist = tf.reduce_mean(tf.boolean_mask(dist, example_input['y_is_valid']))
        tf.summary.scalar('dist', mean_dist)
        metric_vars['dist'] = mean_dist

    model_inst = graph.ModelInstance(
        example, run_opts, outputs, init_state, final_state,
        batchsz=outputs['y'].shape.as_list()[0], ntimesteps=ntimesteps,
        imheight=imheight, imwidth=imwidth)

    r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_var += r
    metric_vars['loss'] = loss_var
    tf.summary.scalar('regularization', r)
    tf.summary.scalar('total', loss_var)

    global_step_var = tf.Variable(0, name='global_step', trainable=False)

    lr = make_learning_rate(global_step_var, num_steps, lr_init,
                            schedule=lr_schedule, schedule_params=lr_params)
    tf.summary.scalar('lr', lr, collections=['summaries_train'])
    optimizer_obj = make_optimizer(lr, optimizer, **optimizer_params)
    grads_and_vars = optimizer_obj.compute_gradients(loss_var)
    if grad_clip:
        grads_and_vars = clip_gradients(grads_and_vars, **(grad_clip_params or {}))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize_op = optimizer_obj.apply_gradients(grads_and_vars, global_step=global_step_var)

    summary_vars = {}
    summary_vars_with_preview = {}
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        image_summaries = tf.get_collection('IMAGE_SUMMARIES')
        for mode in modes:
            with tf.name_scope(mode):
                # Merge summaries with any that are specific to the mode.
                summaries = (global_summaries + tf.get_collection('summaries_' + mode))
                summary_vars[mode] = tf.summary.merge(summaries)
                summaries.extend(image_summaries)
                # Assume that model summaries could contain images.
                # TODO: Separate model summaries into image and non-image.
                summaries.extend(tf.get_collection('summaries_model'))
                summary_vars_with_preview[mode] = tf.summary.merge(summaries)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=100)

    # if args.siamese_pretrain:
    #     siamese_vars = tf.get_collection('siamese')
    #     print 'siamese vars:'
    #     pprint.pprint(siamese_vars)
    #     saver_siamese = tf.train.Saver(siamese_vars)

    # Training metrics are accumulated over many steps.
    train_series = {}
    train_accum = helpers.DictAccumulator()
    val_accum = helpers.DictAccumulator()
    # Tracking metrics are obtained periodically.
    track_series = {}

    t_total = time.time()
    with tf.Session(config=make_session_config(**session_config_kwargs)) as sess:
        print('\ntraining starts! --------------------------------------------')
        sys.stdout.flush()

        if only_evaluate_existing:
            return _evaluate_at_existing_checkpoints(
                saver, eval_sets, sess, model_inst,
                path_ckpt=path_ckpt,
                period_skip=period_skip,
                period_assess=period_assess,
                extra_assess=extra_assess,
                # For _evaluate():
                eval_tre_num=eval_tre_num,
                path_output=path_output,
                visualize=visualize,
                keep_frames=keep_frames)

        # 1. resume (full restore), 2. initialize from scratch, 3. curriculume learning (partial restore)
        prev_ckpt = 0
        if resume:
            model_file = tf.train.latest_checkpoint(path_ckpt)
            if model_file is None:
                raise ValueError('could not find checkpoint')
            print('restore: {}'.format(model_file))
            saver.restore(sess, model_file)
            print('done: restore')
            sys.stdout.flush()
            prev_ckpt = np.asscalar(global_step_var.eval())
        else:
            sess.run(init_op)
            model.init(sess)  # Loads pre-trained features if applicable.
            # if args.siamese_pretrain:
            #     saver_siamese.restore(sess, args.siamese_model_file)
            #     # vars_uninit = sess.run(tf.report_uninitialized_variables())
            #     # print 'vars_uninit:'
            #     # pprint.pprint(vars_uninit)
            #     # sess.run(tf.variables_initializer([v for v in tf.global_variables()
            #     #                                    if v.name.split(':')[0] in vars_uninit]))
            #     # assert len(sess.run(tf.report_uninitialized_variables())) == 0

        if use_queues:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess, coord)
            # Run the feed loops in another thread.
            threads = [threading.Thread(target=feed_loop[mode],
                                        args=(sess, coord, sequences[mode]))
                       for mode in modes]
            for t in threads:
                t.start()

        if tfdb:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = {}
        for mode in modes:
            # Only include graph in one summary.
            if mode == 'train':
                writer[mode] = tf.summary.FileWriter(os.path.join(path_summary, mode), sess.graph)
            else:
                writer[mode] = tf.summary.FileWriter(os.path.join(path_summary, mode))

        while True:
            global_step = np.asscalar(global_step_var.eval())
            assess_step = _to_assess_step(global_step, period_assess, period_skip, extra_assess)
            if not nosave:
                if global_step > prev_ckpt and (global_step % period_ckpt == 0 or assess_step):
                    if not os.path.isdir(path_ckpt):
                        os.makedirs(path_ckpt)
                    print('save model')
                    saver.save(sess, path_ckpt + '/iteration', global_step=global_step)
                    print('done: save model')
                    sys.stdout.flush()
                    prev_ckpt = global_step
            # intermediate evaluation of model
            if assess_step:
                for eval_id, sampler in eval_sets.items():
                    eval_sequences = sampler()
                    result = _evaluate(
                        global_step, eval_id, sess, model_inst, eval_sequences,
                        eval_tre_num=eval_tre_num,
                        path_output=path_output,
                        visualize=visualize,
                        keep_frames=keep_frames)
                    track_series.setdefault(global_step, {})[eval_id] = result

            if global_step > 0 and global_step % metrics_resolution == 0:
                train_series[global_step] = combine_dicts(train=train_accum.flush(),
                                                          val=val_accum.flush())

            if global_step >= num_steps:
                break

            if use_gt_train:
                # TODO: Use a different base!
                gt_ratio = max(1.0 * np.exp(-gt_decay_rate * global_step), min_gt_ratio)
            else:
                gt_ratio = 0

            # Take a training step.
            start = time.time()
            feed_dict = {run_opts['use_gt']: use_gt_train,
                         run_opts['is_training']: True,
                         run_opts['is_tracking']: False,
                         run_opts['gt_ratio']: gt_ratio}
            if use_queues:
                feed_dict.update({queue_index: 0})  # Choose validation queue.
            else:
                batch_seqs = [next(sequences['train']) for i in range(batchsz)]
                # batch = _load_batch(batch_seqs, args)
                batch = graph.py_load_batch(batch_seqs, ntimesteps, (imheight, imwidth))
                feed_dict.update({example[k]: v for k, v in batch.items()})
                dur_load = time.time() - start
            if global_step % period_summary == 0:
                summary_var = (summary_vars_with_preview['train']
                               if global_step % period_preview == 0
                               else summary_vars['train'])
                _, loss, train_metrics, summary = sess.run(
                    [optimize_op, loss_var, metric_vars, summary_var], feed_dict=feed_dict)
                dur = time.time() - start
                writer['train'].add_summary(summary, global_step=global_step)
            else:
                _, loss, train_metrics = sess.run(
                    [optimize_op, loss_var, metric_vars], feed_dict=feed_dict)
                dur = time.time() - start
            train_accum.add(train_metrics)

            # TODO: Avoid copy paste here!

            newval = False
            # Evaluate validation error.
            if global_step % period_summary == 0:
                start = time.time()
                feed_dict = {run_opts['use_gt']: use_gt_train,  # Match training.
                             run_opts['is_training']: False,  # Do not update bnorm stats.
                             run_opts['is_tracking']: False,
                             run_opts['gt_ratio']: gt_ratio}  # Match training.
                if use_queues:
                    feed_dict.update({queue_index: 1})  # Choose validation queue.
                else:
                    batch_seqs = [next(sequences['val']) for i in range(batchsz)]
                    # batch = _load_batch(batch_seqs, args)
                    batch = graph.py_load_batch(batch_seqs, ntimesteps, (imheight, imwidth))
                    feed_dict.update({example[k]: v for k, v in batch.items()})
                    dur_load = time.time() - start
                summary_var = (summary_vars_with_preview['val']
                               if global_step % period_preview == 0
                               else summary_vars['val'])
                # For now assume that val_metrics variables are same as train_metrics.
                loss_val, val_metrics, summary = sess.run(
                    [loss_var, metric_vars, summary_var], feed_dict=feed_dict)
                dur_val = time.time() - start
                writer['val'].add_summary(summary, global_step=global_step)
                newval = True
                val_accum.add(val_metrics)

            # Print result of one batch update
            if verbose_train:
                losstime = '|loss:{:.5f}/{:.5f} (time:{:.2f}/{:.2f}) - with val'.format(
                    loss, loss_val, dur, dur_val) if newval else \
                    '|loss:{:.5f} (time:{:.2f})'.format(loss, dur)
                print('global_step {} {}'.format(global_step, losstime))
                sys.stdout.flush()

        # **training finished
        print('\ntraining finished! ------------------------------------------')
        print('total time elapsed: {0:.2f}'.format(time.time() - t_total))

        return train_series, track_series


def make_learning_rate(global_step, num_steps, init,
                       schedule='constant', schedule_params=None,
                       name='learning_rate'):
    schedule_params = schedule_params or {}
    # lr = init * decay^(step)
    #    = init * decay^(step / period * period / decay_steps)
    #    = init * [decay^(period / decay_steps)]^(step / period)
    try:
        fn = {
            'constant': _learning_rate_constant,
            'exponential': _learning_rate_exponential,
            'remain': _learning_rate_remain,
        }[schedule]
    except KeyError as ex:
        raise ValueError('unknown schedule: "{}"'.format(schedule))

    with tf.name_scope(name) as scope:
        return fn(global_step, num_steps, init, **schedule_params)


def _learning_rate_constant(t, n, init):
    return tf.constant(init)


def _learning_rate_exponential(t, n, init, **kwargs):
    return tf.train.exponential_decay(init, t, **kwargs)


def _learning_rate_remain(t, n, init, decay_rate=0.1, remain_rate=0.5, max_power=None):
    frac_rem = tf.to_float(n - t) / n
    k = tf.floor(tf.log(frac_rem) / tf.log(remain_rate))
    if max_power is not None:
        k = tf.minimum(k, float(max_power))
    return init * tf.pow(float(decay_rate), k)


def make_optimizer(learning_rate, name, **kwargs):
    try:
        optimizer_fn = {
            'sgd': tf.train.GradientDescentOptimizer,
            'momentum': tf.train.MomentumOptimizer,
            'adam': tf.train.AdamOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer,
        }[name]
    except KeyError as ex:
        raise ValueError('unknown optimizer: {}'.format(ex))

    return optimizer_fn(learning_rate, **kwargs)


def clip_gradients(grads_and_vars, max_grad_norm):
    def clip(x):
        return None if x is None else tf.clip_by_norm(x, max_grad_norm)
    return [(clip(g), v) for g, v in grad_and_vars.items()]


def make_session_config(gpu_manctrl=False, gpu_frac=1.0, log_device_placement=False):
    config = tf.ConfigProto(log_device_placement=log_device_placement)
    # TODO: not sure if this should be always true.
    config.allow_soft_placement = True
    # config.log_device_placement = True
    if gpu_manctrl:
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac


def _make_input_pipeline(ntimesteps, batchsz, im_size, dtype=tf.float32,
                         example_capacity=4, load_capacity=4, batch_capacity=4,
                         num_load_threads=1, num_batch_threads=1,
                         name='pipeline'):
    '''
    Args:
        im_size: (height, width) to construct tensor
    '''
    with tf.name_scope(name) as scope:
        height, width = im_size
        files, feed_loop = pipeline.get_example_filenames(capacity=example_capacity)
        images = pipeline.load_images(files, capacity=load_capacity,
                                      num_threads=num_load_threads, image_size=[height, width, 3])
        images_batch = pipeline.batch(images,
                                      batch_size=batchsz, sequence_length=ntimesteps + 1,
                                      capacity=batch_capacity, num_threads=num_batch_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, ntimesteps + 1, None, None, None])
        images_batch['labels'].set_shape([None, ntimesteps + 1, None])
        # Cast type of images.
        # JV: convert_image_dtype changes range to 1, as expected by other tf functions
        # images_batch['images'] = tf.cast(images_batch['images'], tf.float32)
        images_batch['images'] = tf.image.convert_image_dtype(images_batch['images'], tf.float32)
        # Put in format expected by model.
        # is_valid = (range(1, ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        example_batch = {
            'x0': images_batch['images'][:, 0],
            'y0': images_batch['labels'][:, 0],
            'x': images_batch['images'][:, 1:],
            'y': images_batch['labels'][:, 1:],
            'y_is_valid': images_batch['label_is_valid'][:, 1:],
            'aspect': images_batch['aspect'],
        }
        return example_batch, feed_loop


# def _perform_color_augmentation(example_raw, o, name='color_augmentation'):
#
#     example = dict(example_raw)
#
#     xs_aug = tf.concat([tf.expand_dims(example['x0'], 1), example['x']], 1)
#
#     if o.color_augmentation.get('brightness', False):
#         xs_aug = tf.image.random_brightness(xs_aug, 0.1)
#
#     if o.color_augmentation.get('contrast', False):
#         xs_aug = tf.image.random_contrast(xs_aug, 0.1, 1.1)
#
#     if o.color_augmentation.get('grayscale', False):
#         max_grayscale_ratio = 0.2
#         rand_prob = tf.random_uniform(shape=[], minval=0, maxval=1)
#         xs_aug = tf.cond(tf.less_equal(rand_prob, max_grayscale_ratio),
#                          lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(xs_aug)),
#                          lambda: xs_aug)
#
#     example['x0'] = xs_aug[:, 0]
#     example['x'] = xs_aug[:, 1:]
#     return example


def _generate_report(args, samplers, datasets,
                     modes=['OPE', 'TRE'],
                     metrics=['iou_mean', 'auc']):
    '''Finds all results for each evaluation distribution.

    Identifies the best result for each metric.
    Caution: More frequent evaluations might lead to better results.
    '''
    def helper():
        def eval_id_fn(sampler, dataset):
            return '{}-{}'.format(dataset, sampler)
        best_fn = {'iou_mean': np.amax, 'auc': np.amax, 'cle_mean': np.amin, 'cle_representative': np.amax}
        report_dir = os.path.join(args.path_output, 'report')
        if not os.path.isdir(report_dir): os.makedirs(report_dir)

        # Plot each metric versus iteration.
        # Create one plot per sampler, with a line per dataset.
        for sampler in samplers:
            # Load all results using this sampler.
            results = {dataset: load_results(eval_id_fn(sampler, dataset)) for dataset in datasets}
            # Print results for each dataset.
            for dataset in datasets:
                for mode in modes:
                    print('==== evaluation: sampler {}, dataset {}, mode {} ===='.format(sampler, dataset, mode))
                    steps = sorted(results[dataset].keys())
                    # for step in steps:
                    #     print 'iter {}:  {}'.format(step,
                    #         '; '.join(['{}: {:.3g}'.format(metric, results[dataset][step][mode][metric])
                    #                    for metric in metrics]))
                    # for metric in metrics:
                    #     values = [results[dataset][step][mode][metric] for step in steps]
                    #     print 'best {}: {:.3g}'.format(metric, np.asscalar(best_fn[metric](values)))
                    for metric in metrics:
                        r = {step: results[dataset][step][mode] for step in steps}
                        print(metric)
                        print(';'.join([str(step) for step in steps]))
                        print(';'.join(['{:04g}'.format(r[step][metric]) for step in steps]))
                        metric_stddev = metric + '_std_err'
                        print(';'.join(['{:04g}'.format(r[step][metric_stddev]) for step in steps]))
            for mode in modes:
                # Generate plot for each metric.
                # Take union of steps for all datasets.
                steps = sorted(set.union(*[set(r.keys()) for r in results.values()]))
                for metric in metrics:
                    # Plot this metric over time for all datasets.
                    data_file = 'sampler-{}-mode-{}-metric-{}'.format(sampler, mode, metric)
                    with open(os.path.join(report_dir, data_file + '.tsv'), 'w') as f:
                        write_data_file(f, mode, metric, steps, results)
                    try:
                        plot_file = plot_data(report_dir, data_file)
                        print('plot created:', plot_file)
                    except Exception as e:
                        print('could not create plot:', e)

    def load_results(eval_id):
        '''Returns a dictionary from step number to dictionary of metrics.'''
        dirname = os.path.join(args.path_output, 'assess', eval_id)
        pattern = re.compile(r'^iteration(\d+)\.json$')
        results = {}
        for f in os.listdir(dirname):
            if not os.path.isfile(os.path.join(dirname, f)):
                continue
            match = pattern.match(f)
            if not match:
                continue
            step = int(match.group(1))
            with open(os.path.join(dirname, f), 'r') as r:
                results[step] = json.load(r)
        if not results:
            print('warning: no results found:', eval_id)
        return results

    def write_data_file(f, mode, metric, steps, results):
        # Create a column for the variance.
        fieldnames = ['step'] + [x + suffix for x in datasets for suffix in ['', '_std_err']]
        w = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        w.writeheader()
        for step in steps:
            # Map variance of metric to variance of
            row = {
                dataset + suffix:
                    gnuplot_str(results[dataset].get(step, {}).get(mode, {}).get(metric + suffix, None))
                for dataset in datasets
                for suffix in ['', '_std_err']}
            row['step'] = step
            w.writerow(row)

    def plot_data(plot_dir, filename):
        src_dir = os.path.dirname(__file__)
        args = ['gnuplot',
                '-e', 'filename = "{}"'.format(filename),
                os.path.join(src_dir, 'plot_eval_metric.gnuplot'),
                ]
        p = subprocess.Popen(args, cwd=plot_dir)
        p.wait()
        return os.path.join(plot_dir, filename + '.png')

    return helper()


def gnuplot_str(x):
    if x is None:
        return '?'
    return str(x)


def _add_losses(losses, loss_coeffs, name='add_losses'):
    with tf.name_scope(name) as scope:
        assert isinstance(losses, dict)
        assert isinstance(loss_coeffs, dict)
        for k in loss_coeffs:
            if k not in losses:
                raise AssertionError('loss not found: {}'.format(k))
        return tf.add_n([float(loss_coeffs.get(k, 1)) * v for k, v in losses.items()], name=scope)


def _loss_summary(losses, name='loss_summary'):
    with tf.name_scope(name) as scope:
        for key, loss in losses.items():
            tf.summary.scalar(key, loss)


def _image_summary(example, outputs, name='image_summary'):
    with tf.name_scope(name) as scope:
        ntimesteps = example['x'].shape.as_list()[1]
        assert ntimesteps is not None
        tf.summary.image(
            'image_0', max_outputs=1, collections=['IMAGE_SUMMARIES'],
            tensor=_draw_rectangles([example['x0'][0]], gt=[example['y0'][0]]))
        tf.summary.image(
            'image_1_to_n', max_outputs=ntimesteps, collections=['IMAGE_SUMMARIES'],
            tensor=_draw_rectangles(example['x'][0], gt=example['y'][0], pred=outputs['y'][0],
                                    gt_is_valid=example['y_is_valid'][0]))


def _draw_rectangles(im, gt, gt_is_valid=None, pred=None):
    im = tf.convert_to_tensor(im)
    if im.dtype != tf.float32:
        im = tf.image.convert_image_dtype(im, tf.float32)
    if gt_is_valid is not None:
        gt = tf.where(gt_is_valid, gt, tf.zeros_like(gt) + geom.unit_rect())
    boxes = [gt]
    if pred is not None:
        boxes.append(pred)
    boxes = list(map(geom.rect_to_tf_box, boxes))
    im = tf.image.draw_bounding_boxes(im, tf.stack(boxes, axis=1))
    return tf.image.convert_image_dtype(im, tf.uint8, saturate=True)


def _evaluate_at_existing_checkpoints(saver, eval_sets, sess, model_inst,
                                      path_ckpt, period_skip, period_assess, extra_assess,
                                      **kwargs):
    '''
    Args:
        kwargs: For _evaluate()
    '''
    # TODO: Add option to raise exception if not all requested checkpoints exist.

    metrics = {}
    # Identify which checkpoints are available.
    state = tf.train.get_checkpoint_state(path_ckpt)
    logger.debug('model files: %s', state.all_model_checkpoint_paths)
    model_files = {_index_from_checkpoint(os.path.basename(s)): s
                   for s in state.all_model_checkpoint_paths}
    # Identify which of these satisfy conditions.
    subset = sorted([index for index in model_files
                     if (index in extra_assess or
                         (index >= period_skip and index % period_assess == 0))])
    logger.debug('global steps to assess: %s', subset)
    # Evaluate each (with cache).
    for global_step in subset:
        saver.restore(sess, model_files[global_step])
        for eval_id, sampler in eval_sets.items():
            eval_sequences = sampler()
            result = _evaluate(global_step, eval_id, sess, model_inst, eval_sequences, **kwargs)
            metrics.setdefault(global_step, {})[eval_id] = result
    return metrics


def _index_from_checkpoint(s):
    parts = s.split('-')
    assert len(parts) == 2
    return int(parts[1])


def _evaluate(
        global_step, eval_id, sess, model_inst, eval_sequences,
        # Evaluation args:
        eval_tre_num,
        # Args that do not affect result:
        path_output,
        visualize,
        keep_frames):
    iter_id = 'iteration{}'.format(global_step)
    vis_dir = os.path.join(path_output, iter_id, eval_id)
    if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0o755)
    # visualizer = visualize.VideoFileWriter(vis_dir)
    # Run the tracker on a full epoch.
    print('evaluation: {}'.format(eval_id))
    # Cache the results.
    result_file = os.path.join(path_output, 'assess', eval_id, iter_id + '.json')
    result = helpers.cache_json(
        result_file,
        lambda: track.track_model(
            sess, model_inst, eval_sequences,
            visualize=visualize, vis_dir=vis_dir, keep_frames=keep_frames,
            tre_num=eval_tre_num),
        makedir=True)

    modes = ['OPE', 'TRE_{}'.format(eval_tre_num)]
    metric_keys = ['iou_seq_mean', 'iou_frame_mean', 'iou_success_0.5']
    for mode in modes:
        for metric_key in metric_keys:
            full_key = mode + '_' + metric_key
            var_key = full_key + '_var'
            if full_key in result:
                value = '{:.3g}'.format(result[full_key])
                if var_key in result:
                    value += ' (1.96 sigma = {:.3g})'.format(1.96 * math.sqrt(result[var_key]))
            else:
                value = '--'
            print('{} {}: {}'.format(mode, metric_key, value))
        # print 'mode {}: IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}, Prec.@20px: {:.3f}'.format(
        #     mode, result[mode]['iou_mean'], result[mode]['auc'],
        #     result[mode]['cle_mean'], result[mode]['cle_representative'])

    return result


def _is_pair(x):
    return (isinstance(x, list) or isinstance(x, tuple)) and len(x) == 2


def summarize_trials(trial_metrics, val_dataset, sort_key):
    '''Takes the mean over multiple trials of the best checkpoint.

    Args:
        trial_metrics: List of dicts, each of which is the result of train().
            trial_metrics[trial][step][dataset]
        sort_key: The key for comparing two metric dictionaries.
            The maximum sort key will be chosen.
    '''
    num_trials = len(trial_metrics)
    # For each training run, choose the iteration which gave the best performance.
    best = [max(trial_metrics[trial].values(), key=lambda x: sort_key(x[val_dataset]))
            for trial in range(num_trials)]
    # Compute the mean of each metric
    # and the variance of a metric if there is enough information.
    # Take union of metrics from all trials (should all be identical).
    datasets = set(dataset for trial in range(num_trials) for dataset in best[trial].keys())

    metrics = {}
    for dataset in datasets:
        # Take union of metric keys across trials (should be identical).
        keys = set(key for trial in range(num_trials) for key in best[trial][dataset].keys())
        # If there exists a metric xxx and xxx_var, then remove xxx_var from the list.
        basic_keys = keys.difference(set(key + '_var' for key in keys))
        for key in basic_keys:
            metrics[dataset + '_' + key] = np.mean(
                [best[trial][dataset][key] for trial in range(num_trials)])
            if key + '_var' in keys:
                # Use variance of means plus mean of variances.
                # This assumes that each metric (which has a variance) is a mean.
                metrics[dataset + '_' + key + '_var'] = (
                    np.mean([best[trial][dataset][key + '_var'] for trial in range(num_trials)]) +
                    np.var([best[trial][dataset][key] for trial in range(num_trials)]))
            else:
                # TODO: Could report simple variance across trials if xxx_var is not present?
                # However, this might be confusing because then some variances are more correct.
                pass

    return metrics


def combine_dicts(delim='/', **kwargs):
    together = {}
    for name, d in kwargs.items():
        together.update({name + delim + k: v for k, v in d.items()})
    return together


def _to_assess_step(step, period_assess, period_skip, extra_assess):
    return (step in extra_assess or
            (step > 0 and step >= period_skip and step % period_assess == 0))
