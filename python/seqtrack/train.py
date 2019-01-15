from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import csv
import json
import math
import numpy as np
import time
import os
import pprint
import random
import re
import six
import subprocess
from functools import partial
from itertools import chain

import tensorflow as tf
from tensorflow.python import debug as tf_debug
nest = tf.contrib.framework.nest

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import track
from seqtrack import geom
from seqtrack import helpers
from seqtrack import motion
from seqtrack import sample
from seqtrack.models import itermodel
from seqtrack.models import siamfc

NUM_PREFETCH = 8
INSTANTIATE_METHOD_TRACK = 'assign'


# Pickling a function in one module to load in another module does not work if
# the function is defined in the same module.
# Therefore provide a worker function here that takes tmp_dir from the mapper.
def train_worker(context, kwargs, train_dir='train'):
    # The mapper context provides a temporary directory and a name.
    return train(dir=os.path.join(train_dir, context.name),
                 summary_name=context.name,
                 tmp_dir=context.tmp_dir(),
                 **kwargs)


def train_worker_params(context, params_dict, kwargs, train_dir='train'):
    kwargs = dict(kwargs)
    kwargs['params_dict'] = params_dict
    return train_worker(context, kwargs, train_dir=train_dir)


class TrainParams(object):
    '''Collects the "mathematical" training arguments.

    That is, the arguments which describe the training procedure, not the system configuration.
    These are the parameters which may be optimized in hyper-param search.

    This kwargs structure will be flattened using `nest.flatten_with_joined_string_paths`.
    If the values are lists/tuples/namedtuples, these will be flattened.
    '''

    def __init__(self, **kwargs):
        params = dict(
            model_params=None,
            seed=0,
            # Dataset:
            train_dataset=None,
            val_dataset=None,
            eval_datasets=None,  # TODO: Remove from TrainParams?
            pool_datasets=None,
            pool_split=None,
            # Sampling:
            sampler='uniform',
            sampler_params=None,
            augment_motion=False,
            motion_params=None,
            # Evaluation:
            eval_tre_num=None,
            max_eval_videos=None,
            # Training process:
            ntimesteps=None,
            metrics_resolution=1000,
            period_assess=40000,
            extra_assess=None,
            period_skip=0,
            # Training args:
            num_steps=None,
            batchsz=None,
            imwidth=None,
            imheight=None,
            preproc_id=None,
            resize_online=False,  # Tied to `preproc_id`.
            resize_method='bilinear',
            lr_schedule='constant',
            lr_init=1e-3,
            lr_params=None,
            optimizer=None,
            optimizer_params=None,
            grad_clip=False,
            grad_clip_params=None,
            siamese_pretrain=None,
            siamese_model_file=None,
            use_gt_train=True,
            gt_decay_rate=1,
            min_gt_ratio=0,
        )

        helpers.update_existing_keys(params, kwargs)
        for key, value in params.items():
            setattr(self, key, value)


# class TrainConfig(object):
#     '''System configuration for training.'''
# 
#     def __init__(self, **kwargs):
#         params = dict(
#             dir=None,
#             tmp_dir=None,
#             only_evaluate_existing=False,
#             untar=False,
#             data_dir=None,
#             use_tmp_data_dir=False,
#             data_cache_dir=None,
#             resume=False,
#             tfdb=False,
#             use_queues=True,
#             summary_dir=None,
#             summary_name=None,
#             nosave=False,
#             period_summary=100,
#             period_preview=1000,
#             verbose_train=False,
#             visualize=False,
#             keep_frames=False,
#             session_config_kwargs=None,
#         )
# 
#         helpers.update_existing_keys(params, kwargs)
#         for key, value in params.items():
#             setattr(self, key, value)


def train(
        params_dict=None,
        dir=None,
        tmp_dir=None,
        # only_evaluate_existing=False,
        data_dir=None,
        use_tmp_data_dir=False,
        untar=False,
        tar_dir=None,
        data_cache_dir=None,
        # resume=False,
        # tfdb=False,
        # use_queues=True,
        # summary_dir=None,
        # summary_name=None,
        # nosave=False,
        # period_ckpt=10000,
        # period_summary=100,
        # period_preview=1000,
        # verbose_train=False,
        # visualize=False,
        # keep_frames=False,
        # session_config_kwargs=None,
        **kwargs):
    '''
    All arguments should be JSON-serializable to facilitate RPC (for experiments).

    Args:
        params_dict: Dict of kwargs for TrainParams().
        kwargs: For train_model_data().

    Specify either `data_dir` or `use_tmp_data_dir`.
    The option `use_tmp_data_dir` implies `untar`.
    It is possible to specify `untar` without `use_tmp_data_dir` to untar to `data_dir`.

    Side effects:
        Resets global random seed.

    Writes output/{model_params.json,train_series.csv,track_series.csv}.
    '''
    assert params_dict is not None
    params = TrainParams(**params_dict)

    helpers.mkdir_p(os.path.join(dir, 'output'))
    with open(os.path.join(dir, 'output', 'model_params.json'), 'w') as f:
        json.dump(params.model_params, f)

    tf.reset_default_graph()
    _set_global_seed(params.seed)  # Caution: Global side effects!

    # # TODO: Move? Modifying params is ugly?
    # if only_evaluate_existing:
    #     params.train_dataset = None
    #     params.val_dataset = None

    if use_tmp_data_dir:
        untar = True
        data_dir = os.path.join(tmp_dir, 'data')

    # TODO: How to get datasets from train_dataset and eval_datasets?
    dataset_names = list(set(chain(_datasets_in_sampler(params.train_dataset),
                                   _datasets_in_sampler(params.val_dataset),
                                   params.eval_datasets)))
    # TODO: Flag to enable/disable.
    datasets = setup_data(
        dataset_names=dataset_names,
        pool_datasets=params.pool_datasets,
        pool_split=params.pool_split,
        data_dir=data_dir,
        untar=untar,
        tar_dir=tar_dir,
        preproc_id=params.preproc_id,
        data_cache_dir=data_cache_dir)

    create_iter_model_fn = partial(siamfc.SiamFC, params=params.model_params)
    # model_properties = iter_model_fn.derived_properties()

    TRAIN_VAL = ('train', 'val')
    video_sampler_specs = {'train': params.train_dataset, 'val': params.val_dataset}
    example_streams = {
        k: _make_example_stream(video_sampler_spec=video_sampler_specs[k],
                                datasets=datasets,
                                example_sampler=params.sampler,
                                example_sampler_params=params.sampler_params,
                                seed=params.seed)
        for k in TRAIN_VAL}
    eval_sample_fns = {
        dataset_name: functools.partial(_make_sequence_stream,
                                        datasets[dataset_name],
                                        params.seed,
                                        params.max_eval_videos)
        for dataset_name in params.eval_datasets}

    train_series, track_series = train_model_data(
        params, dir, create_iter_model_fn, example_streams, eval_sample_fns,
        # only_evaluate_existing=only_evaluate_existing,
        **kwargs)

    result = make_train_result(
        model_properties={},
        # model_properties=model_properties,
        train_series=train_series,
        track_series=track_series,
    )

    # Dump time-series for training and tracking for each seed.
    with open(os.path.join(dir, 'output', 'train_series.csv'), 'w') as f:
        helpers.dump_csv(f, _flatten_dicts(result['train_series']))
    with open(os.path.join(dir, 'output', 'track_series.csv'), 'w') as f:
        helpers.dump_csv(f, _flatten_dicts(result['track_series']))

    return result


def _make_sequence_stream(dataset, seed, max_num):
    video_sampler = sample.EpochSampler(datasets[names], rand=rand, repeat=False)
    track_ids = video_sampler.sample()
    if max_num is not None and max_num > 0:
        track_ids = itertools.islice(track_ids, max_num)
    for track_id in track_ids:
        yield sample.extract_sequence_from_dataset(dataset, track_id)


def _make_example_stream(video_sampler_spec,
                         datasets,
                         example_sampler,
                         example_sampler_params,
                         seed):
    video_sampler = _make_video_sampler(video_sampler_spec, datasets, params.seed, repeat=True)
    sampler_dataset = video_sampler.dataset()

    rand = np.random.RandomState(seed)
    example_fn = partial(sample.EXAMPLE_FNS[example_sampler], **example_sampler_params)
    for track_id in video_sampler.sample():
        seq = sample.extract_sequence_from_dataset(sampler_dataset, track_id)
        yield example_fn(rand, seq)


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


def _set_global_seed(seed):
    # TODO: Not sure if TensorFlow initialisation is deterministic given global seed?
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _datasets_in_sampler(names):
    if names is None:
        return []
    if isinstance(names, six.string_types):
        return [names]
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, six.string_types) for name in names):
            return names
        elif all(_is_pair(elem) for elem in names):
            return [name for _, name in names]
    raise ValueError('invalid structure: {}'.format(repr(names)))


def _make_video_sampler(names, datasets, seed, repeat):
    '''
    Args:
        names: string (EpochSampler), list of strings (EpochSampler of Concat) or
            list of float-string-pairs (MixtureSampler).
    '''
    rand = np.random.RandomState(seed)
    if isinstance(names, six.string_types):
        return sample.EpochSampler(datasets[names], rand=rand, repeat=repeat)
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, six.string_types) for name in names):
            concat = data.Concat({name: datasets[name] for name in names})
            return sample.EpochSampler(concat, rand=rand, repeat=repeat)
        elif all(_is_pair(elem) for elem in names):
            samplers = {
                name: sample.EpochSampler(datasets[name], rand=rand, repeat=True)
                for _, name in names}
            weights = {name: weight for weight, name in names}
            return sample.MixtureSampler(samplers, weights)
    raise ValueError('invalid structure: {}'.format(repr(names)))


# TODO: Look at resume and seeds.

def train_model_data(
        params,
        dir, create_iter_model_fn, sequences, eval_sets,
        # override_ckpt_dir=None,
        # only_evaluate_existing=False,
        resume=False,
        tfdb=False,
        use_queues=True,
        summary_dir=None,
        summary_name=None,
        nosave=False,
        period_ckpt=10000,
        period_summary=100,
        period_preview=1000,
        verbose_train=False,
        visualize=False,
        keep_frames=False,
        session_config_kwargs=None,
        ):
    '''Trains a network.

    Args:
        create_iter_model_fn: (models.interface.IterModel) Maps example to outputs.
        datasets: Dictionary of datasets with keys 'train' and 'val'.
        eval_sets: A dictionary of sampling functions which return collections
            of sequences on which to evaluate the tracker.

    Returns:
        Tuple of (train_series, track_series).
        train_series[step][metric]
        track_series[step][dataset][metric]

    The method model_fn.instantiate() constructs the model in the tf graph.
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
    session_config_kwargs = session_config_kwargs or {}

    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    # path_ckpt = override_ckpt_dir or os.path.join(dir, 'ckpt')
    path_ckpt = os.path.join(dir, 'ckpt')
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
    #
    # Update (2018-12):
    # Replace explicit queues with tf.data.Dataset and prefetch().
    # Use separate instance (with shared variables) for tracking.

    modes = ['train', 'val']

    datasets = {
        mode: (
            _dataset_from_sequence_generator(partial(_identity, sequences[mode]),
                                             sequence_len=params.ntimesteps + 1)
            .map(partial(_load_images_unroll,
                         resize=params.resize_online,
                         size=(params.imheight, params.imwidth),
                         method=params.resize_method))
            .batch(params.batchsz, drop_remainder=True)
        ) for mode in modes
    }
    if use_queues:
        datasets = {mode: datasets[mode].prefetch(NUM_PREFETCH) for mode in modes}
    iterators = {mode: datasets[mode].make_one_shot_iterator() for mode in modes}

    # https://www.tensorflow.org/guide/datasets
    switch_handle = tf.placeholder(tf.string, shape=[])
    switch_iterator = tf.data.Iterator.from_string_handle(
        switch_handle,
        output_types=datasets['train'].output_types,
        output_shapes=datasets['train'].output_shapes)
    example = switch_iterator.get_next()
    run_opts = _make_run_opts_placeholders()

    # example = _perform_color_augmentation(example, args)
    metric_vars = {}

    iter_model_fn = create_iter_model_fn(mode=tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope('model', reuse=False) as vs:
        ops = itermodel.instantiate_unroll(iter_model_fn, example, run_opts=run_opts, scope=vs)
        # outputs, losses, init_state, final_state = model_fn.instantiate(
        #     example_input, run_opts, enable_loss=True,
        #     image_summaries_collections=['IMAGE_SUMMARIES'])
    metric_vars.update(ops.losses)
    _loss_summary(ops.losses)
    # _image_summary(example, outputs)
    # loss_var = _add_losses(losses, args.loss_coeffs)
    loss_var = _add_losses(ops.losses, {})

    with tf.name_scope('diagnostic'):
        iou = geom.rect_iou(ops.predictions['rect'], example.labels['rect'])
        mean_iou = tf.reduce_mean(tf.boolean_mask(iou, example.labels['valid']))
        metric_vars['iou'] = mean_iou
        tf.summary.scalar('iou', mean_iou)

        dist = tf.norm((geom.rect_center(ops.predictions['rect']) -
                        geom.rect_center(example.labels['rect'])), axis=-1)
        dist = dist / tf.reduce_mean(geom.rect_size(example.labels['rect']), axis=-1)
        mean_dist = tf.reduce_mean(tf.boolean_mask(dist, example.labels['valid']))
        tf.summary.scalar('dist', mean_dist)
        metric_vars['dist'] = mean_dist

    r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_var += r
    metric_vars['loss'] = loss_var
    tf.summary.scalar('regularization', r)
    tf.summary.scalar('total', loss_var)

    global_step_var = tf.Variable(0, name='global_step', trainable=False)

    lr = make_learning_rate(global_step_var, params.num_steps, params.lr_init,
                            schedule=params.lr_schedule, schedule_params=params.lr_params)
    tf.summary.scalar('lr', lr, collections=['summaries_train'])
    optimizer_obj = make_optimizer(lr, params.optimizer, **(params.optimizer_params or {}))
    grads_and_vars = optimizer_obj.compute_gradients(loss_var)
    if params.grad_clip:
        grads_and_vars = clip_gradients(grads_and_vars, **(params.grad_clip_params or {}))
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

    model_fn_track = create_iter_model_fn(mode=tf.estimator.ModeKeys.PREDICT)
    # TODO: Ensure that the `track` instance does not include ops such as bnorm and summaries.
    # Feed image files here:
    example_track_with_files = _make_iter_placeholders(batch_size=1)
    # Evaluate model_fn on these features:
    example_track = _load_images_iter(example_track_with_files)
    if INSTANTIATE_METHOD_TRACK == 'assign':
        # Create external scope for the local variables (reuse is false).
        with tf.variable_scope('instance') as local_scope:
            pass
        with tf.variable_scope('model', reuse=True) as scope:
            model_inst_track = itermodel.TrackerAssign(
                example_track_with_files,
                itermodel.instantiate_iter_assign(model_fn_track, example_track,
                                                  run_opts=_make_run_opts_tracking(),
                                                  local_scope=local_scope, scope=scope))
    else:
        with tf.variable_scope('model', reuse=True) as scope:
            model_inst_track = itermodel.TrackerFeed(
                example_track_with_files,
                itermodel.instantiate_iter_feed(model_fn_track, example_track,
                                                run_opts=_make_run_opts_tracking(),
                                                scope=scope))

    init_op = [
        tf.initializers.global_variables(),
        tf.initializers.local_variables(),
        # [iterator.initializer for iterator in iterators.values()],
    ]
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

        iterator_handles = {mode: sess.run(iterators[mode].string_handle()) for mode in modes}

        # if only_evaluate_existing:
        #     return _evaluate_at_existing_checkpoints(
        #         saver, eval_sets, sess, model_inst,
        #         path_ckpt=path_ckpt,
        #         period_skip=period_skip,
        #         period_assess=period_assess,
        #         extra_assess=extra_assess,
        #         # For _evaluate():
        #         eval_tre_num=eval_tre_num,
        #         path_output=path_output,
        #         visualize=visualize,
        #         keep_frames=keep_frames)

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
            # TODO: This method should belong to the instance?
            # model_fn.init(sess)  # Loads pre-trained features if applicable.

            # if args.siamese_pretrain:
            #     saver_siamese.restore(sess, args.siamese_model_file)
            #     # vars_uninit = sess.run(tf.report_uninitialized_variables())
            #     # print 'vars_uninit:'
            #     # pprint.pprint(vars_uninit)
            #     # sess.run(tf.variables_initializer([v for v in tf.global_variables()
            #     #                                    if v.name.split(':')[0] in vars_uninit]))
            #     # assert len(sess.run(tf.report_uninitialized_variables())) == 0

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
            assess_step = _to_assess_step(
                global_step, params.period_assess, params.period_skip, params.extra_assess)
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
                        global_step, eval_id, sess, model_inst_track, eval_sequences,
                        eval_tre_num=params.eval_tre_num,
                        path_output=path_output,
                        visualize=visualize,
                        keep_frames=keep_frames)
                    track_series.setdefault(global_step, {})[eval_id] = result

            if global_step > 0 and global_step % params.metrics_resolution == 0:
                train_series[global_step] = combine_dicts(train=train_accum.flush(),
                                                          val=val_accum.flush())

            if global_step >= params.num_steps:
                break

            if params.use_gt_train:
                # TODO: Use a different base!
                gt_ratio = max(1.0 * np.exp(-params.gt_decay_rate * global_step),
                               params.min_gt_ratio)
            else:
                gt_ratio = 0

            # Take a training step.
            start = time.time()
            feed_dict = {run_opts['use_gt']: params.use_gt_train,
                         run_opts['is_training']: True,
                         run_opts['is_tracking']: False,
                         run_opts['gt_ratio']: gt_ratio,
                         switch_handle: iterator_handles['train']}
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
                feed_dict = {run_opts['use_gt']: params.use_gt_train,  # Match training.
                             run_opts['is_training']: False,  # Do not update bnorm stats.
                             run_opts['is_tracking']: False,
                             run_opts['gt_ratio']: gt_ratio,  # Match training.
                             switch_handle: iterator_handles['val']}
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
    '''
    Returns:
        Dict of metrics for dataset assessment.

    Result is cached in '{path_output}/assess/{eval_id}/{iter_id}.json'.

    Writes per-sequence assessments to '{path_output}/assess_per_sequence/{eval_id}/{iter_id}.csv'.
    '''
    iter_id = 'iteration{}'.format(global_step)
    vis_dir = os.path.join(path_output, 'vis', eval_id, iter_id)
    # if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0o755)
    # visualizer = visualize.VideoFileWriter(vis_dir)
    # Run the tracker on a full epoch.
    print('evaluation: {}'.format(eval_id))
    # Cache the results.
    result_file = os.path.join(path_output, 'assess', eval_id, iter_id + '.json')
    # Dump the per-sequence results to a file.
    report_file = os.path.join(path_output, 'assess_per_sequence', eval_id, iter_id + '.csv')

    def _eval_and_dump_report():
        dataset_metrics, sequence_metrics = track.track_and_assess(
            sess, model_inst, eval_sequences,
            visualize=visualize, vis_dir=vis_dir, keep_frames=keep_frames,
            tre_num=eval_tre_num)
        helpers.mkdir_p(os.path.dirname(report_file))
        with open(report_file, 'w') as f:
            helpers.dump_csv(f, sequence_metrics)
        return dataset_metrics
    result = helpers.cache_json(result_file, _eval_and_dump_report, makedir=True)

    # Print results.
    modes = ['OPE']
    if eval_tre_num and eval_tre_num > 1:
        modes.append('TRE_{}'.format(eval_tre_num))
    metric_keys = ['iou_seq_mean', 'iou_success_0.5']
    for mode in modes:
        for metric_key in metric_keys:
            full_key = mode + '/' + metric_key
            var_key = full_key + '_var'
            if full_key in result:
                value = '{:.3g}'.format(result[full_key])
                if var_key in result:
                    value += ' (1.96 sigma = {:.3g})'.format(1.96 * math.sqrt(result[var_key]))
            else:
                value = '--'
            print('{} {}: {}'.format(mode, metric_key, value))

    return result



def _is_pair(x):
    return (isinstance(x, list) or isinstance(x, tuple)) and len(x) == 2


def summarize_trials(trial_metrics, val_dataset, sort_key):
    '''Takes the mean over multiple trials of the best checkpoint.

    Args:
        trial_metrics: List of dicts, each of which is the result of train().
            trial_metrics[trial]['track_series'][step][dataset]
        sort_key: The key for comparing two metric dictionaries.
            The maximum sort key will be chosen.
    '''
    num_trials = len(trial_metrics)
    # Take just the time-series of tracking data for each.
    # For each training run, choose the iteration which gave the best performance.
    best_step = [find_arg_max(trial_metrics[i]['track_series'],
                              key=lambda x: sort_key(x[val_dataset]))
                 for i in range(num_trials)]
    best = [trial_metrics[i]['track_series'][best_step[i]] for i in range(num_trials)]
    # Flatten all datasets into one dictionary.
    best = [dict(nest.flatten_with_joined_string_paths(best[i])) for i in range(num_trials)]
    # Augment with arg max.
    for i in range(num_trials):
        assert 'step' not in best[i]
        # best[i] = dict(best[i])
        best[i]['step'] = best_step[i]

    # Compute the mean of each metric
    # and the variance of a metric if there is enough information.
    # Take union of metrics from all trials (should all be identical).
    metrics = {}
    # Take union of metric keys across trials (should be identical).
    keys = set(key for i in range(num_trials) for key in best[i].keys())
    # If there exists a metric xxx and xxx_var, then remove xxx_var from the list.
    basic_keys = keys.difference(set(key + '_var' for key in keys))
    for key in basic_keys:
        metrics[key] = np.mean([best[i][key] for i in range(num_trials)])
        if key + '_var' in keys:
            # Use variance of means plus mean of variances.
            # This assumes that each metric (which has a variance) is a mean.
            metrics[key + '_var'] = (
                np.mean([best[i][key + '_var'] for i in range(num_trials)]) +
                np.var([best[i][key] for i in range(num_trials)]))
        else:
            # TODO: Could report simple variance across trials if xxx_var is not present?
            # However, this might be confusing because then some variances are more correct.
            pass

    return metrics


def find_arg_max(series, key):
    keys = list(series.keys())
    return max(keys, key=lambda x: key(series[x]))
    # # TODO: Use OrderedDict and put 'arg' first?
    # result = dict(series[arg])
    # assert 'arg' not in result
    # result['arg'] = arg
    # return result


def combine_dicts(delim='/', **kwargs):
    together = {}
    for name, d in kwargs.items():
        together.update({name + delim + k: v for k, v in d.items()})
    return together


def _to_assess_step(step, period_assess, period_skip, extra_assess):
    extra_assess = extra_assess or []
    return (step in extra_assess or
            (step > 0 and step >= period_skip and step % period_assess == 0))


def _make_iter_placeholders(batch_size=None):
    with tf.name_scope('features_init'):
        features_init = {
            'image': {
                'file': tf.placeholder(tf.string, [batch_size], name='image_file'),
            },
            'aspect': tf.placeholder(tf.float32, [batch_size], name='aspect'),
            'rect': tf.placeholder(tf.float32, [batch_size, 4], name='rect'),
        }
    with tf.name_scope('features_curr'):
        features_curr = {
            'image': {
                'file': tf.placeholder(tf.string, [batch_size], name='image_file'),
            }
        }
        labels_curr = {
            'valid': tf.placeholder(tf.bool, [batch_size], name='valid'),
            'rect': tf.placeholder(tf.float32, [batch_size, 4], name='rect'),
        }
    return itermodel.ExampleIter(
        features_init=features_init,
        features_curr=features_curr,
        labels_curr=labels_curr,
    )


def _make_run_opts_placeholders():
    return {
        'use_gt': tf.placeholder(tf.bool, [], name='use_gt'),
        'is_training': tf.placeholder(tf.bool, [], name='is_training'),
        'is_tracking': tf.placeholder(tf.bool, [], name='is_tracking'),
        'gt_ratio': tf.placeholder(tf.float32, [], name='gt_ratio'),
    }


def _make_run_opts_tracking():
    return {
        'use_gt': tf.constant(False),
        'is_training': tf.constant(False),
        'is_tracking': tf.constant(True),
        'gt_ratio': tf.constant(np.nan),  # Not used during tracking.
    }


def _dataset_from_sequence_generator(sequences, sequence_len):
    '''
    Args:
        sequences: Callable that returns iterable (like `from_generator`).
        sequence_len: Dimension of tensors (int or None).

    Returns:
        tf.data.Dataset
    '''
    types = {
        'image_files': tf.string,
        'labels': tf.float32,
        'label_is_valid': tf.bool,
        'aspect': tf.float32,
        'video_name': tf.string,
    }
    shapes = {
        'image_files': [sequence_len],
        'labels': [sequence_len, 4],
        'label_is_valid': [sequence_len],
        'aspect': [],
        'video_name': [],
    }
    return tf.data.Dataset.from_generator(sequences, types, output_shapes=shapes)


def _sequence_to_example_unroll(sequence):
    '''
    Args:
        sequence: Dict (defined in `sample`)

    Returns:
        itermodel.ExampleUnroll
    '''
    # TODO: Assert `sequence['label_is_valid'][0]`?
    return itermodel.ExampleUnroll(
        features_init={
            'image': {'file': sequence['image_files'][0]},
            'aspect': sequence['aspect'],
            'rect': sequence['labels'][0],
        },
        features={
            'image': {'file': sequence['image_files'][1:]},
        },
        labels={
            'valid': sequence['label_is_valid'][1:],
            'rect': sequence['labels'][1:],
        },
    )


def _load_images_unroll(with_files, **kwargs):
    '''
    Args:
        with_files: itermodel.ExampleUnroll

    Returns:
        itermodel.ExampleUnroll
    '''
    # Load init image.
    features_init = dict(with_files.features_init)
    features_init['image'] = {
        'data': load_and_resize_images(features_init['image']['file'], **kwargs)
    }
    # Load curr image.
    features = dict(with_files.features)
    features['image'] = {
        'data': load_and_resize_images(features['image']['file'], **kwargs)
    }
    return itermodel.ExampleUnroll(features_init, features, with_files.labels)


def _load_images_iter(with_files, **kwargs):
    '''
    Args:
        with_files: itermodel.ExampleIter

    Returns:
        itermodel.ExampleIter
    '''
    # Load init image.
    features_init = dict(with_files.features_init)
    features_init['image'] = {
        'data': load_and_resize_images(features_init['image']['file'], **kwargs)
    }
    # Load curr image.
    features_curr = dict(with_files.features_curr)
    features_curr['image'] = {
        'data': load_and_resize_images(features_curr['image']['file'], **kwargs)
    }
    return itermodel.ExampleIter(features_init, features_curr, with_files.labels_curr)


def load_and_resize_images(image_files, resize=False, size=None, method='bilinear',
                           name='load_and_resize_images'):
    '''
    Args:
        image_files: Tensor of type string with shape `[k[0], ..., k[n-1]]`.
        size: Tuple (h, w).

    If `resize` is true, images will be resized to this `size`.
    If `resize` is false and `size` is specified, the shape will be set with `set_shape`.
    Otherwise the shape of the image tensor will be left unspecified.

    Returns:
        Tensor of type uint8 with shape `[k[0], ..., k[n-1], h, w, 3]`.

    If `resize` is false, then the image files must be the same size.
    '''
    with tf.name_scope(name) as scope:
        image_files, restore_fn = helpers.merge_dims(image_files, None, None)
        images = tf.map_fn(
            partial(_load_and_resize_image, resize=resize, size=size, method=method),
            image_files,
            dtype=tf.float32)
        images = restore_fn(images, 0)
        return images


def _load_and_resize_image(image_file, resize=False, size=None, method='bilinear',
                           name='_load_and_resize_image'):
    '''Loads a single image file and performs optional resize.

    Args:
        image_file: Tensor with shape [].
    '''
    with tf.name_scope(name) as scope:
        with tf.device('/cpu:0'):
            file_contents = tf.read_file(image_file)
            image = tf.image.decode_jpeg(file_contents, channels=3)
            # The function `resize_images` will convert to float32 (unless using nearest).
            image = tf.image.convert_image_dtype(image, tf.float32)
            if resize:
                method = getattr(tf.image.ResizeMethod, method.upper())
                image = tf.image.resize_images(image, size, method=method)
            else:
                if size:
                    image.set_shape(list(size) + [3])
            return image


def _flatten_dicts(series):
    return {k: dict(nest.flatten_with_joined_string_paths(v)) for k, v in series.items()}


def _identity(x):
    return x
