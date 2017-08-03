import sys
import csv
import functools
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import debug as tf_debug
import time
import os
import random
import re
import subprocess
import threading

import draw
import evaluate
import geom
import helpers
import pipeline
import sample
import visualize

from model import convert_rec_to_heatmap
from helpers import load_image, im_to_arr, pad_to, cache_json, merge_dims

EXAMPLE_KEYS = ['x0', 'y0', 'x', 'y', 'y_is_valid']
SUMMARIES_IMAGES = 'summaries_images'


def train(create_model, train_distribution, val_datasets, eval_sets, o, use_queues=False):
    '''Trains a network.

    Args:
        model: Function that takes as input a dictionary of tensors and
            returns a model object.
        train_distribution: List that contains tuples of
            (mixture weight, dataset sampler).
        datasets: Dictionary of datasets with keys 'train' and 'val'.
        eval_sets: A dictionary of sampling functions which return collections
            of sequences on which to evaluate the tracker.

    Returns:
        The results obtained using the tracker at different stages of training.
        This is a dictionary with the same keys as eval_sets:
            all_results[val_set] = list of (iter_num, results)
        Note that this is represented as a list of pairs instead of a dictionary
        to facilitate saving to JSON (does not support integer keys).

    The reason that the model is provided as a *function* is so that
    the code which uses the model is free to decide how to instantiate it.
    For example, training code may construct a single instance of the model with input placeholders,
    or it may construct two instances of the model, each with its own input queue.

    The model should use `tf.get_variable` rather than `tf.Variable` to facilitate variable sharing between multiple instances.
    The model will be used in the same manner as an input to `tf.make_template`.

    The input dictionary has fields::

        'x0'         # First image in sequence, shape [b, h, w, 3]
        'y0'         # Position of target in first image, shape [b, 4]
        'x'          # Input images, shape [b, n, h, w, 3]
        'y_is_valid' # Booleans indicating presence of frame, shape [b, n]

    and the output dictionary has fields::

        'y'       # (optional) Predicted position of target in each frame, shape [b, n, 4]
        'heatmap' # (optional) Score for pixel belonging to target, shape [b, n, h, w, 1]

    The images provided to the model are already normalized (e.g. dataset mean subtracted).

    Each sampler in the eval_sets dictionary is a function that
    returns a collection of sequences or is a finite generator of sequences.
    '''

    # How should we compute training and validation error with pipelines?
    # Option 1 is to have multiple copies of the network with shared parameters.
    # However, this makes it difficult to plot training and validation error on the same axes
    # since there are separate summary ops for training and validation (with different names).
    # Option 2 is to use FIFOQueue.from_list()

    assert 'train' in datasets
    val_datasets = {dataset for dataset in datasets if dataset != 'train'}
    if 'val' in val_datasets:
        val_order = ['val'] + sorted(dataset for dataset in val_datasets if dataset != 'val')
    else:
        val_order = sorted(val_datasets)
    dataset_order = ['train'] + val_order
    dataset_index = {name: i for i, name in enumerate(dataset_order)}

    feed_loop = {} # Each value is a function to call in a thread.
    with tf.name_scope('input'):
        from_queue = None
        if use_queues:
            queues = []
            for dataset in dataset_order:
                # Create a queue each for training and validation data.
                queue, feed_loop[dataset] = _make_input_pipeline(o,
                    num_load_threads=1, num_batch_threads=1, name='pipeline_'+dataset)
                queues.append(queue)
            queue_index, from_queue = pipeline.make_multiplexer(queues,
                capacity=4, num_threads=1)
        example = _make_example_placeholders(default=from_queue,
            ntimesteps=o.ntimesteps, frmsz=o.frmsz, dtype=o.dtype)
        run_opts = _make_option_placeholders()

    # Always use same statistics for whitening (not set dependent).
    # stat = datasets['train'].stat
    stat = {'mean': 0.5, 'std': 1.0}

    example_pre_aug = dict(example)
    # Sample a training sequence from the augmentation distribution.
    example = _perform_data_augmentation(example, o, pad_value=stat['mean'])
    example_post_aug = dict(example)

    model = create_model(stat=stat,
                         image_summaries_collections=[SUMMARIES_IMAGES])

    prediction_crop, window, prediction, init_state, final_state = process_sequence(
        example, run_opts, model,
        batchsz=o.batchsz, ntimesteps=o.ntimesteps, frmsz=o.frmsz, dtype=o.dtype,
    )
    # Crop ground truth label for loss.
    example_crop = geom.crop_example(example, window,
        crop_size=[o.frmsz, o.frmsz],
        pad_value=stat['mean'],
    )

    eval_model = evaluate.Model(
        batch_size=o.batchsz,
        image_size=(o.frmsz, o.frmsz),
        sequence_len=o.ntimesteps,
        example=example_post_aug,
        run_opts=run_opts,
        window=window,
        prediction_crop=prediction_crop,
        prediction=prediction,
        init_state=init_state,
        final_state=final_state,
    )

    # model = create_model(_whiten(_guard_labels(example, run_opts), dtype=o.dtype, stat=stat),
    #                      image_summaries_collections=[SUMMARIES_IMAGES])

    loss_var = get_loss(example_crop, prediction_crop, o,
                        image_summaries_collections=[SUMMARIES_IMAGES])
    r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('regularization', r)
    loss_var += r
    tf.summary.scalar('total', loss_var)

    # if o.object_centric:
    #     inv_window_rect = geom.crop_inverse(window_rect)
    #     pred_whole = geom.crop_prediction(prediction, inv_window_rect, crop_size=[o.frmsz, o.frmsz])

    # nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = len(datasets['train'].videos)/o.batchsz if not o.debugmode else 30

    global_step_var = tf.Variable(0, name='global_step', trainable=False)
    # lr = init * decay^(step)
    #    = init * decay^(step / period * period / decay_steps)
    #    = init * [decay^(period / decay_steps)]^(step / period)
    lr = tf.train.exponential_decay(o.lr_init, global_step_var,
                                    decay_steps=o.lr_decay_steps,
                                    decay_rate=o.lr_decay_rate,
                                    staircase=True)
    tf.summary.scalar('lr', lr, collections=['summaries_train'])
    optimizer = _get_optimizer(lr, o)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if not o.grad_clip:
            optimize_op = optimizer.minimize(loss_var, global_step=global_step_var)
        else: # Gradient clipping by norm; NOTE: `global graident clipping` may be another correct way.
            gradients, variables = zip(*optimizer.compute_gradients(loss_var))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, o.max_grad_norm)
                         for gradient in gradients]
            optimize_op = optimizer.apply_gradients(zip(gradients, variables),
                                                    global_step=global_step_var)

    summary_vars = {}
    extended_summary_vars = {}
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        # Create a preview and add it to the list of summaries.
        with tf.name_scope('batch'):
            tf.summary.image('0',
                _draw_init_bounding_boxes(example, num_sequences=None),
                max_outputs=1, collections=[SUMMARIES_IMAGES])
        with tf.name_scope('crop'):
            tf.summary.image('1_to_n',
                _draw_bounding_boxes(example_crop, prediction_crop),
                max_outputs=o.ntimesteps, collections=[SUMMARIES_IMAGES])
        with tf.name_scope('image'):
            tf.summary.image('0',
                _draw_init_bounding_boxes(example, num_sequences=1),
                max_outputs=1, collections=[SUMMARIES_IMAGES])
            tf.summary.image('1_to_n',
                _draw_bounding_boxes(example, prediction),
                max_outputs=o.ntimesteps, collections=[SUMMARIES_IMAGES])
        # Produce an image summary of the heatmap.
        if 'hmap' in prediction:
            tf.summary.image('hmap', _draw_heatmap(tf.nn.softmax(prediction['hmap'])[0]),
                max_outputs=o.ntimesteps+1, collections=[SUMMARIES_IMAGES])
        # # Produce an image summary of the LSTM memory states (h or c).
        # if hasattr(model, 'memory'):
        #     for mtype in model.memory.keys():
        #         if model.memory[mtype][0] is not None:
        #             tf.summary.image(mtype, _draw_memory_state(model, mtype),
        #                 max_outputs=o.ntimesteps+1, collections=[SUMMARIES_IMAGES])
        for dataset in datasets:
            with tf.name_scope(dataset):
                # Merge summaries with any that are specific to the mode.
                summaries = (global_summaries + tf.get_collection('summaries_' + dataset))
                summary_vars[dataset] = tf.summary.merge(summaries)
                summaries.extend(tf.get_collection(SUMMARIES_IMAGES))
                extended_summary_vars[dataset] = tf.summary.merge(summaries)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    # Use a separate random number generator for each sampler.
    sequences = {
        dataset: iter_examples(
            train_distribution,
            rand=np.random.RandomState(o.seed_global),
        )
        for dataset in datasets
    }

    if o.curriculum_learning:
        ''' Curriculum learning.
        Restore values of trainable variables from pre-trained model on short sequence,
        to initialize and train a model on longer sequences.
        Note that since I define restoring variables from `trainable variables`
        in the current model, if the pre-trained model doesn't have those variables,
        it will fail to restore by the saver.
        '''
        vars_to_restore = list(tf.trainable_variables())
        saver_pretrained = tf.train.Saver(vars_to_restore)

    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        print '\ntraining starts! --------------------------------------------'

        # 1. resume (full restore), 2. initialize from scratch, 3. curriculume learning (partial restore)
        prev_ckpt = 0
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            if model_file is None:
                raise ValueError('could not find checkpoint')
            print 'restore: {}'.format(model_file)
            saver.restore(sess, model_file)
            print 'done: restore'
            prev_ckpt = global_step_var.eval()
        else:
            sess.run(init_op)
            if o.curriculum_learning:
                if o.model_file is None: # e.g., '/some_path/ckpt/iteration-150000'
                    raise ValueError('could not find checkpoint')
                print 'restore: {}'.format(o.model_file)
                saver_pretrained.restore(sess, o.model_file)
                print 'done: (partial) restore for curriculum learning'

        if use_queues:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess, coord)
            # Run the feed loops in another thread.
            threads = [threading.Thread(target=feed_loop[dataset],
                                        args=(sess, coord, sequences[dataset]))
                       for dataset in datasets]
            for t in threads:
                t.start()

        if o.tfdb:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = {}
        for dataset in dataset_order:
            path_summary = os.path.join(o.path_summary, dataset)
            # Only include graph in one summary.
            if dataset == 'train':
                writer[dataset] = tf.summary.FileWriter(path_summary, sess.graph)
            else:
                writer[dataset] = tf.summary.FileWriter(path_summary)

        while True: # Loop over epochs
            global_step = global_step_var.eval() # Number of steps taken.
            if global_step >= o.num_steps:
                break
            ie = global_step / nbatch
            t_epoch = time.time()

            loss_ep = []
            for ib in range(nbatch): # Loop over batches in epoch.
                global_step = global_step_var.eval() # Number of steps taken.

                if not o.nosave:
                    period_ckpt = o.period_ckpt if not o.debugmode else 40
                    if global_step % period_ckpt == 0 and global_step > prev_ckpt:
                        if not os.path.isdir(o.path_ckpt):
                            os.makedirs(o.path_ckpt)
                        print 'save model'
                        saver.save(sess, o.path_ckpt+'/iteration', global_step=global_step)
                        print 'done: save model'
                        sys.stdout.flush()
                        prev_ckpt = global_step

                # intermediate evaluation of model
                period_assess = o.period_assess if not o.debugmode else 20
                if global_step > 0 and global_step % period_assess == 0:
                    iter_id = 'iteration{}'.format(global_step)
                    for eval_id, sampler in eval_sets.iteritems():
                        vis_dir = os.path.join(o.path_output, iter_id, eval_id)
                        if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0755)
                        visualizer = visualize.VideoFileWriter(vis_dir)
                        # Run the tracker on a full epoch.
                        print 'evaluation: {}'.format(eval_id)
                        eval_sequences = sampler()
                        # Cache the results.
                        result_file = os.path.join(o.path_output, 'assess', eval_id,
                            iter_id+'.json')
                        result = cache_json(result_file,
                            lambda: evaluate.evaluate(sess, eval_model, eval_sequences,
                                visualize=visualizer.visualize if o.visualize_eval else None),
                            makedir=True)
                        print 'IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}'.format(
                            result['iou_mean'], result['auc'], result['cle_mean'])

                # Take a training step.
                loss = {}
                dur = {}
                start = time.time()
                feed_dict = {run_opts['use_gt']:      True,
                             run_opts['is_training']: True,
                             run_opts['gt_ratio']:    max(1.0*np.exp(o.gt_decay_rate*ie), o.min_gt_ratio)}
                if use_queues:
                    feed_dict.update({queue_index: 0}) # Select data source.
                else:
                    batch_seqs = [next(sequences['train']) for i in range(o.batchsz)]
                    batch = _load_batch(batch_seqs, o)
                    feed_dict.update({example_pre_aug[k]: v for k, v in batch.iteritems()})
                    dur_load = time.time() - start
                if global_step % o.period_summary == 0:
                    summary_var = (extended_summary_vars['train']
                                   if global_step % o.period_preview == 0
                                   else summary_vars['train'])
                    _, loss['train'], summary = sess.run([optimize_op, loss_var, summary_var],
                                                feed_dict=feed_dict)
                    dur['train'] = time.time() - start
                    writer['train'].add_summary(summary, global_step=global_step)
                else:
                    _, loss['train'] = sess.run([optimize_op, loss_var], feed_dict=feed_dict)
                    dur['train'] = time.time() - start
                loss_ep.append(loss['train'])

                # Evaluate validation error.
                if global_step % o.period_summary == 0:
                    for dataset in val_datasets:
                        start = time.time()
                        feed_dict = {run_opts['use_gt']:      True,  # Match training.
                                     run_opts['is_training']: False, # Do not update bnorm stats.
                                     run_opts['gt_ratio']:    max(1.0*np.exp(o.gt_decay_rate*ie), o.min_gt_ratio)} # Match training.
                        if use_queues:
                            feed_dict.update({queue_index: dataset_index[dataset]}) # Select data source.
                        else:
                            batch_seqs = [next(sequences[dataset]) for i in range(o.batchsz)]
                            batch = _load_batch(batch_seqs, o)
                            feed_dict.update({example_pre_aug[k]: v for k, v in batch.iteritems()})
                            # dur_load = time.time() - start
                        summary_var = (extended_summary_vars[dataset]
                                       if global_step % o.period_preview == 0
                                       else summary_vars[dataset])
                        loss[dataset], summary = sess.run([loss_var, summary_var],
                                                          feed_dict=feed_dict)
                        dur[dataset] = time.time() - start
                        writer[dataset].add_summary(summary, global_step=global_step)

                # Print result of one batch update
                if o.verbose_train:
                    preamble = 'ep {}, batch {}/{} (bsz:{}), global_step {}'.format(
                        ie+1, ib+1, nbatch, o.batchsz, global_step)
                    active = [dataset for dataset in dataset_order if dataset in loss]
                    detail = 'loss:{} time:{} ({})'.format(
                        '/'.join(['{:.5f}'.format(loss[dataset]) for dataset in active]),
                        '/'.join(['{:.2f}'.format(dur[dataset]) for dataset in active]),
                        '/'.join(active))
                    print preamble + ' ' + detail

            print '[Epoch finished] ep {:d}, global_step {:d} |loss:{:.5f} (time:{:.2f})'.format(
                    ie+1, global_step_var.eval(), np.mean(loss_ep), time.time()-t_epoch)

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)


def process_sequence(example, run_opts, model, batchsz, ntimesteps, frmsz, dtype):
    '''Takes a Model and calls init() and step() on the example.

    Returns prediction in the image reference frame,
    and prediction_crop in the window reference frame.
    The prediction in the window reference frame can be used to apply the loss.
    '''
    example_input = dict(example)
    example = _guard_labels(example, run_opts)
    example_init = {k: example[k] for k in ['x0', 'y0']}
    example_seq = _unstack_dict(example, ['x', 'y', 'y_is_valid'], axis=1)
    # _unstack_example_frames(example)
    state = [None for __ in range(ntimesteps)]
    window = [None for __ in range(ntimesteps)]
    prediction_crop = [None for __ in range(ntimesteps)]
    prediction = [None for __ in range(ntimesteps)]

    with tf.variable_scope('model'):
        with tf.name_scope('frame_0'):
            with tf.variable_scope('init'):
                init_state = model.init(example_init, run_opts)
        for t in range(ntimesteps):
            with tf.name_scope('frame_{}'.format(t+1)):
                with tf.variable_scope('frame', reuse=(t > 0)):
                    # TODO: Whiten after crop i.e. in model.
                    prediction_crop[t], window[t], state[t] = model.step(
                        example_seq[t],
                        state[t-1] if t > 0 else init_state)
                # Obtain prediction in image frame.
                inv_window_rect = geom.crop_inverse(window[t])
                prediction[t] = geom.crop_prediction_frame(prediction_crop[t], inv_window_rect,
                    crop_size=[frmsz, frmsz])

    # TODO: This may include window state!
    final_state = state[-1]
    window = tf.stack(window, axis=1)
    prediction_crop = _stack_dict(prediction_crop, ['y', 'hmap', 'hmap_softmax'], axis=1)
    prediction = _stack_dict(prediction, ['y', 'hmap', 'hmap_softmax'], axis=1)

    return prediction_crop, window, prediction, init_state, final_state

def _unstack_dict(d, keys, axis):
    # Gather lists of all elements at same index.
    # {'x': [x0, x1], 'y': [y0, y1]} => [[x0, y0], [x1, y1]]
    value_lists = zip(*[tf.unstack(d[k], axis=axis) for k in keys])
    # Create a dictionary from each.
    # [[x0, y0], [x1, y1]] => [{'x': x0, 'y': y0}, {'x': x1, 'y': y1}]
    return [dict(zip(keys, vals)) for vals in value_lists]

def _stack_dict(frames, keys, axis):
    return {
        k: tf.stack([frame[k] for frame in frames], axis=axis)
        for k in keys
    }

def _get_optimizer(lr, o):
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif o.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    elif o.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    elif o.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        raise ValueError('optimizer not implemented or simply wrong.')
    return optimizer


def _make_input_pipeline(o,
        example_capacity=4, load_capacity=4, batch_capacity=4,
        num_load_threads=1, num_batch_threads=1,
        name='pipeline'):
    with tf.name_scope(name) as scope:
        files, feed_loop = pipeline.get_example_filenames(capacity=example_capacity)
        images = pipeline.load_images(files,
            image_size=[o.frmsz, o.frmsz], resize=True,
            capacity=load_capacity, num_threads=num_load_threads)
        images_batch = pipeline.batch(images,
            batch_size=o.batchsz, sequence_length=o.ntimesteps+1,
            capacity=batch_capacity, num_threads=num_batch_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, o.ntimesteps+1, None, None, None])
        images_batch['labels'].set_shape([None, o.ntimesteps+1, None])
        # Cast type of images.
        # Put in format expected by model.
        # is_valid = (range(1, o.ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        example_batch = {
            'x0':         images_batch['images'][:, 0],
            'y0':         images_batch['labels'][:, 0],
            'x':          images_batch['images'][:, 1:],
            'y':          images_batch['labels'][:, 1:],
            'y_is_valid': images_batch['label_is_valid'][:, 1:],
        }
        return example_batch, feed_loop


def _make_example_placeholders(ntimesteps, frmsz, dtype, default=None):
    shapes = {
        'x0':         [None, frmsz, frmsz, 3],
        'y0':         [None, 4],
        'x':          [None, ntimesteps, frmsz, frmsz, 3],
        'y':          [None, ntimesteps, 4],
        'y_is_valid': [None, ntimesteps],
    }
    key_dtype = lambda k: tf.bool if k.endswith('_is_valid') else dtype

    if default is not None:
        assert(set(default.keys()) == set(shapes.keys()))
        example = {
            k: tf.placeholder_with_default(default[k], shapes[k], name='placeholder_'+k)
            for k in shapes.keys()}
    else:
        example = {
            k: tf.placeholder(key_dtype(k), shapes[k], name='placeholder_'+k)
            for k in EXAMPLE_KEYS}
    return example

def _make_option_placeholders():
    run_opts = {}
    # Add a placeholder for models that use ground-truth during training.
    run_opts['use_gt'] = tf.placeholder_with_default(False, [], name='use_gt')
    # Add a placeholder that specifies training mode for e.g. batch_norm.
    run_opts['is_training'] = tf.placeholder_with_default(False, [], name='is_training')
    # Add a placeholder for scheduled sampling of y_prev_GT during training
    run_opts['gt_ratio'] = tf.placeholder_with_default(1.0, [], name='gt_ratio')
    return run_opts


def _perform_data_augmentation(example_raw, o, pad_value=None, name='data_augmentation'):

    example = dict(example_raw)

    xs_aug = tf.concat([tf.expand_dims(example['x0'], 1), example['x']], 1)
    ys_aug = tf.concat([tf.expand_dims(example['y0'], 1), example['y']], 1)

    if o.data_augmentation.get('hue', False):
        xs_aug = _data_augmentation_hue(xs_aug, o)

    if o.data_augmentation.get('saturation', False):
        xs_aug = _data_augmentation_saturation(xs_aug, o)

    if o.data_augmentation.get('brightness', False):
        xs_aug = tf.image.random_brightness(xs_aug, 0.1)

    if o.data_augmentation.get('contrast', False):
        xs_aug = tf.image.random_contrast(xs_aug, 0.5, 1.5)

    if o.data_augmentation.get('scale_shift', False):
        xs_aug, ys_aug = _data_augmentation_scale_shift(xs_aug, ys_aug, o, pad_value=pad_value)

    if o.data_augmentation.get('flip_up_down', False):
        xs_aug, ys_aug = _data_augmentation_flip_up_down(xs_aug, ys_aug, o)

    if o.data_augmentation.get('flip_left_right', False):
        xs_aug, ys_aug = _data_augmentation_flip_left_right(xs_aug, ys_aug, o)

    # TODO: May try other augmentations at expense - tf.image.{rot90, etc.}

    example['x0'] = xs_aug[:,0]
    example['y0'] = ys_aug[:,0]
    example['x']  = xs_aug[:,1:]
    example['y']  = ys_aug[:,1:]
    return example


def _data_augmentation_scale_shift(xs, ys, o, pad_value=None):
    def _augment(x, y, ratio):
        x_min = tf.random_uniform([], maxval=(1-ratio))
        y_min = tf.random_uniform([], maxval=(1-ratio))
        x_max = x_min + ratio
        y_max = y_min + ratio
        # Now invert these boxes.
        # (a, b) in (0, 1) is like (0, 1) in (c, d)
        # This gives:
        #   (b - a) / (1 - 0) = (1 - 0) / (d - c)
        #   ratio = 1 / (d - c)
        # Therefore:
        #   (a - 0) / (1 - 0) = (0 - c) / (d - c)
        #   c = -a / ratio
        u_min = -x_min / ratio
        v_min = -y_min / ratio
        u_max = u_min + 1/ratio
        v_max = v_min + 1/ratio
        boxes = tf.stack([v_min, u_min, v_max, u_max])
        boxes = tf.expand_dims(boxes, 0)
        # Same box in every image.
        n = x.shape.as_list()[0] # tf.shape(x)[0]
        assert n is not None
        boxes = tf.tile(boxes, [n, 1])
        x_aug = tf.image.crop_and_resize(x, boxes, box_ind=tf.range(n),
            crop_size=(o.frmsz, o.frmsz),
            method='bilinear',
            extrapolation_value=pad_value)
        y_aug = y*ratio + tf.stack([x_min, y_min, x_min, y_min])
        return x_aug, y_aug

    max_side_before = tf.reduce_max(tf.maximum(ys[:,:,2]-ys[:,:,0], ys[:,:,3]-ys[:,:,1]), 1)
    max_side_after = tf.random_uniform(tf.shape(max_side_before), minval=0.05, maxval=0.5)
    ratios = tf.divide(max_side_after, max_side_before)
    ratios = tf.maximum(0.5, ratios) # minimum scale
    ratios = tf.minimum(1.0, ratios)
    xs_aug, ys_aug = tf.map_fn(
        lambda (x, y, ratio): _augment(x, y, ratio),
        (xs, ys, ratios),
        dtype=(o.dtype, o.dtype))
    return xs_aug, ys_aug
    # xs_aug = []
    # ys_aug = []
    # # TODO: Use tf.map_fn
    # for i in range(o.batchsz):
    #     x_aug, y_aug = tf.cond(tf.less(ratio[i], 1.0),
    #                            lambda: _augment_pad(xs[i], ys[i], ratio[i]),
    #                            lambda: _augment_crop(xs[i], ys[i], ratio[i]))
    #     xs_aug.append(x_aug)
    #     ys_aug.append(y_aug)
    # return tf.stack(xs_aug), tf.stack(ys_aug)


def _data_augmentation_flip_up_down(xs, ys, o):
    xs_flip = []
    ys_flip = []
    prob_up_down = tf.random_uniform([o.batchsz])
    for i in range(o.batchsz):
        def _flip_up_down(x, y):
            x_flip = []
            y_flip = []
            for t in range(o.ntimesteps+1):
                x_flip.append(tf.image.flip_up_down(x[t])) # NOTE: doesn't support batch processing
                y_flip.append(tf.stack([y[t][k] if k % 2 == 0 else 1-y[t][k] for k in [0, 3, 2, 1]]))
            return tf.stack(x_flip), tf.stack(y_flip)
        x_flip, y_flip = tf.cond(tf.less(prob_up_down[i], 0.5),
                                 lambda: _flip_up_down(xs[i], ys[i]),
                                 lambda: (tf.identity(xs[i]), tf.identity(ys[i])))
        xs_flip.append(x_flip)
        ys_flip.append(y_flip)
    return tf.stack(xs_flip), tf.stack(ys_flip)


def _data_augmentation_flip_left_right(xs, ys, o):
    xs_flip = []
    ys_flip = []
    prob_left_right = tf.random_uniform([o.batchsz])
    for i in range(o.batchsz):
        def _flip_left_right(x, y):
            x_flip = []
            y_flip = []
            for t in range(o.ntimesteps+1):
                x_flip.append(tf.image.flip_left_right(x[t])) # NOTE: doesn't support batch processing
                y_flip.append(tf.stack([y[t][k] if k % 2 == 1 else 1-y[t][k] for k in [2, 1, 0, 3]]))
            return tf.stack(x_flip), tf.stack(y_flip)
        x_flip, y_flip = tf.cond(tf.less(prob_left_right[i], 0.5),
                                 lambda: _flip_left_right(xs[i], ys[i]),
                                 lambda: (tf.identity(xs[i]), tf.identity(ys[i])))
        xs_flip.append(x_flip)
        ys_flip.append(y_flip)
    return tf.stack(xs_flip), tf.stack(ys_flip)


def _data_augmentation_hue(xs, o, max_delta=0.1):
    '''
    This data augmentation is applied by frame.
    '''
    xs_aug = []
    for i in range(o.batchsz):
        for t in range(o.ntimesteps+1):
            xs_aug.append(tf.image.random_hue(xs[i,t], max_delta))
    return tf.reshape(tf.stack(xs_aug), [-1, o.ntimesteps+1, o.frmsz, o.frmsz, 3])


def _data_augmentation_saturation(xs, o, lower=0.9, upper=1.1):
    '''
    This data augmentation is applied by sequence.
    '''
    xs_aug = []
    for i in range(o.batchsz):
        for t in range(o.ntimesteps+1):
            xs_aug.append(tf.image.random_saturation(xs[i,t], lower, upper))
    return tf.reshape(tf.stack(xs_aug), [-1, o.ntimesteps+1, o.frmsz, o.frmsz, 3])


def _guard_labels(example, run_opts):
    '''Hides the 'y' labels if 'use_gt' is False.

    This prevents the model from accidentally using 'y'.
    '''
    # example['x'] -- [b, t, h, w, 3]
    # example['y']     -- [b, t, 4]
    images = example['x']
    safe = dict(example)
    images_shape = _most_specific_shape(images)
    safe['y'] = tf.cond(run_opts['use_gt'],
        lambda: example['y'],
        lambda: tf.fill(images_shape[0:2] + [4], float('nan')),
        name='labels_safe')
    return safe

def _most_specific_shape(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [static[i] or dynamic[i] for i in range(len(static))]


# def _whiten(example_raw, dtype, stat, name='whiten'):
#     stat = stat or {}
#     with tf.name_scope(name) as scope:
#         # Normalize mean and variance.
#         mean = stat.get('mean', 0.0)
#         std = stat.get('std', 1.0)
#         example = dict(example_raw) # Copy dictionary before modifying.
#         # Replace raw x (images) with whitened x (images).
#         if 'x' in example:
#             example['x'] = _whiten_image(example['x'], mean, std, name='x')
#         if 'x0' in example:
#             example['x0'] = _whiten_image(example['x0'], mean, std, name='x0')
#         return example
# 
# 
# def _whiten_image(x, mean, std, name='whiten_image'):
#     with tf.name_scope(name) as scope:
#         return tf.divide(x - mean, std, name=scope)




def iter_examples(train_distribution, rand=None): # , num_epochs=None):
    '''Returns generator that produces multiple epochs of examples for SGD.

    train_distribution -- List of tuples (name, p, source).
    '''
    names, p, dataset_samplers = zip(*train_distribution)
    # dataset_names = train_distribution.keys()
    # p = np.array([train_distribution[name]['weight'] for name in dataset_names])
    p = np.array(p)
    p = p / float(sum(p))

    num_epochs = [0 for __ in train_distribution]
    num_sequences = [0 for __ in train_distribution]
    # Collection of sequences in current epoch of each dataset.
    epochs = [iter(()) for __ in train_distribution]

    while True:
        # Choose a dataset from which to pull the next sequence.
        dataset_ind = rand.choice(range(len(train_distribution)), p=p)
        while True: # Re-try if next() fails.
            try:
                sequence_name, sequence = next(epochs[dataset_ind])
                num_sequences[dataset_ind] += 1
            except StopIteration:
                print 'epoch {}: num sequences {}'.format(
                    num_epochs[dataset_ind]+1,
                    num_sequences[dataset_ind],
                )
                if num_sequences[dataset_ind] == 0:
                    # Prevent infinite loop with no data.
                    raise ValueError('epoch was empty')
                epochs[dataset_ind] = sample.epoch(
                    train_distribution[dataset_ind]['sequence_sampler'],
                    rand,
                    max_objects=1,
                )
                num_epochs[dataset_ind] += 1
                num_sequences[dataset_ind] = 0
                continue
            break
        yield sequence


def get_loss(example, pred, o, summaries_collections=None, image_summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        y          = example['y']
        y_is_valid = example['y_is_valid']
        assert(y.get_shape().as_list()[1] == o.ntimesteps)
        # TODO: Should we enforce a larger minimum rectangle size here?
        hmap = convert_rec_to_heatmap(y, o.frmsz, min_size=1.0)
        if o.heatmap_stride != 1:
            hmap, unmerge = merge_dims(hmap, 0, 2)
            hmap = slim.avg_pool2d(hmap,
                kernel_size=o.heatmap_stride+1,
                stride=o.heatmap_stride,
                padding='SAME')
            hmap = unmerge(hmap, 0)

        losses = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in o.losses or 'l1_relative' in o.losses:
            y_pred = pred['y']
            y_valid = tf.boolean_mask(y, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            loss_l1 = tf.reduce_mean(tf.abs(y_valid - y_pred_valid), axis=-1)
            if 'l1' in o.losses:
                losses['l1'] = tf.reduce_mean(loss_l1)
            if 'l1_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_valid[:,2] - y_valid[:,0])
                y_size = tf.abs(y_valid[:,3] - y_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                loss_l1_relative = loss_l1 / (tf.reduce_mean(size, axis=-1) + 0.05)
                losses['l1_relative'] = tf.reduce_mean(loss_l1_relative)

        # CLE (center location error). Measured in l2 distance.
        if 'cle' in o.losses or 'cle_relative' in o.losses:
            y_pred = pred['y']
            y_valid = tf.boolean_mask(y, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            x_center = (y_valid[:,2] + y_valid[:,0]) * 0.5
            y_center = (y_valid[:,3] + y_valid[:,1]) * 0.5
            center = tf.stack([x_center, y_center], axis=-1)
            x_pred_center = (y_pred_valid[:,2] + y_pred_valid[:,0]) * 0.5
            y_pred_center = (y_pred_valid[:,3] + y_pred_valid[:,1]) * 0.5
            pred_center = tf.stack([x_pred_center, y_pred_center], axis=-1)
            loss_cle = tf.norm(center - pred_center, axis=-1)
            if 'cle' in o.losses:
                losses['cle'] = tf.reduce_mean(loss_cle)
            if 'cle_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_valid[:,2] - y_valid[:,0])
                y_size = tf.abs(y_valid[:,3] - y_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                radius = tf.exp(tf.reduce_mean(tf.log(size), axis=-1))
                loss_cle_relative = loss_cle / (radius + 0.05)
                losses['cle_relative'] = tf.reduce_mean(loss_cle_relative)

        # Cross-entropy between probabilty maps (need to change label)
        if 'ce' in o.losses or 'ce_balanced' in o.losses:
            hmap_pred = pred['hmap']
            print 'hmap_pred.shape:', hmap_pred.shape.as_list()
            print 'hmap.shape:', hmap.shape.as_list()
            hmap_valid = tf.boolean_mask(hmap, y_is_valid)
            hmap_pred_valid = tf.boolean_mask(hmap_pred, y_is_valid)
            # hmap is [valid_images, height, width, 2]
            mass = tf.reduce_sum(hmap_valid, axis=(1, 2), keep_dims=True)
            class_mass = 0.5 / tf.cast(mass+1, tf.float32)
            # TODO: Does this work when labels are not [1, 0] or [0, 1]?
            coeff = tf.reduce_sum(hmap_valid * class_mass, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_valid, unmerge = merge_dims(hmap_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=hmap_valid,
                logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            if 'ce' in o.losses:
                losses['ce'] = tf.reduce_mean(loss_ce)
            if 'ce_balanced' in o.losses:
                losses['ce_balanced'] = tf.reduce_mean(
                    tf.reduce_sum(coeff * loss_ce, axis=(1, 2)))

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

            if 'hmap' in pred:
                with tf.name_scope('hmap'):
                    tf.summary.image('gt', _draw_heatmap(hmap[0]),
                        collections=image_summaries_collections)
                    tf.summary.image('pred', _draw_heatmap(tf.nn.softmax(pred['hmap'])[0]),
                        max_outputs=o.ntimesteps+1, collections=image_summaries_collections)

        return tf.reduce_sum(losses.values(), name=scope)


def _draw_init_bounding_boxes(example, num_sequences=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x0']   -- [b, h, w, 3]
        # example['y0']       -- [b, 4]
        # Just do the first example in the batch.
        # image = (1.0/255)*example['x0'][0:1]
        image = example['x0'][:num_sequences]
        y_gt = example['y0'][:num_sequences]
        y = tf.stack([y_gt], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        image = tf.image.draw_bounding_boxes(image, boxes, name=scope)
        # Preserve absolute colors in summary.
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
        return image

def _draw_bounding_boxes(example, pred, time_stride=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x']   -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # pred['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        # image = (1.0/255)*example['x'][0][::time_stride]
        image = example['x'][0][::time_stride]
        y_gt = example['y'][0][::time_stride]
        y_pred = pred['y'][0][::time_stride]
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        image = tf.image.draw_bounding_boxes(image, boxes, name=scope)
        # Preserve absolute colors in summary.
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
        return image

def _draw_heatmap(prob, time_stride=1, name='draw_heatmap'):
    assert len(prob.shape) == 4
    with tf.name_scope(name) as scope:
        # prob -- [b, t, frmsz, frmsz, 2]
        # Take first channel and convert to int.
        return tf.cast(tf.round(255*prob[:,:,:,0:1]), tf.uint8)

def _draw_memory_state(model, mtype, time_stride=1, name='draw_memory_states'):
    with tf.name_scope(name) as scope:
        p = tf.nn.softmax(model.memory[mtype][0,::time_stride])
        return tf.cast(tf.round(255*p[:,:,:,0:1]), tf.uint8)

def _load_sequence(seq, o):
    '''
    Sequence has keys:
        'image_files'    # Tensor with shape [n] containing strings.
        'labels'         # Tensor with shape [n, 4] containing rectangles.
        'label_is_valid' # Tensor with shape [n] containing booleans.
    Example has keys:
        'x0'         # First image in sequence, shape [h, w, 3]
        'y0'         # Position of target in first image, shape [4]
        'x'          # Input images, shape [n-1, h, w, 3]
        'y'          # Position of target in following frames, shape [n-1, 4]
        'y_is_valid' # Booleans indicating presence of frame, shape [n-1]
    '''
    seq_len = len(seq['image_files'])
    assert(len(seq['labels']) == seq_len)
    assert(len(seq['label_is_valid']) == seq_len)
    assert(seq['label_is_valid'][0] == True)
    f = lambda x: im_to_arr(load_image(x, size=(o.frmsz, o.frmsz), resize=True),
                            dtype=o.dtype.as_numpy_dtype)
    images = map(f, seq['image_files'])
    return {
        'x0':         np.array(images[0]),
        'y0':         np.array(seq['labels'][0]),
        'x':          np.array(images[1:]),
        'y':          np.array(seq['labels'][1:]),
        'y_is_valid': np.array(seq['label_is_valid'][1:]),
    }

def _load_batch(seqs, o):
    sequence_keys = set(['x', 'y', 'y_is_valid'])
    examples = map(lambda x: _load_sequence(x, o), seqs)
    # Pad all sequences to o.ntimesteps.
    # NOTE: Assumes that none of the arrays to be padded are empty.
    return {k: np.stack([pad_to(x[k], o.ntimesteps, axis=0)
                             if k in sequence_keys else x[k]
                         for x in examples])
            for k in EXAMPLE_KEYS}

def generate_report(samplers, datasets, o, metrics=['iou_mean', 'auc', 'cle_mean']):
    '''Finds all results for each evaluation distribution.

    Identifies the best result for each metric.
    Caution: More frequent evaluations might lead to better results.
    '''
    def helper():
        eval_id_fn = lambda sampler, dataset: '{}-{}'.format(dataset, sampler)
        best_fn = {'iou_mean': np.amax, 'auc': np.amax, 'cle_mean': np.amin}
        report_dir = os.path.join(o.path_output, 'report')
        if not os.path.isdir(report_dir): os.makedirs(report_dir)

        # Plot each metric versus iteration.
        # Create one plot per sampler, with a line per dataset.
        for sampler in samplers:
            # Load all results using this sampler.
            results = {dataset: load_results(eval_id_fn(sampler, dataset)) for dataset in datasets}
            # Print results for each dataset.
            for dataset in datasets:
                print '==== evaluation: sampler {}, dataset {} ===='.format(sampler, dataset)
                steps = sorted(results[dataset].keys())
                for step in steps:
                    print 'iter {}:  {}'.format(step,
                        '; '.join(['{}: {:.3g}'.format(metric, results[dataset][step][metric])
                                   for metric in metrics]))
                for metric in metrics:
                    values = [results[dataset][step][metric] for step in steps]
                    print 'best {}: {:.3g}'.format(metric, np.asscalar(best_fn[metric](values)))
            # Generate plot for each metric.
            # Take union of steps for all datasets.
            steps = sorted(set.union(*[set(r.keys()) for r in results.values()]))
            for metric in metrics:
                # Plot this metric over time for all datasets.
                data_file = 'sampler-{}-metric-{}'.format(sampler, metric)
                with open(os.path.join(report_dir, data_file+'.tsv'), 'w') as f:
                    write_data_file(f, metric, steps, results)
                try:
                    plot_file = plot_data(report_dir, data_file)
                    print 'plot created:', plot_file
                except Exception as e:
                    print 'could not create plot:', e

    def load_results(eval_id):
        '''Returns a dictionary from step number to dictionary of metrics.'''
        dirname = os.path.join(o.path_output, 'assess', eval_id)
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
            print 'warning: no results found:', eval_id
        return results

    def write_data_file(f, metric, steps, results):
        # Create a column for the variance.
        fieldnames = ['step'] + [x+suffix for x in datasets for suffix in ['', '_std_err']]
        w = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        w.writeheader()
        for step in steps:
            # Map variance of metric to variance of 
            row = {
                dataset+suffix:
                    gnuplot_str(results[dataset].get(step, {}).get(metric+suffix, None))
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
        return os.path.join(plot_dir, filename+'.png')

    return helper()


def gnuplot_str(x):
    if x is None:
        return '?'
    return str(x)
