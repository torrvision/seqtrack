import pdb
import sys
import csv
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
import helpers
import pipeline
import sample
import visualize

from model import convert_rec_to_heatmap
from helpers import load_image, im_to_arr, pad_to, cache_json, merge_dims

EXAMPLE_KEYS = ['x0_raw', 'y0', 'x_raw', 'y', 'y_is_valid']


def train(create_model, datasets, eval_sets, o, use_queues=False):
    '''Trains a network.

    Args:
        create_model: Function that takes as input a dictionary of tensors and
            returns a model object.
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

    modes = ['train', 'val']

    feed_loop = {} # Each value is a function to call in a thread.
    with tf.name_scope('input'):
        from_queue = None
        if use_queues:
            queues = []
            for mode in modes:
                # Create a queue each for training and validation data.
                queue, feed_loop[mode] = _make_input_pipeline(o,
                    num_load_threads=1, num_batch_threads=1, name='pipeline_'+mode)
                queues.append(queue)
            queue_index, from_queue = pipeline.make_multiplexer(queues,
                capacity=4, num_threads=1)
        example = _make_placeholders(o, default=from_queue)

    # data augmentation
    example = _perform_data_augmentation(example, o)

    # Always use same statistics for whitening (not set dependent).
    stat = datasets['train'].stat
    # TODO: Mask y with use_gt to prevent accidental use.
    model = create_model(_whiten(_guard_labels(example), o, stat=stat))
    loss_var, model.outputs = get_loss(example, model.outputs, o)
    r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('regularization', r)
    loss_var += r
    tf.summary.scalar('total', loss_var)

    nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = len(datasets['train'].videos)/o.batchsz if not o.debugmode else 30
    nbatch_val = len(datasets['val'].videos)/o.batchsz if not o.debugmode else 30

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
    summary_vars_with_preview = {}
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        # Create a preview and add it to the list of summaries.
        if 'y_pred' in model.outputs:
            boxes = tf.summary.image('box',
                _draw_bounding_boxes(example, model),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries = [boxes]
        # Produce an image summary of the heatmap (image-centric).
        if 'hmap_pred' in model.outputs:
            hmap_pred = tf.summary.image('hmap_pred',
                _draw_heatmap(model, 'hmap_pred'),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries.append(hmap_pred)
        # Produce an image summary of the heatmap (object-centric).
        # TODO: After fixing bug, maybe remove this.
        if 'hmap_pred_oc' in model.outputs:
            hmap_pred = tf.summary.image('hmap_pred_oc',
                _draw_heatmap(model, 'hmap_pred_oc'),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries.append(hmap_pred)
        # Produce an image summary of the heatmap-GT.
        if 'hmap_gt' in model.outputs:
            hmap_gt = tf.summary.image('hmap_gt',
                _draw_heatmap(model, 'hmap_gt'),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries.append(hmap_gt)
        # Produce an image summary of the heatmap-GT.
        # TODO: After fixing bug, maybe remove this.
        if 'hmap_gt_oc' in model.outputs:
            hmap_gt = tf.summary.image('hmap_gt_oc',
                _draw_heatmap(model, 'hmap_gt_oc'),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries.append(hmap_gt)
        # Produce an image summary of target and search images (input to CNNs).
        if 'target' in model.outputs and 'search' in model.outputs:
            for key in ['target', 'search']:
                input_image = tf.summary.image('cnn_input_{}'.format(key),
                    _draw_input_image(model, key, name='draw_{}'.format(key)),
                    max_outputs=o.ntimesteps+1, collections=[])
                image_summaries.append(input_image)
        # Produce an image summary of the LSTM memory states (h or c).
        if hasattr(model, 'memory'):
            for mtype in model.memory.keys():
                if model.memory[mtype][0] is not None:
                    image_summaries.append(
                        tf.summary.image(mtype, _draw_memory_state(model, mtype),
                        max_outputs=o.ntimesteps+1, collections=[]))
        for mode in modes:
            with tf.name_scope(mode):
                # Merge summaries with any that are specific to the mode.
                summaries = (global_summaries + tf.get_collection('summaries_' + mode))
                summary_vars[mode] = tf.summary.merge(summaries)
                summaries.extend(image_summaries)
                summary_vars_with_preview[mode] = tf.summary.merge(summaries)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    # Use a separate random number generator for each sampler.
    sequences = {mode: iter_examples(datasets[mode], o,
                                     generator=random.Random(o.seed_global),
                                     num_epochs=None)
                 for mode in modes}

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
            threads = [threading.Thread(target=feed_loop[mode],
                                        args=(sess, coord, sequences[mode]))
                       for mode in modes]
            for t in threads:
                t.start()

        if o.tfdb:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = {}
        for mode in modes:
            path_summary = os.path.join(o.path_summary, mode)
            # Only include graph in one summary.
            if mode == 'train':
                writer[mode] = tf.summary.FileWriter(path_summary, sess.graph)
            else:
                writer[mode] = tf.summary.FileWriter(path_summary)

        while True: # Loop over epochs
            global_step = global_step_var.eval() # Number of steps taken.
            if global_step >= nepoch * nbatch:
                break
            ie = global_step / nbatch
            t_epoch = time.time()

            ib_val = 0
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
                            lambda: evaluate.evaluate(sess, example, model,
                                eval_sequences, visualize=visualizer.visualize if o.save_videos else None,
                                use_gt=o.use_gt_eval,
                                save_frames=o.save_frames),
                            makedir=True)
                        print 'IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}'.format(
                            result['iou_mean'], result['auc'], result['cle_mean'])

                # Take a training step.
                start = time.time()
                feed_dict = {example['use_gt']:      o.use_gt_train,
                             example['is_training']: True,
                             example['gt_ratio']:    max(1.0*np.exp(o.gt_decay_rate*ie), o.min_gt_ratio)}
                if use_queues:
                    feed_dict.update({queue_index: 0}) # Choose validation queue.
                else:
                    batch_seqs = [next(sequences['train']) for i in range(o.batchsz)]
                    batch = _load_batch(batch_seqs, o)
                    feed_dict.update({example[k]: v for k, v in batch.iteritems()})
                    dur_load = time.time() - start
                if global_step % o.period_summary == 0:
                    summary_var = (summary_vars_with_preview['train']
                                   if global_step % o.period_preview == 0
                                   else summary_vars['train'])
                    #_, loss, summary = sess.run([optimize_op, loss_var, summary_var],
                    #                            feed_dict=feed_dict)
                    _, loss, summary, y_init = sess.run([optimize_op, loss_var, summary_var,
                                                         model.state['y'][0]],
                                                feed_dict=feed_dict)
                    dur = time.time() - start
                    writer['train'].add_summary(summary, global_step=global_step)
                else:
                    #_, loss = sess.run([optimize_op, loss_var], feed_dict=feed_dict)
                    _, loss, y_init = sess.run([optimize_op, loss_var, model.state['y'][0]], feed_dict=feed_dict)
                    dur = time.time() - start
                loss_ep.append(loss)

                newval = False
                # Evaluate validation error.
                if global_step % o.period_summary == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        feed_dict = {example['use_gt']:      o.use_gt_train,  # Match training.
                                     example['is_training']: False, # Do not update bnorm stats.
                                     example['gt_ratio']:    max(1.0*np.exp(o.gt_decay_rate*ie), o.min_gt_ratio)} # Match training.
                        if use_queues:
                            feed_dict.update({queue_index: 1}) # Choose validation queue.
                        else:
                            batch_seqs = [next(sequences['val']) for i in range(o.batchsz)]
                            batch = _load_batch(batch_seqs, o)
                            feed_dict.update({example[k]: v for k, v in batch.iteritems()})
                            dur_load = time.time() - start
                        summary_var = (summary_vars_with_preview['val']
                                       if global_step % o.period_preview == 0
                                       else summary_vars['val'])
                        loss_val, summary = sess.run([loss_var, summary_var],
                                                     feed_dict=feed_dict)
                        dur_val = time.time() - start
                        writer['val'].add_summary(summary, global_step=global_step)
                        ib_val += 1
                        newval = True

                # Print result of one batch update
                if o.verbose_train:
                    losstime = '|loss:{:.5f}/{:.5f} (time:{:.2f}/{:.2f}) - with val'.format(
                            loss, loss_val, dur, dur_val) if newval else \
                            '|loss:{:.5f} (time:{:.2f})'.format(loss, dur)
                    print 'ep {}/{}, batch {}/{} (bsz:{}), global_step {} {}'.format(
                            ie+1, nepoch, ib+1, nbatch, o.batchsz, global_step, losstime)

            print '[Epoch finished] ep {:d}/{:d}, global_step {:d} |loss:{:.5f} (time:{:.2f})'.format(
                    ie+1, nepoch, global_step_var.eval(), np.mean(loss_ep), time.time()-t_epoch)

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)


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


def _make_input_pipeline(o, dtype=tf.float32,
        example_capacity=4, load_capacity=4, batch_capacity=4,
        num_load_threads=1, num_batch_threads=1,
        name='pipeline'):
    with tf.name_scope(name) as scope:
        files, feed_loop = pipeline.get_example_filenames(capacity=example_capacity)
        images = pipeline.load_images(files, capacity=load_capacity,
                num_threads=num_load_threads, image_size=[o.frmsz, o.frmsz, 3])
        images_batch = pipeline.batch(images,
            batch_size=o.batchsz, sequence_length=o.ntimesteps+1,
            capacity=batch_capacity, num_threads=num_batch_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, o.ntimesteps+1, None, None, None])
        images_batch['labels'].set_shape([None, o.ntimesteps+1, None])
        # Cast type of images.
        images_batch['images'] = tf.cast(images_batch['images'], o.dtype)
        # Put in format expected by model.
        # is_valid = (range(1, o.ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        example_batch = {
            'x0_raw':     images_batch['images'][:, 0],
            'y0':         images_batch['labels'][:, 0],
            'x_raw':      images_batch['images'][:, 1:],
            'y':          images_batch['labels'][:, 1:],
            'y_is_valid': images_batch['label_is_valid'][:, 1:],
        }
        return example_batch, feed_loop


def _make_placeholders(o, default=None):
    shapes = {
        'x0_raw':     [None, o.frmsz, o.frmsz, 3],
        'y0':         [None, 4],
        'x_raw':      [None, o.ntimesteps, o.frmsz, o.frmsz, 3],
        'y':          [None, o.ntimesteps, 4],
        'y_is_valid': [None, o.ntimesteps],
    }
    dtype = lambda k: tf.bool if k.endswith('_is_valid') else o.dtype

    if default is not None:
        assert(set(default.keys()) == set(shapes.keys()))
        example = {
            k: tf.placeholder_with_default(default[k], shapes[k], name='placeholder_'+k)
            for k in shapes.keys()}
    else:
        example = {
            k: tf.placeholder(dtype(k), shapes[k], name='placeholder_'+k)
            for k in EXAMPLE_KEYS}
    # Add a placeholder for models that use ground-truth during training.
    example['use_gt'] = tf.placeholder_with_default(False, [], name='use_gt')
    # Add a placeholder that specifies training mode for e.g. batch_norm.
    example['is_training'] = tf.placeholder_with_default(False, [], name='is_training')
    # Add a placeholder for scheduled sampling of y_prev_GT during training
    example['gt_ratio'] = tf.placeholder_with_default(1.0, [], name='gt_ratio')
    return example


def _perform_data_augmentation(example_raw, o, name='data_augmentation'):

    example = dict(example_raw)

    xs_aug = tf.concat([tf.expand_dims(example['x0_raw'], 1), example['x_raw']], 1)
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
        xs_aug, ys_aug = _data_augmentation_scale_shift(xs_aug, ys_aug, o)

    if o.data_augmentation.get('flip_up_down', False):
        xs_aug, ys_aug = _data_augmentation_flip_up_down(xs_aug, ys_aug, o)

    if o.data_augmentation.get('flip_left_right', False):
        xs_aug, ys_aug = _data_augmentation_flip_left_right(xs_aug, ys_aug, o)

    # TODO: May try other augmentations at expense - tf.image.{rot90, etc.}

    example['x0_raw'] = xs_aug[:,0]
    example['x_raw']  = xs_aug[:,1:]
    example['y0']     = ys_aug[:,0]
    example['y']      = ys_aug[:,1:]
    return example


def _data_augmentation_scale_shift(xs, ys, o):
    max_side_before = tf.reduce_max(tf.maximum(ys[:,:,2]-ys[:,:,0], ys[:,:,3]-ys[:,:,1]), 1)
    max_side_after = tf.random_uniform(tf.shape(max_side_before), minval=0.05, maxval=1.0)
    ratio = tf.divide(max_side_after, max_side_before)
    xs_aug = []
    ys_aug = []
    for i in range(o.batchsz):
        def _augment_pad(x, y, ratio):
            ''' Case: ratio < 1.
            Frames get resized (smaller) and padded to original size.
            '''
            ratio = tf.maximum(0.2, ratio) # minimum scale
            x_resize = tf.image.resize_images(x, [tf.to_int32(ratio*o.frmsz)]*2,
                                              method=tf.image.ResizeMethod.BICUBIC)
            offset_h = tf.to_int32(tf.random_uniform([], maxval=o.frmsz*(1-ratio)))
            offset_w = tf.to_int32(tf.random_uniform([], maxval=o.frmsz*(1-ratio)))
            # NOTE: `tf.pad` doesn't take pad value and only pad with zeros.
            x_aug = tf.image.pad_to_bounding_box(x_resize, offset_h, offset_w, o.frmsz, o.frmsz)
            y_aug = y*ratio + tf.cast(tf.divide(tf.stack([offset_w, offset_h]*2), o.frmsz), o.dtype)
            return x_aug, y_aug
        def _augment_crop(x, y, ratio):
            return tf.identity(x), tf.identity(y) # TODO: implement.
        x_aug, y_aug = tf.cond(tf.less(ratio[i], 1.0),
                               lambda: _augment_pad(xs[i], ys[i], ratio[i]),
                               lambda: _augment_crop(xs[i], ys[i], ratio[i]))
        xs_aug.append(x_aug)
        ys_aug.append(y_aug)
    return tf.stack(xs_aug), tf.stack(ys_aug)


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


def _guard_labels(unsafe):
    '''Hides the 'y' labels if 'use_gt' is False.

    This prevents the model from accidentally using 'y'.
    '''
    # unsafe['x_raw'] -- [b, t, h, w, 3]
    # unsafe['y']     -- [b, t, 4]
    images = unsafe['x_raw']
    safe = dict(unsafe)
    safe['y'] = tf.cond(unsafe['use_gt'],
        lambda: unsafe['y'],
        lambda: tf.fill(tf.concat([tf.shape(images)[0:2], [4]], axis=0), float('nan')),
        name='labels_safe')
    return safe


def _whiten(example_raw, o, stat=None, name='whiten'):
    with tf.name_scope(name) as scope:
        # Normalize mean and variance.
        assert(stat is not None)
        # TODO: Check that this does not create two variables:
        mean = tf.constant(stat['mean'] if stat else 0.0, o.dtype, name='mean')
        std = tf.constant(stat['std'] if stat else 1.0,  o.dtype, name='std')
        example = dict(example_raw) # Copy dictionary before modifying.
        # Replace raw x (images) with whitened x (images).
        example['x'] = _whiten_image(example['x_raw'], mean, std, name='x')
        del example['x_raw']
        example['x0'] = _whiten_image(example['x0_raw'], mean, std, name='x0')
        del example['x0_raw']
        return example


def _whiten_image(x, mean, std, name='whiten_image'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)


def iter_examples(dataset, o, generator=None, num_epochs=None):
    '''Generator that produces multiple epochs of examples for SGD.'''
    if num_epochs:
        epochs = xrange(num_epochs)
    else:
        epochs = itertools.count()
    for i in epochs:
        sequences = sample.sample(dataset, generator=generator,
                                  shuffle=True, max_objects=1, ntimesteps=o.ntimesteps,
                                  **o.sampler_params)
        for sequence in sequences:
            yield sequence

def _convert_hmap_object_centric(hmap, box_s_raw, box_s_val, o):
    x1_s_val, y1_s_val, x2_s_val, y2_s_val = tf.unstack(box_s_val, axis=2)
    x1_s_raw, y1_s_raw, x2_s_raw, y2_s_raw = tf.unstack(box_s_raw, axis=2)
    s_val_h = tf.cast((y2_s_val-y1_s_val)*o.frmsz, tf.int32)
    s_val_w = tf.cast((x2_s_val-x1_s_val)*o.frmsz, tf.int32)
    s_raw_h = tf.cast((y2_s_raw-y1_s_raw)*o.frmsz, tf.int32)
    s_raw_w = tf.cast((x2_s_raw-x1_s_raw)*o.frmsz, tf.int32)
    hmap_oc = []
    for b in range(o.batchsz):
        hmap_gt_oc_b = []
        for t in range(o.ntimesteps):
            # crop (only within valid region)
            crop_val = tf.image.crop_to_bounding_box(
                image=hmap[b,t],
                offset_height=tf.cast(y1_s_val[b,t]*o.frmsz, tf.int32),
                offset_width =tf.cast(x1_s_val[b,t]*o.frmsz, tf.int32),
                target_height=s_val_h[b,t],
                target_width =s_val_w[b,t])
            # pad crop so that it preserves aspect ratio.
            crop_pad = tf.image.pad_to_bounding_box(
                image=crop_val,
                offset_height=tf.cast((y1_s_val[b,t]-y1_s_raw[b,t])*o.frmsz, tf.int32),
                offset_width =tf.cast((x1_s_val[b,t]-x1_s_raw[b,t])*o.frmsz, tf.int32),
                target_height=s_raw_h[b,t],
                target_width =s_raw_w[b,t])
            # resize to 241 x 241
            crop_res = tf.image.resize_images(crop_pad, [o.frmsz, o.frmsz])
            hmap_gt_oc_b.append(crop_res)
        hmap_oc.append(tf.stack(hmap_gt_oc_b, 0))
    return tf.stack(hmap_oc, 0)

def get_loss(example, outputs, o, summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        if 'y_gt_oc' in outputs: # object-centric approach.
            assert False
            y_gt = outputs['y_gt_oc']
        else:
            y_gt = example['y']
        assert(y_gt.get_shape().as_list()[1] == o.ntimesteps)

        # Gaussian gt hmap
        hmap_gt = convert_rec_to_heatmap(y_gt, o, min_size=1.0, Gaussian=True)
        outputs['hmap_gt'] = hmap_gt # To visualize hmap_gt (image-centric) in summary.
        hmap_gt = _convert_hmap_object_centric( # convert hmap_gt to hmap_gt_oc.
                hmap_gt, outputs['box_s_raw'], outputs['box_s_val'], o)
        outputs['hmap_gt_oc'] = hmap_gt # To visualize hmap_gt (object-centric) in summary.

        if o.heatmap_stride != 1:
            hmap_gt, unmerge = merge_dims(hmap_gt, 0, 2)
            hmap_gt = slim.avg_pool2d(hmap_gt,
                kernel_size=o.heatmap_stride+1,
                stride=o.heatmap_stride,
                padding='SAME')
            hmap_gt = unmerge(hmap_gt, 0)

        y_is_valid = example['y_is_valid']
        losses = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in o.losses or 'l1_relative' in o.losses:
            assert False
            if 'y_pred_oc' in outputs:
                y_pred = outputs['y_pred_oc']
            else:
                y_pred = outputs['y_pred']
            y_gt_valid = tf.boolean_mask(y_gt, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            loss_l1 = tf.reduce_mean(tf.abs(y_gt_valid - y_pred_valid), axis=-1)
            if 'l1' in o.losses:
                losses['l1'] = tf.reduce_mean(loss_l1)
            if 'l1_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:,2] - y_gt_valid[:,0])
                y_size = tf.abs(y_gt_valid[:,3] - y_gt_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                loss_l1_relative = loss_l1 / (tf.reduce_mean(size, axis=-1) + 0.05)
                losses['l1_relative'] = tf.reduce_mean(loss_l1_relative)

        # CLE (center location error). Measured in l2 distance.
        if 'cle' in o.losses or 'cle_relative' in o.losses:
            assert False
            if 'y_pred_oc' in outputs:
                y_pred = outputs['y_pred_oc']
            else:
                y_pred = outputs['y_pred']
            y_gt_valid = tf.boolean_mask(y_gt, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            x_center = (y_gt_valid[:,2] + y_gt_valid[:,0]) * 0.5
            y_center = (y_gt_valid[:,3] + y_gt_valid[:,1]) * 0.5
            center = tf.stack([x_center, y_center], axis=-1)
            x_pred_center = (y_pred_valid[:,2] + y_pred_valid[:,0]) * 0.5
            y_pred_center = (y_pred_valid[:,3] + y_pred_valid[:,1]) * 0.5
            pred_center = tf.stack([x_pred_center, y_pred_center], axis=-1)
            loss_cle = tf.norm(center - pred_center, axis=-1)
            if 'cle' in o.losses:
                losses['cle'] = tf.reduce_mean(loss_cle)
            if 'cle_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:,2] - y_gt_valid[:,0])
                y_size = tf.abs(y_gt_valid[:,3] - y_gt_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                radius = tf.exp(tf.reduce_mean(tf.log(size), axis=-1))
                loss_cle_relative = loss_cle / (radius + 0.05)
                losses['cle_relative'] = tf.reduce_mean(loss_cle_relative)

        # Cross-entropy between probabilty maps (need to change label)
        if 'ce' in o.losses or 'ce_balanced' in o.losses:
            if 'hmap_pred_oc' in outputs:
                hmap_pred = outputs['hmap_pred_oc']
            else:
                assert False, 'This should not happen in obj-centric approach.'
                hmap_pred = outputs['hmap_pred']
            print 'hmap_pred.shape:', hmap_pred.shape.as_list()
            print 'hmap_gt.shape:', hmap_gt.shape.as_list()
            hmap_gt_valid = tf.boolean_mask(hmap_gt, y_is_valid)
            hmap_pred_valid = tf.boolean_mask(hmap_pred, y_is_valid)
            # hmap is [valid_images, height, width, 2]
            count = tf.reduce_sum(hmap_gt_valid, axis=(1, 2), keep_dims=True)
            class_weight = 0.5 / tf.cast(count+1, tf.float32)
            weight = tf.reduce_sum(hmap_gt_valid * class_weight, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                    labels=hmap_gt_valid,
                    logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            if 'ce' in o.losses:
                losses['ce'] = tf.reduce_mean(loss_ce)
            if 'ce_balanced' in o.losses:
                losses['ce_balanced'] = tf.reduce_mean(
                        tf.reduce_sum(weight * loss_ce, axis=(1, 2)))

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

        return tf.reduce_sum(losses.values(), name=scope), outputs


def _draw_bounding_boxes(example, model, time_stride=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x_raw']   -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # model.outputs['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        image = (1.0/255)*example['x_raw'][0][::time_stride]
        y_gt   = example['y'][0][::time_stride]
        y_pred = model.outputs['y_pred'][0][::time_stride]
        image  = tf.concat((tf.expand_dims(model.state['x'][0][0], 0),  image), 0) # add init frame
        y_gt   = tf.concat((tf.expand_dims(model.state['y'][0][0], 0),   y_gt), 0) # add init y_gt
        y_pred = tf.concat((tf.expand_dims(model.state['y'][0][0], 0), y_pred), 0) # add init y_gt for pred too
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        return tf.image.draw_bounding_boxes(image, boxes, name=scope)

def _draw_heatmap(model, kind, time_stride=1, name='draw_heatmap'):
    with tf.name_scope(name) as scope:
        # model.outputs['hmap'] -- [b, t, frmsz, frmsz, 2]
        p = model.outputs[kind][0,::time_stride]
        if kind == 'hmap_pred_oc':
            p = tf.nn.softmax(p)
        hmaps = tf.concat((tf.expand_dims(model.state['hmap'][0][0], 0), p[:,:,:,0:1]), 0) # add init hmap
        return hmaps
        #return tf.cast(tf.round(255*hmaps), tf.uint8) # convert to int.

def _draw_input_image(model, key, time_stride=1, name='draw_input_image'):
    with tf.name_scope(name) as scope:
        input_image = model.outputs[key][0,::time_stride]
        return input_image

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
        'x0_raw'     # First image in sequence, shape [h, w, 3]
        'y0'         # Position of target in first image, shape [4]
        'x_raw'      # Input images, shape [n-1, h, w, 3]
        'y'          # Position of target in following frames, shape [n-1, 4]
        'y_is_valid' # Booleans indicating presence of frame, shape [n-1]
    '''
    seq_len = len(seq['image_files'])
    assert(len(seq['labels']) == seq_len)
    assert(len(seq['label_is_valid']) == seq_len)
    assert(seq['label_is_valid'][0] == True)
    # TODO: Use o.dtype here? Numpy complains.
    f = lambda x: im_to_arr(load_image(x, size=(o.frmsz, o.frmsz), resize=False),
                            dtype=np.float32)
    images = map(f, seq['image_files'])
    return {
        'x0_raw':     np.array(images[0]),
        'y0':         np.array(seq['labels'][0]),
        'x_raw':      np.array(images[1:]),
        'y':          np.array(seq['labels'][1:]),
        'y_is_valid': np.array(seq['label_is_valid'][1:]),
    }

def _load_batch(seqs, o):
    sequence_keys = set(['x_raw', 'y', 'y_is_valid'])
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
