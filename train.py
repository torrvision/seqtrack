import pdb
import sys
import csv
import itertools
import json
import math
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
import motion
import pipeline
import sample
import visualize

from model import convert_rec_to_heatmap, to_object_centric_coordinate
from helpers import load_image_viewport, im_to_arr, pad_to, cache_json, merge_dims

EXAMPLE_KEYS = ['x0_raw', 'y0', 'x_raw', 'y', 'y_is_valid', 'aspect']


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
        'aspect'     # Aspect ratio (width/height) of original image, shape [b]

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
    loss_var, model.gt = get_loss(example, model.outputs, model.gt, o)
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
        if 'y' in model.outputs:
            boxes = tf.summary.image('box',
                _draw_bounding_boxes(example, model),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries = [boxes]
        # Produce an image summary of the heatmap prediction (ic and oc).
        if 'hmap' in model.outputs:
            for key in model.outputs['hmap']:
                hmap = tf.summary.image('hmap_pred_{}'.format(key),
                    _draw_heatmap(model, pred=True, perspective=key),
                    max_outputs=o.ntimesteps+1, collections=[])
                image_summaries.append(hmap)
        # Produce an image summary of the heatmap gt (ic and oc).
        if 'hmap' in model.gt:
            for key in model.gt['hmap']:
                hmap = tf.summary.image('hmap_gt_{}'.format(key),
                    _draw_heatmap(model, pred=False, perspective=key),
                    max_outputs=o.ntimesteps+1, collections=[])
                image_summaries.append(hmap)
        # Produce an image summary of target and search images (input to CNNs).
        if 'target' in model.outputs and 'search' in model.outputs:
            for key in ['target', 'search']:
                input_image = tf.summary.image('cnn_input_{}'.format(key),
                    _draw_input_image(model, key, o, name='draw_{}'.format(key)),
                    max_outputs=o.ntimesteps+1, collections=[])
                image_summaries.append(input_image)
        # Produce an image summary of s_prev and s_recon.
        if 's_prev' in model.outputs and 's_recon' in model.outputs:
            if model.outputs['s_recon'] is not None:
                for key in ['s_prev', 's_recon']:
                    input_image = tf.summary.image('flow_{}'.format(key),
                        _draw_input_image(model, key, name='draw_{}'.format(key)),
                        max_outputs=o.ntimesteps+1, collections=[])
                    image_summaries.append(input_image)
        # Produce an image summary of flow.
        if 'flow' in model.outputs:
            if model.outputs['flow'] is not None:
                for key in ['u', 'v']:
                    flow_fields = tf.summary.image('flow_{}'.format(key),
                        _draw_flow_fields(model, key, name='draw_{}'.format(key)),
                        max_outputs=o.ntimesteps+1, collections=[])
                    image_summaries.append(flow_fields)
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
                                     generator=np.random.RandomState(o.seed_global),
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
        saver_cl = tf.train.Saver(vars_to_restore)

    if o.cnn_pretrain:
        ''' In case of loading pre-trained CNN (e.g., vgg_16), create a separate
        Saver object that is going to be used to restore when session starts.
        '''
        #from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        #print_tensors_in_checkpoint_file('./pretrained/vgg_16.ckpt', None, False)
        # or
        #from tensorflow.python import pywrap_tensorflow
        #reader = pywrap_tensorflow.NewCheckpointReader('./pretrained/vgg_16.ckpt')
        #var_to_shape_map = reader.get_variable_to_shape_map()
        # Approach 1. Use tf.trainable_variables won't work if variables are non-trainable.
        #vars_to_restore = {v.name.split(':')[0]: v for v in tf.trainable_variables()
        #                   if o.cnn_model in v.name}
        # Approach 2. Use collection to get variables.
        vars_to_restore = {v.name.split(':')[0]: v for v in tf.get_collection(o.cnn_model)}
        saver_external = tf.train.Saver(vars_to_restore)

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
        elif o.cnn_pretrain:
            model_file = os.path.join(o.path_data_home, 'pretrained', '{}.ckpt'.format(o.cnn_model))
            saver_external.restore(sess, model_file)
            #print sess.run(tf.report_uninitialized_variables()) # To check
            # initialize uninitialized variables
            vars_uninit = sess.run(tf.report_uninitialized_variables())
            sess.run(tf.variables_initializer([v for v in tf.global_variables()
                                               if v.name.split(':')[0] in vars_uninit]))
            assert len(sess.run(tf.report_uninitialized_variables())) == 0
        else:
            sess.run(init_op)
            if o.curriculum_learning:
                if o.pretrained_cl is None: # e.g., '/some_path/ckpt/iteration-150000'
                    raise ValueError('could not find checkpoint')
                print 'restore: {}'.format(o.pretrained_cl)
                saver_cl.restore(sess, o.pretrained_cl)
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
                if global_step > 0 and global_step > o.period_skip and global_step % period_assess == 0:
                    iter_id = 'iteration{}'.format(global_step)
                    for eval_id, sampler in eval_sets.iteritems():
                        vis_dir = os.path.join(o.path_output, iter_id, eval_id)
                        if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0755)
                        # visualizer = visualize.VideoFileWriter(vis_dir)
                        # Run the tracker on a full epoch.
                        print 'evaluation: {}'.format(eval_id)
                        eval_sequences = sampler()
                        # eval_sequences = [
                        #     motion.augment(sequence, rand=np.random,
                        #         translate_kind='laplace',
                        #         translate_amount=0.1,
                        #         scale_kind='laplace',
                        #         scale_exp_amount=math.exp(math.log(1.01)/30.))
                        #     for sequence in eval_sequences
                        # ]
                        # Cache the results.
                        result_file = os.path.join(o.path_output, 'assess', eval_id,
                            iter_id+'.json')
                        result = cache_json(result_file,
                            lambda: evaluate.evaluate(sess, example, model, eval_sequences,
                                # visualize=visualizer.visualize if o.save_videos else None,
                                visualize=True, vis_dir=vis_dir,
                                use_gt=o.use_gt_eval,
                                save_frames=o.save_frames),
                            makedir=True)
                        print 'IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}, Prec.@20px: {:.3f}'.format(
                            result['iou_mean'], result['auc'], result['cle_mean'],result['cle_representative'])

                # Take a training step.
                start = time.time()
                feed_dict = {example['use_gt']:      o.use_gt_train,
                             example['is_training']: True,
                             example['gt_ratio']:    max(1.0*np.exp(-o.gt_decay_rate*global_step),
                                                         o.min_gt_ratio)}
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
                    _, loss, summary = sess.run([optimize_op, loss_var, summary_var],
                                                feed_dict=feed_dict)
                    dur = time.time() - start
                    writer['train'].add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, loss_var], feed_dict=feed_dict)
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
            'aspect':     images_batch['aspect'],
        }
        return example_batch, feed_loop


def _make_placeholders(o, default=None):
    shapes = {
        'x0_raw':     [None, o.frmsz, o.frmsz, 3],
        'y0':         [None, 4],
        'x_raw':      [None, o.ntimesteps, o.frmsz, o.frmsz, 3],
        'y':          [None, o.ntimesteps, 4],
        'y_is_valid': [None, o.ntimesteps],
        'aspect':     [None],
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
        #return tf.divide(x - mean, std, name=scope)
        return tf.divide(x - 0.0, 1.0, name=scope)


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
            # JV: Add motion augmentation.
            # yield sequence
            if o.augment_motion:
                sequence = motion.augment(sequence, rand=generator, **o.motion_params)
            yield sequence

def get_loss(example, outputs, gt, o, summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        y_gt    = {'ic': None, 'oc': None}
        hmap_gt = {'ic': None, 'oc': None}

        y_gt['ic'] = example['y']
        y_gt['oc'] = to_object_centric_coordinate(example['y'], outputs['box_s_raw'], outputs['box_s_val'], o)
        hmap_gt['oc'] = convert_rec_to_heatmap(y_gt['oc'], o, min_size=1.0, **o.heatmap_params)
        hmap_gt['ic'] = convert_rec_to_heatmap(y_gt['ic'], o, min_size=1.0, **o.heatmap_params)

        # Regress displacement rather than absolute location. Update y_gt.
        if outputs['boxreg_delta']:
            y_gt['ic'] = y_gt['ic'] - tf.concat([tf.expand_dims(example['y0'],1), y_gt['ic'][:,:o.ntimesteps-1]],1) 
            delta0 = y_gt['oc'][:,0] - tf.stack([0.5 - 1./o.search_scale/2., 0.5 + 1./o.search_scale/2.]*2)
            y_gt['oc'] = tf.concat([tf.expand_dims(delta0,1), y_gt['oc'][:,1:]-y_gt['oc'][:,:o.ntimesteps-1]], 1)

        assert(y_gt['ic'].get_shape().as_list()[1] == o.ntimesteps)

        for key in outputs['hmap_interm']:
            if outputs['hmap_interm'][key] is not None:
                pred_size = outputs['hmap_interm'][key].shape.as_list()[2:4]
                hmap_interm_gt, unmerge = merge_dims(hmap_gt['oc'], 0, 2)
                hmap_interm_gt = tf.image.resize_images(hmap_interm_gt, pred_size,
                        method=tf.image.ResizeMethod.BILINEAR,
                        align_corners=True)
                hmap_interm_gt = unmerge(hmap_interm_gt, axis=0)
                break

        if 'oc' in outputs['hmap']:
            # Resize GT heatmap to match size of prediction if necessary.
            pred_size = outputs['hmap']['oc'].shape.as_list()[2:4]
            assert all(pred_size) # Must not be None.
            gt_size = hmap_gt['oc'].shape.as_list()[2:4]
            if gt_size != pred_size:
                hmap_gt['oc'], unmerge = merge_dims(hmap_gt['oc'], 0, 2)
                hmap_gt['oc'] = tf.image.resize_images(hmap_gt['oc'], pred_size,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True)
                hmap_gt['oc'] = unmerge(hmap_gt['oc'], axis=0)

        losses = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in o.losses or 'l1_relative' in o.losses:
            y_gt_valid   = tf.boolean_mask(y_gt[o.perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][o.perspective], example['y_is_valid'])
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
            y_gt_valid   = tf.boolean_mask(y_gt[o.perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][o.perspective], example['y_is_valid'])
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
            hmap_gt_valid   = tf.boolean_mask(hmap_gt[o.perspective], example['y_is_valid'])
            hmap_pred_valid = tf.boolean_mask(outputs['hmap'][o.perspective], example['y_is_valid'])
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

        for key in outputs['hmap_interm']:
            if outputs['hmap_interm'][key] is not None:
                hmap_gt_valid   = tf.boolean_mask(hmap_interm_gt, example['y_is_valid'])
                hmap_pred_valid = tf.boolean_mask(outputs['hmap_interm'][key], example['y_is_valid'])
                # Flatten to feed into softmax_cross_entropy_with_logits.
                hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
                hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
                loss_ce_interm = tf.nn.softmax_cross_entropy_with_logits(
                        labels=hmap_gt_valid,
                        logits=hmap_pred_valid)
                loss_ce_interm = unmerge(loss_ce_interm, 0)
                losses['ce_{}'.format(key)] = tf.reduce_mean(loss_ce_interm)

        # Reconstruction loss using generalized Charbonnier penalty
        if 'recon' in o.losses:
            alpha = 0.25
            s_prev_valid  = tf.boolean_mask(outputs['s_prev'],  example['y_is_valid'])
            s_recon_valid = tf.boolean_mask(outputs['s_recon'], example['y_is_valid'])
            charbonnier_penalty = tf.pow(tf.square(s_prev_valid - s_recon_valid) + 1e-10, alpha)
            losses['recon'] = tf.reduce_mean(charbonnier_penalty)

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

        #gt['y']    = {'ic': y_gt['ic'],    'oc': y_gt['oc']}
        gt['hmap'] = {'ic': hmap_gt['ic'], 'oc': hmap_gt['oc']} # for visualization in summary.
        return tf.reduce_sum(losses.values(), name=scope), gt

def _draw_bounding_boxes(example, model, time_stride=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x_raw']   -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # model.outputs['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        image = (1.0/255)*example['x_raw'][0][::time_stride]
        y_gt   = example['y'][0][::time_stride]
        y_pred = model.outputs['y']['ic'][0][::time_stride]
        # TODO: Do not use model.state here. Breaks encapsulation.
        # JV: Use new model state format.
        # image  = tf.concat((tf.expand_dims(model.state['x'][0][0], 0),  image), 0) # add init frame
        # y_gt   = tf.concat((tf.expand_dims(model.state['y'][0][0], 0),   y_gt), 0) # add init y_gt
        # y_pred = tf.concat((tf.expand_dims(model.state['y'][0][0], 0), y_pred), 0) # add init y_gt for pred too
        image  = tf.concat((tf.expand_dims(model.state_init['x'][0], 0),  image), 0) # add init frame
        y_gt   = tf.concat((tf.expand_dims(model.state_init['y'][0], 0),   y_gt), 0) # add init y_gt
        y_pred = tf.concat((tf.expand_dims(model.state_init['y'][0], 0), y_pred), 0) # add init y_gt for pred too
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        return tf.image.draw_bounding_boxes(image, boxes, name=scope)

def _draw_heatmap(model, pred, perspective, time_stride=1, name='draw_heatmap'):
    with tf.name_scope(name) as scope:
        # model.outputs['hmap'] -- [b, t, frmsz, frmsz, 2]
        if pred:
            p = model.outputs['hmap'][perspective][0,::time_stride]
            if perspective == 'oc':
                p = tf.nn.softmax(p)
        else:
            p = model.gt['hmap'][perspective][0,::time_stride]

        # JV: Not sure what model.state['hmap'] is, and...
        # JV: concat() fails when hmap is coarse (lower resolution than input image).
        # hmaps = tf.concat((model.state['hmap'][0][0:1], p[:,:,:,0:1]), 0) # add init hmap
        hmaps = p[:,:,:,0:1]
        # Convert to uint8 for absolute scale.
        hmaps = tf.image.convert_image_dtype(hmaps, tf.uint8)
        return hmaps

def _draw_flow_fields(model, key, time_stride=1, name='draw_flow_fields'):
    with tf.name_scope(name) as scope:
        if key == 'u':
            input_image = tf.expand_dims(model.outputs['flow'][0,::time_stride, :, :, 0], -1)
        elif key =='v':
            input_image = tf.expand_dims(model.outputs['flow'][0,::time_stride, :, :, 1], -1)
        else:
            assert False , 'No available flow fields'
        return input_image

def _draw_input_image(model, key, o, time_stride=1, name='draw_input_image'):
    with tf.name_scope(name) as scope:
        input_image = model.outputs[key][0,::time_stride]
        if key == 'search':
            if model.outputs['boxreg_delta']:
                y_pred_delta = model.outputs['y']['oc'][0][::time_stride]
                y_pred = y_pred_delta + tf.stack([0.5 - 1./o.search_scale/2., 0.5 + 1./o.search_scale/2.]*2)
            else:
                y_pred = model.outputs['y']['oc'][0][::time_stride]
            coords = tf.unstack(y_pred, axis=1)
            boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=1)
            boxes = tf.expand_dims(boxes, 1)
            return tf.image.draw_bounding_boxes(input_image, boxes, name=scope)
        else:
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
        'aspect'         # Tensor with shape [] containing aspect ratio.
    Example has keys:
        'x0_raw'     # First image in sequence, shape [h, w, 3]
        'y0'         # Position of target in first image, shape [4]
        'x_raw'      # Input images, shape [n-1, h, w, 3]
        'y'          # Position of target in following frames, shape [n-1, 4]
        'y_is_valid' # Booleans indicating presence of frame, shape [n-1]
        'aspect'     # Aspect ratio of original image.
    '''
    seq_len = len(seq['image_files'])
    assert(len(seq['labels']) == seq_len)
    assert(len(seq['label_is_valid']) == seq_len)
    assert(seq['label_is_valid'][0] == True)
    # f = lambda x: im_to_arr(load_image(x, size=(o.frmsz, o.frmsz), resize=False),
    #                         dtype=np.float32)
    images = [
        im_to_arr(load_image_viewport(
            seq['image_files'][t],
            seq['viewports'][t],
            size=(o.frmsz, o.frmsz)))
        for t in range(seq_len)
    ]
    return {
        'x0_raw':     np.array(images[0]),
        'y0':         np.array(seq['labels'][0]),
        'x_raw':      np.array(images[1:]),
        'y':          np.array(seq['labels'][1:]),
        'y_is_valid': np.array(seq['label_is_valid'][1:]),
        'aspect':     seq['aspect'],
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

def generate_report(samplers, datasets, o, metrics=['iou_mean', 'auc', 'cle_mean', 'cle_representative']):
    '''Finds all results for each evaluation distribution.

    Identifies the best result for each metric.
    Caution: More frequent evaluations might lead to better results.
    '''
    def helper():
        eval_id_fn = lambda sampler, dataset: '{}-{}'.format(dataset, sampler)
        best_fn = {'iou_mean': np.amax, 'auc': np.amax, 'cle_mean': np.amin, 'cle_representative': np.amax}
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
