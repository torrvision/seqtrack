import pdb
import sys
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import os
import random
import threading

import draw
import evaluate
import helpers
import pipeline
import sample
import visualize

from model import convert_rec_to_heatmap
from helpers import load_image, im_to_arr, pad_to, cache_json

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

    # Always use same statistics for whitening (not set dependent).
    stat = datasets['train'].stat
    # TODO: Mask y with use_gt to prevent accidental use.
    model = create_model(_whiten(_guard_labels(example), o, stat=stat))
    loss_var = get_loss(example, model.outputs, o)
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
        optimize_op = optimizer.minimize(loss_var, global_step=global_step_var)

    summary_vars = {}
    summary_vars_with_preview = {}
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        # Create a preview and add it to the list of summaries.
        boxes = tf.summary.image('box',
            _draw_bounding_boxes(example, model),
            max_outputs=o.ntimesteps+1, collections=[])
        image_summaries = [boxes]
        # Produce an image summary of the heatmap.
        if 'hmap' in model.outputs:
            hmap = tf.summary.image('hmap', _draw_heatmap(model),
                max_outputs=o.ntimesteps+1, collections=[])
            image_summaries.append(hmap)
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

    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        print '\ntraining starts! --------------------------------------------'

        # Either initialize or restore model.
        model_file = None
        prev_ckpt = 0
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            if model_file is None:
                print 'could not find checkpoint'
        if model_file:
            print 'restore: {}'.format(model_file)
            saver.restore(sess, model_file)
            print 'done: restore'
            prev_ckpt = global_step_var.eval()
        else:
            sess.run(init_op)

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
                        fname = os.path.join(o.path_ckpt, 'iteration{}.ckpt'.format(global_step))
                        # saved_model = saver.save(sess, fname)
                        print 'save model'
                        #saver.save(sess, fname)
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
                            'iter-{}.json'.format(iter_id))
                        result = cache_json(result_file,
                            lambda: evaluate.evaluate(sess, example, model,
                                eval_sequences, visualize=visualizer.visualize),
                            makedir=True)
                        print 'IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}'.format(
                            result['iou_mean'], result['auc'], result['cle_mean'])

                # Take a training step.
                start = time.time()
                feed_dict = {example['use_gt']:      True,
                             example['is_training']: True}
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
                        feed_dict = {example['use_gt']:      True,  # Match training.
                                     example['is_training']: False} # Do not update bnorm stats.
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
    return example


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


def get_loss(example, outputs, o, summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        y          = example['y']
        y_is_valid = example['y_is_valid']
        assert(y.get_shape().as_list()[1] == o.ntimesteps)
        hmap = convert_rec_to_heatmap(y, o, min_size=1.0)

        losses = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in o.losses:
            y_pred = outputs['y']
            y_valid = tf.boolean_mask(y, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            loss_l1 = tf.reduce_mean(tf.abs(y_valid - y_pred_valid))
            losses['l1'] = loss_l1

        # CLE (center location error). Measured in l2 distance.
        if 'cle' in o.losses:
            y_pred = outputs['y']
            y_valid = tf.boolean_mask(y, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            x_center = (y_valid[:,2] + y_valid[:,0]) * 0.5
            y_center = (y_valid[:,3] + y_valid[:,1]) * 0.5
            x_pred_center = (y_pred_valid[:,2] + y_pred_valid[:,0]) * 0.5
            y_pred_center = (y_pred_valid[:,3] + y_pred_valid[:,1]) * 0.5
            loss_cle = tf.reduce_mean(
                    tf.norm(tf.stack([x_center - x_pred_center, y_center - y_pred_center], axis=1),
                    axis=1))
            losses['cle'] = loss_cle

        # Cross-entropy between probabilty maps (need to change label)
        if 'ce' in o.losses:
            hmap_pred = outputs['hmap']
            hmap_valid = tf.boolean_mask(hmap, y_is_valid)
            hmap_pred_valid = tf.boolean_mask(hmap_pred, y_is_valid)
            loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.reshape(hmap_valid, [-1, 2]),
                        logits=tf.reshape(hmap_pred_valid, [-1, 2])))
            losses['ce'] = loss_ce

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

        return tf.reduce_sum(losses.values(), name=scope)


def _draw_bounding_boxes(example, model, time_stride=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x_raw']   -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # model.outputs['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        image = (1.0/255)*example['x_raw'][0][::time_stride]
        y_gt = example['y'][0][::time_stride]
        y_pred = model.outputs['y'][0][::time_stride]
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        return tf.image.draw_bounding_boxes(image, boxes, name=scope)

def _draw_heatmap(model, time_stride=1, name='draw_heatmap'):
    with tf.name_scope(name) as scope:
        # model.outputs['hmap'] -- [b, t, frmsz, frmsz, 2]
        # return tf.nn.softmax(model.outputs['hmap'][0,::time_stride,:,:,0:1])
        p = tf.nn.softmax(model.outputs['hmap'][0,::time_stride])
        # Take first channel and convert to int.
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
