import pdb
import sys
import numpy as np
import tensorflow as tf
import time
import os
import itertools
import threading

import draw
import evaluate
import helpers
import pipeline
import sample
import visualize

from model import convert_rec_to_heatmap


def train(create_model, datasets, val_sets, o):
    '''Trains a network.

    Args:
        create_model: Function that takes as input a dictionary of tensors and
            returns a model object.
        datasets: Dictionary of datasets with keys 'train' and 'val'.
        val_sets: A dictionary of collections of sequences on which to evaluate the tracker.

    Returns:
        The results obtained using the tracker at different stages of training.
        This is a dictionary with the same keys as val_sets:
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

        'x'         # Input images, shape [b, n+1, h, w, 3]
        'x0'        # First image in sequence, shape [b, h, w, 3]
        'y0'        # Position of target in first image, shape [b, 4]
        'y_valid'   # Booleans indicating presence of frame, shape [b, n]

    and the output dictionary has fields::

        'y'       # (optional) Predicted position of target in each frame, shape [b, n, 4]
        'heatmap' # (optional) Score for pixel belonging to target, shape [b, n, h, w, 1]

    The images provided to the model are already normalized (e.g. dataset mean subtracted).
    '''

    # How should we compute training and validation error with pipelines?
    # Option 1 is to have multiple copies of the network with shared parameters.
    # However, this makes it difficult to plot training and validation error on the same axes
    # since there are separate summary ops for training and validation (with different names).
    # Option 2 is to use FIFOQueue.from_list()

    modes = ['train', 'val']

    example   = {}
    feed_loop = {}
    # Create a queue for each set.
    with tf.name_scope('input'):
        for mode in modes:
            with tf.name_scope(mode):
                example[mode], feed_loop[mode] = _setup_input(o)

    model     = {}
    loss_vars = {}
    # Always use same statistics for whitening (not set dependent).
    stat = datasets['train'].stat
    # Create copies of model.
    with tf.variable_scope('model', reuse=False) as vs:
        for i, mode in enumerate(modes):
            with tf.name_scope(mode):
                # Necessary to have separate collections of summaries.
                # Otherwise evaluating the summary op will dequeue an example!
                # TODO: Use tf.make_template? But how to have different collections?
                model[mode] = create_model(_whiten(example[mode], o, stat=stat),
                                           is_training=(mode == 'train'),
                                           summaries_collections=['summaries_' + mode])
                # TODO: Ensure that get_loss does not create any variables?
                loss_vars[mode] = get_loss(example[mode], model[mode].outputs, o,
                    summaries_collections=['summaries_' + mode])
                r = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                tf.summary.scalar('regularization', r, collections=['summaries_' + mode])
                loss_vars[mode] += r
                tf.summary.scalar('total', loss_vars[mode], collections=['summaries_' + mode])
                # In next loop, reuse variables.
                vs.reuse_variables()

    summary_vars = {}
    summary_vars_with_preview = {}
    # This must contain only summaries that do not pull from the queues.
    # For example, fraction of pipeline that is full.
    global_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    with tf.name_scope('summary'):
        for mode in modes:
            with tf.name_scope(mode):
                # Merge model summaries with others (e.g. pipeline).
                summaries = (global_summaries + tf.get_collection('summaries_' + mode))
                summary_vars[mode] = tf.summary.merge(summaries)
                # Create a preview and add it to the list of summaries.
                preview = tf.summary.image('box',
                    _draw_bounding_boxes(example[mode], model[mode]),
                    max_outputs=o.ntimesteps+1, collections=[])
                summaries.append(preview)
                # Produce an image summary of the heatmap.
                if 'hmap' in model[mode].outputs:
                    hmap = tf.summary.image('hmap', _draw_heatmap(model[mode]),
                        max_outputs=o.ntimesteps+1, collections=[])
                    summaries.append(hmap)
                summary_vars_with_preview[mode] = tf.summary.merge(summaries)

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
    optimize_op = optimizer.minimize(loss_vars['train'], global_step=global_step_var)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sequences = {mode: iter_examples(datasets[mode], o, num_epochs=None)
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

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        # Run the feed loops in another thread.
        threads = [threading.Thread(target=feed_loop[mode],
                                    args=(sess, coord, sequences[mode]))
                   for mode in modes]
        for t in threads:
            t.start()

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
                        saver.save(sess, fname)
                        print 'done: save model'
                        prev_ckpt = global_step

                # intermediate evaluation of model
                period_assess = o.period_assess if not o.debugmode else 10
                if global_step > 0 and global_step % period_assess == 0:
                    iter_id = 'iteration{}'.format(global_step)
                    for eval_id, sampler in val_sets.iteritems():
                        vis_dir = os.path.join(o.path_output, iter_id, eval_id)
                        if not os.path.isdir(vis_dir): os.makedirs(vis_dir, 0755)
                        visualizer = visualize.VideoFileWriter(vis_dir)
                        # Run the tracker on a full epoch.
                        print 'evaluation: {}'.format(eval_id)
                        sequences = sampler()
                        result = evaluate.evaluate(sess, example['val'], model['val'], sequences,
                            visualize=visualizer.visualize)
                        print 'IOU: {:.3f}, AUC: {:.3f}, CLE: {:.3f}'.format(
                            result['iou_mean'], result['auc'], result['cle_mean'])
                    # print 'ep {:d}/{:d} (STEP-{:d}) '\
                    #     '|(train/{:s}) IOU: {:.3f}/{:.3f}, '\
                    #     'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} '.format(
                    #     ie+1, nepoch, global_step+1, val_,
                    #     evals['train']['iou_mean'], evals[val_]['iou_mean'],
                    #     evals['train']['auc'],      evals[val_]['auc'],
                    #     evals['train']['cle_mean'], evals[val_]['cle_mean'])

                # Take a training step.
                start = time.time()
                if global_step % o.period_summary == 0:
                    summary_var = (summary_vars_with_preview['train']
                                   if global_step % o.period_preview == 0
                                   else summary_vars['train'])
                    _, loss, summary = sess.run([optimize_op, loss_vars['train'], summary_var],
                            feed_dict={example['train']['use_gt']: True})
                    dur = time.time() - start
                    writer['train'].add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, loss_vars['train']],
                            feed_dict={example['train']['use_gt']: True})
                    dur = time.time() - start
                loss_ep.append(loss)

                newval = False
                # Evaluate validation error.
                if global_step % o.period_summary == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        summary_var = (summary_vars_with_preview['val']
                                       if global_step % o.period_preview == 0
                                       else summary_vars['val'])
                        loss_val, summary = sess.run([loss_vars['val'], summary_var],
                                feed_dict={example['val']['use_gt']: True})
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
        example_capacity=512, load_capacity=128, batch_capacity=32,
        num_load_threads=8, num_batch_threads=8,
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


def _setup_input(o):
    def shape_var_batch(x):
        shape = x.shape.as_list()
        return [None] + shape[1:]
    # Tune numbers of threads to adjust which queues are full.
    example_queue, feed_loop = _make_input_pipeline(o,
        num_load_threads=4, num_batch_threads=1)
    # Create placeholders with variable batch size.
    # These can be fed manually in evaluation (instead of reading from queues).
    example = {k: tf.placeholder_with_default(example_queue[k],
                                              shape_var_batch(example_queue[k]))
               for k in example_queue}
    # Add a placeholder for models that use ground-truth during training.
    example['use_gt'] = tf.placeholder_with_default(False, [], name='use_gt')
    return example, feed_loop


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


def iter_examples(dataset, o, num_epochs=None):
    '''Generator that produces multiple epochs of examples for SGD.'''
    if num_epochs:
        epochs = xrange(num_epochs)
    else:
        epochs = itertools.count()
    for i in epochs:
        sequences = sample.sample(dataset, ntimesteps=o.ntimesteps,
            seqtype='freq-range-fit', min_freq=15, max_freq=60, shuffle=True)
        for sequence in sequences:
            yield sequence


def get_loss(example, outputs, o, summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        y          = example['y']
        y_is_valid = example['y_is_valid']
        assert(y.get_shape().as_list()[1] == o.ntimesteps)
        # TODO: What happens with NaN rectangles here?
        # hmap = convert_rec_to_heatmap(y, o, min_size=1.0)
        hmap = convert_rec_to_heatmap(y, o, min_size=1.0)

        losses = dict()

        # loss1: sum of two l1 distances for left-top and right-bottom
        if 'l1' in o.losses: # TODO: double check
            y_pred = outputs['y']
            y_valid = tf.boolean_mask(y, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
            loss_l1 = tf.reduce_mean(tf.abs(y_valid - y_pred_valid))
            losses['l1'] = loss_l1

        # loss1: cross-entropy between probabilty maps (need to change label)
        if 'ce' in o.losses:
            hmap_pred = outputs['hmap']
            hmap_valid = tf.boolean_mask(hmap, y_is_valid)
            hmap_pred_valid = tf.boolean_mask(hmap_pred, y_is_valid)
            loss_ce = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.reshape(hmap_valid, [-1, 2]),
                        logits=tf.reshape(hmap_pred_valid, [-1, 2])))
            losses['ce'] = loss_ce
            #y_flat = tf.reshape(hmap_valid, [-1, o.frmsz**2])
            #y_pred_flat = tf.reshape(hmap_pred_valid, [-1, o.frmsz**2])
            #assert_finite = lambda x: tf.Assert(tf.reduce_all(tf.is_finite(x)), [x])
            #with tf.control_dependencies([assert_finite(hmap_pred_valid)]):
            #    hmap_pred_valid = tf.identity(hmap_pred_valid)
            #loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=y_pred_flat)
            ## Wrap with assertion that loss is finite.
            #with tf.control_dependencies([assert_finite(loss_ce)]):
            #    loss_ce = tf.identity(loss_ce)
            #loss_ce = tf.reduce_mean(loss_ce)
            #losses['ce'] = loss_ce

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

        return tf.reduce_sum(losses.values(), name=scope)


def _draw_bounding_boxes(example, model, time_stride=1, name='draw_box'):
    with tf.name_scope(name) as scope:
        # example['x_raw']   -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # model.outputs['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        image = example['x_raw'][0][::time_stride]
        y_gt = example['y'][0][::time_stride]
        y_pred = model.outputs['y'][0][::time_stride]
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        return tf.image.draw_bounding_boxes(image, boxes, name=scope)

def _draw_heatmap(model, time_stride=1, name='draw_heatmap'):
    with tf.name_scope(name) as scope:
        # model.outputs['hmap'] -- [b, t, frmsz, frmsz, 2]
        return tf.nn.softmax(model.outputs['hmap'][0,::time_stride,:,:,0:1])
