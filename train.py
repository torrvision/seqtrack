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


def train(create_model, datasets, o):
    '''Trains a network.

    The `create_model` function takes as input a dictionary of tensors and
    returns a model object.

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

    # Tune these numbers to adjust which queues are full.
    NUM_LOAD_THREADS = 4
    NUM_BATCH_THREADS = 1
    NUM_MULTIPLEX_THREADS = 1

    with tf.name_scope('pipeline'):
        # Create a queue for each set.
        example_train, feed_loop_train = _make_input_pipeline(o,
            num_load_threads=NUM_LOAD_THREADS, num_batch_threads=NUM_BATCH_THREADS,
            name='train')
        example_val, feed_loop_val = _make_input_pipeline(o,
            num_load_threads=NUM_LOAD_THREADS, num_batch_threads=NUM_BATCH_THREADS,
            name='val')
        queue_index, example = pipeline.make_multiplexer([example_train, example_val],
            num_threads=NUM_MULTIPLEX_THREADS)

    # Create placeholders with variable batch size.
    clear_batch_size = lambda shape: [None] + shape[1:]
    example = {k: tf.placeholder_with_default(example[k],
                      clear_batch_size(example[k].shape.as_list()))
               for k in example}

    # Take subset of `example` fields to give inputs.
    inputs = {k: example[k] for k in ['x_raw', 'x0_raw', 'y0']}
    model = create_model(_whiten(inputs, o, stat=datasets['train'].stat))
    loss_var = get_loss(example, model.outputs, o)

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
    optimizer = _get_optimizer(lr, o)
    optimize_op = optimizer.minimize(loss_var, global_step=global_step_var)

    init_op = tf.global_variables_initializer()
    summary_evaluate = tf.summary.merge_all()
    # Optimization summary might include gradients, learning rate, etc.
    summary_optimize = tf.summary.merge([summary_evaluate,
        tf.summary.scalar('lr', lr)])
    saver = tf.train.Saver()

    examples_train = iter_examples(datasets['train'], o, num_epochs=None)
    examples_val = iter_examples(datasets['val'], o, num_epochs=None)

    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        print '\ntraining starts! --------------------------------------------'

        # Either initialize or restore model.
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            print 'restore: {}'.format(model_file)
            saver.restore(sess, model_file)
            print 'done: restore'
        else:
            sess.run(init_op)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)
        # Run the feed_loop in another thread.
        threads = [
            threading.Thread(target=feed_loop_train, args=(sess, coord, examples_train)),
            threading.Thread(target=feed_loop_val, args=(sess, coord, examples_val)),
        ]
        for t in threads:
            t.start()

        path_summary_train = os.path.join(o.path_summary, 'train')
        path_summary_val = os.path.join(o.path_summary, 'val')
        train_writer = tf.summary.FileWriter(path_summary_train, sess.graph)
        val_writer = tf.summary.FileWriter(path_summary_val)

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
                    if global_step > 0 and global_step % period_ckpt == 0: # save intermediate model
                        if not os.path.isdir(o.path_ckpt):
                            os.makedirs(o.path_ckpt)
                        fname = os.path.join(o.path_ckpt, 'iteration{}.ckpt'.format(global_step))
                        # saved_model = saver.save(sess, fname)
                        print 'save model'
                        saver.save(sess, fname)
                        print 'done: save model'

                # **after a certain iteration, perform the followings
                # - evaluate on train/test/val set
                # - print results (loss, eval resutls, time, etc.)
                period_assess = o.period_assess if not o.debugmode else 20
                if global_step > 0 and global_step % period_assess == 0: # evaluate model
                    # # evaluate
                    # val_ = 'test' if o.dataset == 'bouncing_mnist' else 'val'
                    # evals = {
                    #     'train': evaluate.evaluate(sess, output, state, loader, o, 'train',
                    #         np.maximum(int(np.floor(100/o.batchsz)), 1),
                    #         hold_inputs=True, shuffle_local=True),
                    #     val_: evaluate.evaluate(sess, m, loader, o, val_,
                    #         np.maximum(int(np.floor(100/o.batchsz)), 1),
                    #         hold_inputs=True, shuffle_local=True)}
                    # # visualize tracking results examples
                    # draw.show_track_results(
                    #     evals['train'], loader, 'train', o, global_step,nlimit=20)
                    # draw.show_track_results(
                    #     evals[val_], loader, val_, o, global_step,nlimit=20)
                    # # print results
                    # print 'ep {:d}/{:d} (STEP-{:d}) '\
                    #     '|(train/{:s}) IOU: {:.3f}/{:.3f}, '\
                    #     'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} '.format(
                    #     ie+1, nepoch, global_step+1, val_,
                    #     evals['train']['iou_mean'], evals[val_]['iou_mean'],
                    #     evals['train']['auc'],      evals[val_]['auc'],
                    #     evals['train']['cle_mean'], evals[val_]['cle_mean'])

                    # Run the tracker on a full epoch.
                    evals_batch = {}
                    for s in ['train', 'val']:
                        sequences = sample.sample_ILSVRC(datasets[s], o.ntimesteps,
                            seqtype='sampling', shuffle=False)
                        print 'sample sequences'
                        sequences = [next(sequences) for _ in range(20)]
                        print 'done: sample sequences'
                        print 'len(sequences):', len(sequences)
                        sequences = sequences[:20]
                        evals_batch[s] = evaluate.evaluate(sess, inputs, model, sequences)
                    print ('(train/val) IOU: {:.3f}/{:.3f}, '
                        'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} ').format(
                        evals_batch['train']['iou_mean'], evals_batch['val']['iou_mean'],
                        evals_batch['train']['auc'],      evals_batch['val']['auc'],
                        evals_batch['train']['cle_mean'], evals_batch['val']['cle_mean'])

                    # Run the tracker on whole sequences.
                    # TODO: Add other datasets.
                    evals_full = {}
                    for s in ['train', 'val']:
                        sequences = sample.all_tracks_full_ILSVRC(datasets[s])
                        print 'sample sequences'
                        sequences = [next(sequences) for _ in range(20)]
                        print 'done: sample sequences'
                        evals_full[s] = evaluate.evaluate(sess, inputs, model, sequences)
                    print ('(train/val) IOU: {:.3f}/{:.3f}, '
                        'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} ').format(
                        evals_full['train']['iou_mean'], evals_full['val']['iou_mean'],
                        evals_full['train']['auc'],      evals_full['val']['auc'],
                        evals_full['train']['cle_mean'], evals_full['val']['cle_mean'])
                    # TODO: Draw

                # Take a training step.
                start = time.time()
                if ib % o.period_summary == 0:
                    _, loss, summary = sess.run([optimize_op, loss_var, summary_optimize],
                                                feed_dict={queue_index: 0})
                    dur = time.time() - start
                    train_writer.add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, loss_var], feed_dict={queue_index: 0})
                    dur = time.time() - start
                loss_ep.append(loss)

                # **results after every batch
                if o.verbose_train:
                    print ('ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                        '|loss:{5:.5f} |time:{6:.2f}').format(
                        ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur)

                # Evaluate validation error.
                if ib % o.period_summary == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        _, loss, summary = sess.run([optimize_op, loss_var, summary_evaluate],
                                                    feed_dict={queue_index: 1})
                        dur = time.time() - start
                        val_writer.add_summary(summary, global_step=global_step)
                        if o.verbose_train:
                            print ('[val] ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                                '|loss:{5:.5f} |time:{6:.2f}').format(
                                ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur)
                        ib_val += 1

            print 'ep {:d}/{:d} (global_step: {:d}) |loss: {:.4f}, time:{:.2f}'.format(
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
        images = pipeline.load_images(files, capacity=load_capacity, num_threads=num_load_threads)
        images_batch = pipeline.batch(images, batch_size=o.batchsz,
            capacity=batch_capacity, num_threads=num_batch_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, o.ntimesteps+1, None, None, None])
        images_batch['labels'].set_shape([None, o.ntimesteps+1, None])
        # Cast type of images.
        images_batch['images'] = tf.cast(images_batch['images'], o.dtype)
        # Put in format expected by model.
        is_valid = (range(1, o.ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        inputs_batch = {
            'x_raw':        images_batch['images'][:, 1:],
            'y':            images_batch['labels'][:, 1:],
            'x0_raw':       images_batch['images'][:, 0],
            'y0':           images_batch['labels'][:, 0],
            'y_is_valid':   is_valid,
        }
        return inputs_batch, feed_loop


def _whiten(example_raw, o, stat=None, name='whiten'):
    with tf.name_scope(name) as scope:
        # Normalize mean and variance.
        assert(stat is not None)
        # with tf.variable_scope('image_stats'):
        if stat:
            mean = tf.constant(stat['mean'], o.dtype, name='mean')
            std  = tf.constant(stat['std'],  o.dtype, name='std')
        else:
            mean = tf.constant(0.0, o.dtype, name='mean')
            std  = tf.constant(1.0, o.dtype, name='std')
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
        sequences = sample.sample_ILSVRC(dataset, o.ntimesteps,
            seqtype='sampling', shuffle=True)
        for sequence in sequences:
            yield sequence


def get_loss(example, outputs, o, outtype='rectangle', name='loss'):
    y           = example['y']
    y_is_valid  = example['y_is_valid']
    assert(y.get_shape().as_list()[1] == o.ntimesteps)

    with tf.name_scope(name) as scope:
        losses = dict()
        
        if outtype == 'rectangle':
            y_pred = outputs['y']
            assert(y_pred.get_shape().as_list()[1] == o.ntimesteps)

            # loss1: sum of two l1 distances for left-top and right-bottom
            if 'l1' in o.losses: # TODO: double check
                y_valid = tf.boolean_mask(y, y_is_valid)
                y_pred_valid = tf.boolean_mask(y_pred, y_is_valid)
                loss_l1 = tf.reduce_mean(tf.abs(y_valid - y_pred_valid))
                losses['l1'] = loss_l1

        elif outtype == 'heatmap':
            y_pred_heatmap = outputs['heatmap']
            assert(y_pred_heatmap.get_shape().as_list()[1] == o.ntimesteps)

            # First of all, need to convert y into heat maps
            y_heatmap = model.convert_rec_to_heatmap(y, o)

            # valid y and y_pred
            y_valid = tf.boolean_mask(y_heatmap, y_is_valid)
            y_pred_valid = tf.boolean_mask(y_pred_heatmap, y_is_valid)

            # loss1: cross-entropy between probabilty maps (need to change label) 
            if 'ce' in o.losses: 
                y_flat = tf.reshape(y_valid, [-1, o.frmsz**2])
                y_pred_flat = tf.reshape(y_pred_valid, [-1, o.frmsz**2])
                assert_finite = lambda x: tf.Assert(tf.reduce_all(tf.is_finite(x)), [x])
                with tf.control_dependencies([assert_finite(y_pred_valid)]):
                    y_pred_valid = tf.identity(y_pred_valid)
                loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_flat, logits=y_pred_flat)
                # Wrap with assertion that loss is finite.
                with tf.control_dependencies([assert_finite(loss_ce)]):
                    loss_ce = tf.identity(loss_ce)
                loss_ce = tf.reduce_mean(loss_ce)
                losses['ce'] = loss_ce

            # loss2: tf's l2 (without sqrt)
            if 'l2' in o.losses:
                y_flat = tf.reshape(y_valid, [-1, o.frmsz**2])
                y_pred_flat = tf.reshape(y_pred_valid, [-1, o.frmsz**2])
                y_pred_softmax = tf.nn.softmax(y_pred_flat)
                loss_l2 = tf.nn.l2_loss(y_flat - y_pred_softmax)
                losses['l2'] = loss_l2

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss)

        return tf.reduce_sum(losses.values(), name=scope)
