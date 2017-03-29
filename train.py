import pdb
import sys
import numpy as np
import tensorflow as tf
import time
import os
import itertools
import threading

import draw
from evaluate import evaluate
import helpers
import pipeline


def train(model, loader, o):
    '''
    The model is a function that takes as input a dictionary of tensors and
    returns a dictionary of tensors.

    The reason that the model is provided as a *function* is so that
    the code which uses the model is free to decide how to instantiate it.
    For example, training code may construct a single instance of the model with input placeholders,
    or it may construct two instances of the model, each with its own input queue.

    The model should use `tf.get_variable` rather than `tf.Variable` to facilitate variable sharing between multiple instances.
    The model will be used in the same manner as an input to `tf.make_template`.

    The input dictionary has fields::

        'inputs'       # Input images, shape [b, n+1, h, w, 3]
        'inputs_valid' # Booleans indicating presence of frame, shape [b, n]
        'x0'           # First image in sequence, shape [b, h, w, 3]
        'y0'           # Position of target in first image, shape [b, 4]
        'inputs_HW'    # Tensor of image sizes, shape [b, 2]

    and the output dictionary has fields::

        'y'       # (optional) Predicted position of target in each frame, shape [b, n, 4]
        'heatmap' # (optional) Score for pixel belonging to target, shape [b, n, h, w, 1]

    The images provided to the model are already normalized (e.g. dataset mean subtracted).
    '''

    model = tf.make_template('model', model)
    # Create one instance of the model for training.
    example_train, feed_loop_train = make_input_pipeline(o, stat=loader.stat['train'])
    output_train = model(example_train)
    # Create another instance of the model for validation.
    example_val, feed_loop_val = make_input_pipeline(o, stat=loader.stat['train'])
    output_val = model(example_train)
    # Create loss for each instance of the model (should not have any trainable parameters).
    loss_train = get_loss(example_train, output_train, o)
    loss_val = get_loss(example_val, output_val, o)

    nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = loader.nexps['train']/o.batchsz if not o.debugmode else 30
    nbatch_val = loader.nexps['val']/o.batchsz if not o.debugmode else 30

    global_step_var = tf.Variable(0, name='global_step', trainable=False)
    # lr = init * decay^(step)
    #    = init * decay^(step / period * period / decay_steps)
    #    = init * [decay^(period / decay_steps)]^(step / period)
    lr = tf.train.exponential_decay(o.lr_init, global_step_var,
                                    decay_steps=o.lr_decay_steps,
                                    decay_rate=o.lr_decay_rate,
                                    staircase=True)
    optimizer = _get_optimizer(lr, o)
    optimize_op = optimizer.minimize(loss_train, global_step=global_step_var)

    init_op = tf.global_variables_initializer()
    summary_var_eval = tf.summary.merge_all()
    # Optimization summary might include gradients, learning rate, etc.
    summary_var_opt = tf.summary.merge([summary_var_eval,
        tf.summary.scalar('lr', lr)])
    saver = tf.train.Saver()

    examples_train = iter_examples(loader, o, dstype='train', num_epochs=None)
    examples_val = iter_examples(loader, o, dstype='val', num_epochs=None)

    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        # Either initialize or restore model.
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            print "restore: {}".format(model_file)
            saver.restore(sess, model_file)
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
        # path_summary_val = os.path.join(o.path_summary, 'val')
        train_writer = tf.summary.FileWriter(path_summary_train, sess.graph)
        # val_writer = tf.summary.FileWriter(path_summary_val)

        while True: # Loop over epochs
            global_step = global_step_var.eval() # Number of steps taken.
            if global_step >= nepoch * nbatch:
                break
            ie = global_step / nbatch
            t_epoch = time.time()
            # loader.update_epoch_begin('train')
            # loader.update_epoch_begin('val')
            # lr_epoch = lr_recipe[ie] if o.lr_update else o.lr

            ib_val = 0
            for ib in range(nbatch): # Loop over batches in epoch.
                global_step = global_step_var.eval() # Number of steps taken.

                if not o.nosave:
                    period_ckpt = o.period_ckpt if not o.debugmode else 40
                    if global_step > 0 and global_step % period_ckpt == 0: # save intermediate model
                        if not os.path.isdir(o.path_ckpt):
                            os.makedirs(o.path_ckpt)
                        fname = os.path.join(o.path_ckpt, 'iteration{}.ckpt'.format(global_step))
                        # saved_model = saver.save(sess, fname)
                        saver.save(sess, fname)

                # **after a certain iteration, perform the followings
                # - evaluate on train/test/val set
                # - print results (loss, eval resutls, time, etc.)
                period_assess = o.period_assess if not o.debugmode else 20
                if global_step > 0 and global_step % period_assess == 0: # evaluate model
                    print ' '
                    # evaluate
                    val_ = 'test' if o.dataset == 'bouncing_mnist' else 'val'
                    evals = {
                        'train': evaluate(sess, m, loader, o, 'train',
                            np.maximum(int(np.floor(100/o.batchsz)), 1),
                            hold_inputs=True, shuffle_local=True),
                        val_: evaluate(sess, m, loader, o, val_,
                            np.maximum(int(np.floor(100/o.batchsz)), 1),
                            hold_inputs=True, shuffle_local=True)}
                    # visualize tracking results examples
                    draw.show_track_results(
                        evals['train'], loader, 'train', o, global_step,nlimit=20)
                    draw.show_track_results(
                        evals[val_], loader, val_, o, global_step,nlimit=20)
                    # print results
                    print 'ep {:d}/{:d} (STEP-{:d}) '\
                        '|(train/{:s}) IOU: {:.3f}/{:.3f}, '\
                        'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} '.format(
                        ie+1, nepoch, global_step+1, val_,
                        evals['train']['iou_mean'], evals[val_]['iou_mean'],
                        evals['train']['auc'],      evals[val_]['auc'],
                        evals['train']['cle_mean'], evals[val_]['cle_mean'])

                # Take a training step.
                # start = time.time()
                # batch = loader.get_batch(ib, o, dstype='train')
                # load_dur = time.time() - start
                # loss, dur = process_batch(batch, step=global_step, optimize=True,
                #     writer=train_writer,
                #     write_summary=(ib % o.summary_period == 0))
                start = time.time()
                # pdb.set_trace()
                if ib % o.summary_period == 0:
                    _, loss, summary = sess.run([optimize_op, loss_train, summary_var_opt])
                    train_writer.add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, loss_train])
                dur = time.time() - start

                # **results after every batch
                print ('ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                    '|loss:{5:.5f} |time:{6:.2f}').format(
                    ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur)

                # # Evaluate validation error.
                # if ib % o.val_period == 0:
                #     # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                #     if ib * nbatch_val >= ib_val * nbatch:
                #         start = time.time()
                #         batch = loader.get_batch(ib_val, o, dstype='val')
                #         load_dur = time.time() - start
                #         loss, dur = process_batch(batch, step=global_step, optimize=False,
                #             writer=val_writer, write_summary=True)
                #         print ('[val] ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                #             '|loss:{5:.5f} |time:{6:.2f} ({7:.2f})').format(
                #             ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur, load_dur)
                #         ib_val += 1

            print 'ep {0:d}/{1:d} (EPOCH) |time:{2:.2f}'.format(
                    ie+1, nepoch, time.time()-t_epoch)

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


def _make_input_pipeline(o, stat=None, dtype=tf.float32,
        sequence_capacity=128, batch_capacity=32, num_threads=8, name='pipeline'):
    with tf.name_scope(name) as scope:
        files, feed_loop = pipeline.get_example_filenames(capacity=sequence_capacity)
        images = pipeline.load_images(files, capacity=sequence_capacity, num_threads=num_threads)
        images_batch = pipeline.batch(images, batch_size=o.batchsz,
            capacity=batch_capacity, num_threads=num_threads)

        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        images_batch['images'].set_shape([None, o.ntimesteps+1, None, None, None])
        images_batch['labels'].set_shape([None, o.ntimesteps+1, None])
        # Cast type of images.
        images_batch['images'] = tf.cast(images_batch['images'], o.dtype)
        # Put in format expected by model.
        valid = (range(o.ntimesteps+1) < tf.expand_dims(images_batch['num_frames'], -1))
        inputs_batch = {
            'inputs_raw':   images_batch['images'],
            'labels':       images_batch['labels'],
            'x0_raw':       images_batch['images'][:, 0],
            'y0':           images_batch['labels'][:, 0],
            'inputs_valid': valid,
            'inputs_HW':    [o.frmsz, o.frmsz],
        }
        # Normalize mean and variance.
        assert(stat is not None)
        with tf.name_scope('image_stats') as scope:
            if stat:
                mean = tf.constant(stat['mean'], o.dtype, name='mean')
                std  = tf.constant(stat['std'],  o.dtype, name='std')
            else:
                mean = tf.constant(0.0, o.dtype, name='mean')
                std  = tf.constant(1.0, o.dtype, name='std')
        # Replace raw images with whitened images.
        inputs_batch['inputs'] = _whiten(inputs_batch['inputs_raw'], mean, std, name='inputs')
        del inputs_batch['inputs_raw']
        inputs_batch['x0'] = _whiten(inputs_batch['x0_raw'], mean, std, name='x0')
        del inputs_batch['x0_raw']
        return inputs_batch, feed_loop


def _whiten(x, mean, std, name='whiten'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)


def iter_examples(loader, o, dstype='train', num_epochs=None):
    '''Generator that produces multiple epochs of examples for SGD.'''
    if num_epochs:
        epochs = xrange(num_epochs)
    else:
        epochs = itertools.count()
    for i in epochs:
        loader.update_epoch_begin(dstype)
        for j in range(loader.nexps[dstype]):
            yield loader.get_example(j, o, dstype=dstype)


# def get_loss(outputs, labels, inputs_valid, inputs_HW, o, outtype, name='loss'):
def get_loss(example, outputs, outtype='rectangle', name='loss'):
    # NOTE: Be careful about length of labels and outputs. 
    # labels and inputs_valid will be of T+1 length, and y0 shouldn't be used.
    labels       = example['labels']
    inputs_valid = example['inputs_valid']
    inputs_HW    = example['inputs_HW']
    assert(labels.get_shape().as_list()[1] == o.ntimesteps+1)

    with tf.name_scope(name) as scope:
        losses = dict()
        
        if outtype == 'rectangle':
            y = outputs['y']
            assert(y.get_shape().as_list()[1] == o.ntimesteps)

            # loss1: sum of two l1 distances for left-top and right-bottom
            if 'l1' in o.losses: # TODO: double check
                labels_valid = tf.boolean_mask(labels[:,1:], inputs_valid[:,1:])
                outputs_valid = tf.boolean_mask(y, inputs_valid[:,1:])
                loss_l1 = tf.reduce_mean(tf.abs(labels_valid - outputs_valid))
                losses['l1'] = loss_l1

            # loss2: IoU
            if 'iou' in o.losses:
                assert(False) # TODO: change from inputs_length to inputs_valid
                scalar = tf.stack((inputs_HW[:,1], inputs_HW[:,0], 
                    inputs_HW[:,1], inputs_HW[:,0]), axis=1)
                boxA = y * tf.expand_dims(scalar, 1)
                boxB = labels[:,1:,:] * tf.expand_dims(scalar, 1)
                xA = tf.maximum(boxA[:,:,0], boxB[:,:,0])
                yA = tf.maximum(boxA[:,:,1], boxB[:,:,1])
                xB = tf.minimum(boxA[:,:,2], boxB[:,:,2])
                yB = tf.minimum(boxA[:,:,3], boxB[:,:,3])
                interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)
                boxAArea = (boxA[:,:,2] - boxA[:,:,0]) * (boxA[:,:,3] - boxA[:,:,1]) 
                boxBArea = (boxB[:,:,2] - boxB[:,:,0]) * (boxB[:,:,3] - boxB[:,:,1]) 
                # TODO: CHECK tf.div or tf.divide
                #iou = tf.div(interArea, (boxAArea + boxBArea - interArea) + 1e-4)
                iou = interArea / (boxAArea + boxBArea - interArea + 1e-4) 
                iou_valid = []
                for i in range(o.batchsz):
                    iou_valid.append(iou[i, :inputs_length[i]-1])
                iou_mean = tf.reduce_mean(iou_valid)
                loss_iou = 1 - iou_mean # NOTE: Any normalization?
                losses['iou'] = loss_iou

        elif outtype == 'heatmap':
            heatmap = outputs['heatmap']
            assert(heatmap.get_shape().as_list()[1] == o.ntimesteps)

            # First of all, need to convert labels into heat maps
            labels_heatmap = model.convert_rec_to_heatmap(labels, o)

            # valid labels and outputs
            labels_valid = tf.boolean_mask(labels_heatmap[:,1:], inputs_valid[:,1:])
            outputs_valid = tf.boolean_mask(heatmap, inputs_valid[:,1:])

            # loss1: cross-entropy between probabilty maps (need to change label) 
            if 'ce' in o.losses: 
                labels_flat = tf.reshape(labels_valid, [-1, o.frmsz**2])
                outputs_flat = tf.reshape(outputs_valid, [-1, o.frmsz**2])
                assert_finite = lambda x: tf.Assert(tf.reduce_all(tf.is_finite(x)), [x])
                with tf.control_dependencies([assert_finite(outputs_valid)]):
                    outputs_valid = tf.identity(outputs_valid)
                loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=outputs_flat)
                # Wrap with assertion that loss is finite.
                with tf.control_dependencies([assert_finite(loss_ce)]):
                    loss_ce = tf.identity(loss_ce)
                loss_ce = tf.reduce_mean(loss_ce)
                losses['ce'] = loss_ce

            # loss2: tf's l2 (without sqrt)
            if 'l2' in o.losses:
                labels_flat = tf.reshape(labels_valid, [-1, o.frmsz**2])
                outputs_flat = tf.reshape(outputs_valid, [-1, o.frmsz**2])
                outputs_softmax = tf.nn.softmax(outputs_flat)
                loss_l2 = tf.nn.l2_loss(labels_flat - outputs_softmax)
                losses['l2'] = loss_l2

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss)

        return tf.reduce_sum(losses.values(), name=scope)
