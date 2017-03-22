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


def iter_examples(loader, o, dstype='train', num_epochs=None):
    '''Generator that produces examples for SGD.'''
    if num_epochs:
        epochs = xrange(num_epochs)
    else:
        epochs = itertools.count()
    for i in epochs:
        loader.update_epoch_begin(dstype)
        for j in range(loader.nexps[dstype]):
            yield loader.get_example(j, o, dstype=dstype)


def get_example_files(capacity=32, name='get_example'):
    '''Creates an example queue that contains filenames not images.

    Returns:
        The Tensor at the front of the queue and
        a function handle that feeds an iterable collection of examples to the queue.
    '''
    with tf.name_scope(name) as scope:
        # Create queue to write examples to.
        queue = tf.FIFOQueue(capacity=capacity,
                             dtypes=[tf.string, tf.float32],
                             names=['files', 'labels'],
                             name='file_queue')
        example_var = {
            'files':  tf.placeholder(tf.string, shape=[None], name='example_files'),
            'labels': tf.placeholder(tf.float32, shape=[None, 4], name='example_labels'),
        }
        enqueue = queue.enqueue(example_var)
        with tf.name_scope('summary'):
            tf.summary.scalar('fraction_of_%d_full' % capacity,
                              tf.cast(queue.size(), tf.float32) * (1./capacity))
        dequeue = queue.dequeue(name=scope)

    def feed_loop(sess, coord, examples):
        '''Enqueues examples in a loop.
        This should be run in another thread.

        Args:
        examples is an iterable collection of dictionaries.
        '''
        for example in examples:
            if coord.should_stop():
                return
            sess.run(enqueue, feed_dict={
                example_var['files']:  example['files'],
                example_var['labels']: example['labels'],
            })
        coord.request_stop()

    return dequeue, feed_loop


def load_images(example, capacity=32, num_threads=1, name='load_images'):
    # Follow structure of tf.train.batch().
    with tf.name_scope(name) as scope:
        # Create queue to write images to.
        queue = tf.FIFOQueue(capacity=capacity,
                             dtypes=[tf.uint8, tf.float32],
                             names=['images', 'labels'],
                             name='image_queue')
        # Read files from disk.
        file_contents = tf.map_fn(tf.read_file, example['files'], dtype=tf.string)
        # Decode images.
        images = tf.map_fn(tf.image.decode_jpeg, file_contents, dtype=tf.uint8)
        # Replace files with images.
        del example['files']
        example['images'] = images
        enqueue = queue.enqueue(example)
        tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue]*num_threads))
        with tf.name_scope('summary'):
            tf.summary.scalar('fraction_of_%d_full' % capacity,
                              tf.cast(queue.size(), tf.float32) * (1./capacity))
        return queue.dequeue(name=scope)


def make_input_pipeline(o, stat=None, dtype=tf.float32,
        sequence_capacity=128, batch_capacity=32, num_threads=8, name='pipeline'):
    with tf.name_scope(name) as scope:
        example_files, feed_loop = get_example_files(capacity=sequence_capacity)
        example_images = load_images(example_files, capacity=sequence_capacity, num_threads=num_threads)
        # Restore rank information of Tensors for tf.train.batch.
        # TODO: It does not seem possible to preserve this through the FIFOQueue?
        # Let at least the sequence length remain dynamic.
        example_images['images'].set_shape([None, None, None, 3])
        example_images['labels'].set_shape([None, 4])
        # Get the length of the sequence before tf.train.batch.
        example_images['num_frames'] = tf.shape(example_images['images'])[0]
        # TODO: This may produce batches of length < ntimesteps+1
        # since PaddingFIFOQueue pads to the *maximum* length.
        # Is this a problem for the training code?
        batch_images = tf.train.batch(example_images, batch_size=o.batchsz,
            dynamic_pad=True, capacity=batch_capacity, num_threads=num_threads)
        # Set static dimension of sequence length.
        # TODO: This may only be necessary due to how the model is written.
        batch_images['images'].set_shape([None, o.ntimesteps+1, None, None, None])
        batch_images['labels'].set_shape([None, o.ntimesteps+1, None])
        # Cast type of images.
        batch_images['images'] = tf.cast(batch_images['images'], o.dtype)
        # Put in format expected by model.
        valid = (range(o.ntimesteps+1) < tf.expand_dims(batch_images['num_frames'], -1))
        batch = {
            'inputs_raw':   batch_images['images'],
            'labels':       batch_images['labels'],
            'x0_raw':       batch_images['images'][:, 0],
            'y0':           batch_images['labels'][:, 0],
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
        batch['inputs'] = _whiten(batch['inputs_raw'], mean, std, name='inputs')
        del batch['inputs_raw']
        batch['x0'] = _whiten(batch['x0_raw'], mean, std, name='x0')
        del batch['x0_raw']
        return batch, feed_loop


def _whiten(x, mean, std, name='whiten'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)


def train(m, feed_loop, loader, o):
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
    optimize_op = optimizer.minimize(m.net['loss'], global_step=global_step_var)

    init_op = tf.global_variables_initializer()
    summary_var_eval = tf.summary.merge_all()
    # Optimization summary might include gradients, learning rate, etc.
    summary_var_opt = tf.summary.merge([summary_var_eval,
        tf.summary.scalar('lr', lr)])
    saver = tf.train.Saver()

    train_examples = iter_examples(loader, o, dstype='train', num_epochs=None)

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
        t = threading.Thread(target=feed_loop, args=(sess, coord, train_examples))
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
            loader.update_epoch_begin('train')
            loader.update_epoch_begin('val')
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
                    _, loss, summary = sess.run([optimize_op, m.net['loss'], summary_var_opt])
                    train_writer.add_summary(summary, global_step=global_step)
                else:
                    _, loss = sess.run([optimize_op, m.net['loss']])
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


# def _get_lr_recipe():
#     # TODO: may need a different recipe; also consider exponential decay
#     # (previous) lr_epoch = o.lr*(0.1**np.floor(float(ie)/(nepoch/2))) \
#             #if o.lr_update else o.lr
#     # manual learning rate recipe
#     lr_recipe = np.zeros([100], dtype=np.float32)
#     for i in range(lr_recipe.shape[0]):
#         if i < 5:
#             lr_recipe[i] = 0.0001*(0.1**i) # TODO: check if this is alright
#         else:
#             lr_recipe[i] = lr_recipe[4]
#     return lr_recipe

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
