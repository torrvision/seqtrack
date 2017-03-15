import pdb
import sys
import numpy as np
import tensorflow as tf
import time
import os

import draw
from evaluate import evaluate
import helpers


def train(m, loader, o):
    # (manual) learning rate recipe
    lr_recipe = _get_lr_recipe()
    optimizer, global_step_var, lr = _get_optimizer(m, o)

    nepoch     = o.nepoch if not o.debugmode else 2
    nbatch     = loader.nexps['train']/o.batchsz if not o.debugmode else 30
    nbatch_val = loader.nexps['val']/o.batchsz if not o.debugmode else 30

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf.summary.scalar('loss', m.net['loss'])
    # tf.summary.histogram('output', m.net['outputs'])
    summary_op = tf.summary.merge_all()

    def process_batch(batch, step, optimize=True, writer=None, write_summary=False):
        names = ['target_raw', 'inputs_raw', 'x0_raw', 'y0', 'inputs_valid', 'inputs_HW', 'labels']
        fdict = {m.net[name]: batch[name] for name in names}
        if optimize:
            fdict.update({
                lr: lr_epoch,
            })
        start = time.time()
        summary = None
        if optimize:
            if write_summary:
                _, loss, summary = sess.run([optimizer, m.net['loss'], summary_op], feed_dict=fdict)
            else:
                _, loss = sess.run([optimizer, m.net['loss']], feed_dict=fdict)
        else:
            if write_summary:
                loss, summary = sess.run([m.net['loss'], summary_op], feed_dict=fdict)
            else:
                loss = sess.run([m.net['loss']], feed_dict=fdict)
        dur = time.time() - start
        if write_summary:
            writer.add_summary(summary, global_step=step)
        return loss, dur


    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        # Either initialize or restore model.
        if o.resume:
            model_file = tf.train.latest_checkpoint(o.path_ckpt)
            print "restore: {}".format(model_file)
            saver.restore(sess, model_file)
        else:
            sess.run(init_op)


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
            loader.update_epoch_begin('train')
            loader.update_epoch_begin('val')
            lr_epoch = lr_recipe[ie] if o.lr_update else o.lr

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

                # # **after a certain iteration, perform the followings
                # # - evaluate on train/test/val set
                # # - print results (loss, eval resutls, time, etc.)
                # if global_step % (o.period_assess if not o.debugmode else 20) == 0: # save intermediate model
                #     print ' '
                #     # evaluate
                #     val_ = 'test' if o.dataset == 'bouncing_mnist' else 'val'
                #     evals = {
                #         'train': evaluate(sess, m, loader, o, 'train',
                #             np.maximum(int(np.floor(100/o.batchsz)), 1),
                #             hold_inputs=True, shuffle_local=True),
                #         val_: evaluate(sess, m, loader, o, val_,
                #             np.maximum(int(np.floor(100/o.batchsz)), 1),
                #             hold_inputs=True, shuffle_local=True)}
                #     # visualize tracking results examples
                #     draw.show_track_results(
                #         evals['train'], loader, 'train', o, global_step,nlimit=20)
                #     draw.show_track_results(
                #         evals[val_], loader, val_, o, global_step,nlimit=20)
                #     # print results
                #     print 'ep {:d}/{:d} (STEP-{:d}) '\
                #         '|(train/{:s}) IOU: {:.3f}/{:.3f}, '\
                #         'AUC: {:.3f}/{:.3f}, CLE: {:.3f}/{:.3f} '.format(
                #         ie+1, nepoch, global_step+1, val_,
                #         evals['train']['iou_mean'], evals[val_]['iou_mean'],
                #         evals['train']['auc'],      evals[val_]['auc'],
                #         evals['train']['cle_mean'], evals[val_]['cle_mean'])

                # Take a training step.
                start = time.time()
                batch = loader.get_batch(ib, o, dstype='train')
                load_dur = time.time() - start

                loss, dur = process_batch(batch, step=global_step, optimize=True,
                    writer=train_writer,
                    write_summary=(ib % o.summary_period == 0))

                # **results after every batch
                print ('ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                    '|loss:{5:.5f} |time:{6:.2f} ({7:.2f})').format(
                    ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur, load_dur)

                # Evaluate validation error.
                if ib % o.val_period == 0:
                    # Only if (ib / nbatch) >= (ib_val / nbatch_val), or equivalently
                    if ib * nbatch_val >= ib_val * nbatch:
                        start = time.time()
                        batch = loader.get_batch(ib_val, o, dstype='val')
                        load_dur = time.time() - start
                        loss, dur = process_batch(batch, step=global_step, optimize=False,
                            writer=val_writer, write_summary=True)
                        print ('[val] ep {0:d}/{1:d}, batch {2:d}/{3:d} (BATCH:{4:d}) '
                            '|loss:{5:.5f} |time:{6:.2f} ({7:.2f})').format(
                            ie+1, nepoch, ib+1, nbatch, o.batchsz, loss, dur, load_dur)
                        ib_val += 1

            print 'ep {0:d}/{1:d} (EPOCH) |time:{2:.2f}'.format(
                    ie+1, nepoch, time.time()-t_epoch)

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)


def _get_lr_recipe():
    # TODO: may need a different recipe; also consider exponential decay
    # (previous) lr_epoch = o.lr*(0.1**np.floor(float(ie)/(nepoch/2))) \
            #if o.lr_update else o.lr
    # manual learning rate recipe
    lr_recipe = np.zeros([100], dtype=np.float32)
    for i in range(lr_recipe.shape[0]):
        if i < 5:
            lr_recipe[i] = 0.0001*(0.1**i) # TODO: check if this is alright
        else:
            lr_recipe[i] = lr_recipe[4]
    return lr_recipe

def _get_optimizer(m, o):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.placeholder(o.dtype, shape=[])
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(\
                m.net['loss'], global_step=global_step)
    elif o.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(\
                m.net['loss'], global_step=global_step)
    elif o.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(\
                m.net['loss'], global_step=global_step)
    elif o.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(\
                m.net['loss'], global_step=global_step)
    else:
        raise ValueError('optimizer not implemented or simply wrong.')
    return optimizer, global_step, lr
