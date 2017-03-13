import pdb
import sys
import numpy as np
import tensorflow as tf
import time
import os
#import os.path

import draw
from evaluate import evaluate
import helpers


def train(m, loader, o):

    train_opts = _init_train_settings(m, loader, o)
    nepoch          = train_opts['nepoch']
    nbatch          = train_opts['nbatch']
    lr_recipe       = train_opts['lr_recipe']
    losses          = train_opts['losses']
    iteration       = train_opts['iteration']
    ep_start        = train_opts['ep_start']
    resume_model    = train_opts['resume_model']
    optimizer       = train_opts['optimizer']
    lr              = train_opts['lr']

    saver = tf.train.Saver()

    tf.summary.scalar('loss', m.net['loss'])
    # tf.summary.histogram('output', m.net['outputs'])
    summary_op = tf.summary.merge_all()


    '''
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = o.gpu_frac
    with tf.Session(config=config if o.gpu_manctrl else None) as sess:
    '''
    t_total = time.time()
    t_iteration = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        sess.run(tf.global_variables_initializer()) 

        path_summary_train = os.path.join(o.path_summary, 'train')
        if not os.path.exists(path_summary_train): 
            helpers.mkdir_p(path_summary_train)
        train_writer = tf.summary.FileWriter(path_summary_train, sess.graph)
        #train_writer = tf.summary.FileWriter(os.path.join(o.path_logs, 'train'), sess.graph)

        if o.resume: 
            saver.restore(sess, resume_model)
        
        for ie in range(ep_start, nepoch):
            t_epoch = time.time()
            loader.update_epoch_begin('train')
            lr_epoch = lr_recipe[ie] if o.lr_update else o.lr

            loss_curr_ep = np.array([], dtype=np.float32) # for avg epoch loss
            for ib in range(nbatch):
                t_batch = time.time()
                batch = loader.get_batch(ib, o, dstype='train')

                fdict = {
                        m.net['target']: batch['target'],
                        m.net['inputs']: batch['inputs'],
                        m.net['inputs_valid']: batch['inputs_valid'],
                        m.net['inputs_HW']: batch['inputs_HW'],
                        m.net['labels']: batch['labels'],
                        lr: lr_epoch
                        }
                if ib % 10 == 0:
                    _, loss, summary, _ = sess.run([optimizer, m.net['loss'], summary_op, m.net['dbg']], feed_dict=fdict)
                    train_writer.add_summary(summary, ib)
                else:
                    _, loss, _ = sess.run([optimizer, m.net['loss'], m.net['dbg']], feed_dict=fdict)

                # **results after every batch 
                sys.stdout.write(
                        '\rep {0:d}/{1:d}, batch {2:d}/{3:d} '
                        '(BATCH:{4:d}) |loss:{5:.5f} |time:{6:.2f}'\
                                .format(
                                    ie+1, nepoch, ib+1, nbatch, o.batchsz,
                                    loss, time.time()-t_batch))
                sys.stdout.flush()
                losses['batch'] = np.append(losses['batch'], loss)
                loss_curr_ep = np.append(loss_curr_ep, loss) 
                losses['interm'] = np.append(losses['interm'], loss)

                iteration += 1
                # **after a certain iteration, perform the followings
                # - record and plot the loss
                # - evaluate on train/test/val set
                # - check eval losses on train and val sets
                # - print results (loss, eval resutls, time, etc.)
                # - save the model and resume info 
                assess_period = 20000/o.batchsz
                if iteration % (assess_period if not o.debugmode else 20) == 0: # save intermediate model
                    print ' '
                    # record and plot (intermediate) loss
                    loss_interm_avg = np.mean(losses['interm'])
                    losses['interm_avg'] = np.append(
                            losses['interm_avg'], loss_interm_avg)
                    losses['interm'] = np.array([], dtype=np.float32)
                    draw.plot_losses(losses, o, True, str(iteration))
                    # evaluate
                    val_ = 'test' if o.dataset == 'bouncing_mnist' else 'val'
                    evals = {
                        'train': evaluate(sess, m, loader, o, 'train', 
                            np.maximum(int(np.floor(100/o.batchsz)), 1), 
                            hold_inputs=True, shuffle_local=True),
                        val_: evaluate(sess, m, loader, o, val_, 
                            np.maximum(int(np.floor(100/o.batchsz)), 1), 
                            hold_inputs=True, shuffle_local=True)}
                    # check losses on train and val set
                    losses['interm_eval_subset_train'] = np.append(
                            losses['interm_eval_subset_train'],
                            np.mean(evals['train']['loss']))
                    losses['interm_eval_subset_val'] = np.append(
                            losses['interm_eval_subset_val'],
                            np.mean(evals[val_]['loss']))
                    draw.plot_losses_train_val(
                            losses['interm_eval_subset_train'],
                            losses['interm_eval_subset_val'],
                            o, str(iteration))
                    # visualize tracking results examples
                    draw.show_track_results(
                        evals['train'], loader, 'train', o, iteration,nlimit=20)
                    draw.show_track_results(
                        evals[val_], loader, val_, o, iteration,nlimit=20)
                    # print results
                    print 'ep {0:d}/{1:d} (ITERATION-{2:d}) |loss: {3:.5f} '\
                        '|(train/{4:s}) IOU: {5:.3f}/{6:.3f}, '\
                        'AUC: {7:.3f}/{8:.3f}, CLE: {9:.3f}/{10:.3f} '\
                        '|time:{11:.2f}'.format(
                        ie+1, nepoch, iteration, loss_interm_avg, val_, 
                        evals['train']['iou_mean'], evals[val_]['iou_mean'], 
                        evals['train']['auc'], evals[val_]['auc'], 
                        evals['train']['cle_mean'], evals[val_]['cle_mean'], 
                        time.time()-t_iteration)
                    t_iteration = time.time() 
                    # save model and resume info
                    if not o.nosave:
                        savedir = os.path.join(o.path_save, 'models')
                        if not os.path.exists(savedir): helpers.mkdir_p(savedir)
                        saved_model = saver.save(sess, os.path.join(
                            savedir, 'iteration{}.ckpt'.format(iteration)))
                        resume = {}
                        resume['ie'] = ie
                        resume['iteration'] = iteration
                        resume['losses'] = losses
                        resume['model'] = saved_model
                        np.save(o.path_save + '/resume.npy', resume)
            print ' '
            print 'ep {0:d}/{1:d} (EPOCH) |time:{2:.2f}'.format(
                    ie+1, nepoch, time.time()-t_epoch)

            '''Not using below as performing evaluation at iterations
            # **after every epoch, perform the followings
            # - record the loss
            # - plot losses
            # - evaluate on train/test/val set
            # - print results (loss, eval results, time, etc.)
            # - save the model
            # - save resume 
            losses['epoch'] = np.append(losses['epoch'], np.mean(loss_curr_ep))
            draw.plot_losses(losses, o) # TODO: change this. batch loss too long
            val_name = 'test' if o.dataset == 'bouncing_mnist' else 'val'
            eval_results = {
                    'train': evaluate(sess, m, loader, o, 'train', 0.01),
                    val_name: evaluate(sess, m, loader, o, val_name, 0.01)
                    }
            print 'ep {0:d}/{1:d} (EPOCH) |loss:{2:.5f} |(train/{3:s}) '\
                    'IOU: {4:.3f}/{5:.3f}, AUC: {6:.3f}/{7:.3f}, '\
                    'CLE: {8:.3f}/{9:.3f} |time:{10:.2f}'.format(
                    ie+1, nepoch, losses['epoch'][-1], val_name,
                    eval_results['train']['iou'], eval_results[val_name]['iou'], 
                    eval_results['train']['auc'], eval_results[val_name]['auc'], 
                    eval_results['train']['cle'], eval_results[val_name]['cle'], 
                    time.time()-t_epoch)

            if not o.nosave:
                save_path = saver.save(sess, 
                        o.path_model+'/ep{}.ckpt'.format(ie))
                resume = {}
                resume['ie'] = ie
                resume['iteration'] = iteration 
                resume['losses'] = losses
                resume['model'] = o.path_model+'/ep{}.ckpt'.format(ie)
                np.save(o.path_save + '/resume.npy', resume)
            '''

        # **training finished
        print '\ntraining finished! ------------------------------------------'
        print 'total time elapsed: {0:.2f}'.format(time.time()-t_total)
        

def _init_train_settings(m, loader, o):
    # resuming experiments
    if o.resume: 
        resume = np.load(o.resume_data).item()
        losses          = resume['losses']
        iteration       = resume['iteration']
        ep_start        = resume['ie'] + 1
        resume_model    = resume['model']
    else:
        losses = {
                'batch': np.array([], dtype=np.float32),
                'epoch': np.array([], dtype=np.float32),
                'interm': np.array([], dtype=np.float32),
                'interm_avg': np.array([], dtype=np.float32),
                'interm_eval_subset_train': np.array([], dtype=np.float32),
                'interm_eval_subset_val': np.array([], dtype=np.float32)
                }
        iteration = 0
        resume_model = None
        ep_start = 0

    # (manual) learning rate recipe
    lr_recipe = _get_lr_recipe()

    # optimizer
    optimizer, lr = _get_optimizer(m, o)

    train_opts = {
            'nepoch': o.nepoch if not o.debugmode else 2,
            'nbatch': loader.nexps['train']/o.batchsz if not o.debugmode else 300,
            'lr_recipe': lr_recipe,
            'losses': losses,
            'iteration': iteration,
            'ep_start': ep_start,
            'resume_model': resume_model,
            'optimizer': optimizer,
            'lr': lr
            }
    return train_opts

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
    lr = tf.placeholder(o.dtype, shape=[])
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(\
                m.net['loss'])
    elif o.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9).minimize(\
                m.net['loss'])
    elif o.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(\
                m.net['loss'])
    elif o.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(\
                m.net['loss'])
    else:
        raise ValueError('optimizer not implemented or simply wrong.')
    return optimizer, lr

