import pdb
import sys
import numpy as np
import tensorflow as tf
import time

import draw
from evaluate import evaluate


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

    '''
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = o.gpu_frac
    with tf.Session(config=config if o.gpu_manctrl else None) as sess:
    '''
    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        sess.run(tf.global_variables_initializer()) 
        if o.resume: 
            saver.restore(sess, resume_model)
        
        for ie in range(ep_start, nepoch):
            loader.update_epoch_begin('train')
            t_epoch = time.time()
            #lr_epoch = o.lr*(0.1**np.floor(float(ie)/(nepoch/2))) \
                    #if o.lr_update else o.lr 
            # using another (manual) recipe
            lr_epoch = lr_recipe[ie] if o.lr_update else o.lr
            # TODO: may need a different recipe; also consider exponential decay

            loss_curr_ep = np.array([], dtype=np.float32) # for avg epoch loss
            for ib in range(nbatch):
                t_batch = time.time()
                batch = loader.get_batch(ib, o, dstype='train')
                #loader.run_sanitycheck(batch) # TODO: run if change dataset

                fdict = {
                        m.net['inputs']: batch['inputs'],
                        m.net['inputs_length']: batch['inputs_length'],
                        m.net['labels']: batch['labels'],
                        lr: lr_epoch
                        }
                _, loss = sess.run([optimizer, m.net['loss']], feed_dict=fdict)

                # **results after every batch 
                sys.stdout.write(
                        '\rep {0:d}/{1:d}, batch {2:d}/{3:d} '
                        '(BATCH) |loss:{4:.6f} |time:{5:.2f}'\
                                .format(
                                    ie+1, nepoch, ib+1, nbatch,
                                    loss, time.time()-t_batch))
                sys.stdout.flush()
                losses['batch'] = np.append(losses['batch'], loss)
                loss_curr_ep = np.append(loss_curr_ep, loss) 
                losses['interm'] = np.append(losses['interm'], loss)

                iteration += 1
                if iteration % 1000 == 0: # check regularly
                    # draw average loss so far
                    losses['interm_avg'] = np.append(
                            losses['interm_avg'], np.mean(losses['interm']))
                    draw.plot_losses(losses, o, True, str(iteration))
                    losses['interm'] = np.array([], dtype=np.float32)
            print ' '

            # **after every epoch, perform the followings
            # - record the loss
            # - plot losses
            # - evaluate on train/test/val set
            # - print results (loss, eval results, time, etc.)
            # - save the model
            # - save resume 
            losses['epoch'] = np.append(losses['epoch'], np.mean(loss_curr_ep))
            draw.plot_losses(losses, o)
            val_name = 'test' if o.dataset == 'bouncing_mnist' else 'val'
            eval_results = {
                    'train': evaluate(sess, m, loader, o, 'train', 0.01),
                    'test': evaluate(sess, m, loader, o, val_name, 0.01)
                    }
            print 'ep {0:d}/{1:d} (EPOCH) |loss:{2:.6f} |IOU (train/test): '\
            '{3:.3f}/{4:.3f} |time:{5:.2f}'.format(
                    ie+1, nepoch, losses['epoch'][-1], 
                    eval_results['train']['IOU'], eval_results['test']['IOU'], 
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
                'interm_avg': np.array([], dtype=np.float32)
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

