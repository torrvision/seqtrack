import pdb
import sys
import numpy as np
import tensorflow as tf
import time

import draw


def get_optimizer(m, o):
    lr = tf.placeholder(o.dtype, shape=[])
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(\
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

def train(m, loader, o):

    #TODO: add the followings (not urgent though..)
    # summary op 
    # rich resume 

    optimizer, lr = get_optimizer(m, o)

    saver = tf.train.Saver()

    losses = {
            'batch': np.array([], dtype=np.float32),
            'epoch': np.array([], dtype=np.float32)
            }

    '''
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = o.gpu_frac
    with tf.Session(config=config if o.gpu_manctrl else None) as sess:
    '''
    t_total = time.time()
    with tf.Session(config=o.tfconfig) as sess:
        sess.run(tf.initialize_all_variables())
        #sess.run(tf.global_variables_initializer()) # hmm.. is this better?
        if o.resume: 
            saver.restore(sess, o.resume_model)
        
        for ie in range(o.nepoch if not o.debugmode else 3):
            t_epoch = time.time()
            lr_epoch = o.lr*(0.1**np.floor(float(ie)/(o.nepoch/2))) \
                    if o.lr_update else o.lr # TODO: may need a different recipe 

            loss_curr_ep = np.array([], dtype=np.float32) # for avg epoch loss
            for ib in range(loader.ntr/o.batchsz if not o.debugmode else 100):
                t_batch = time.time()
                batch = loader.get_batch(ib, o, data_='train')
                #loader.run_sanitycheck(batch) # TODO: run if change dataset

                fdict = {
                        m.net['inputs']: batch['inputs'],
                        m.net['inputs_length']: batch['inputs_length'],
                        m.net['labels']: batch['labels'],
                        lr: lr_epoch
                        }
                _, loss = sess.run([optimizer, m.net['loss']], feed_dict=fdict)

                # results after every batch 
                sys.stdout.write(
                        '\rep {0:d}/{1:d}, batch {2:5d}/{3:5d}] '
                        '(BATCH) |loss:{4:.3f} |time:{5:.2f}'\
                                .format(
                                    ie+1, o.nepoch, 
                                    ib+1, loader.ntr/o.batchsz, 
                                    loss, time.time()-t_batch))
                sys.stdout.flush()

                losses['batch'] = np.append(losses['batch'], loss)
                loss_curr_ep = np.append(loss_curr_ep, loss) 
            print ' '

            # after every epoch
            # 1. save the model
            # 2. save the loss
            # 3. print results
            if not o.nosave:
                save_path = saver.save(sess, 
                        o.path_model+'/ep{}.ckpt'.format(ie))
            losses['epoch'] = np.append(losses['epoch'], np.mean(loss_curr_ep))
            print 'ep {0:d}/{1:d} (EPOCH) |loss:{2:.3f} |time:{3:.2f}'\
                    .format(ie+1, o.nepoch, 
                            losses['epoch'][-1], time.time()-t_epoch)

        # print after all epoches
        print 'total time elapsed:{.2f}'.format(time.time()-t_total)
            

        # plot losses
        pdb.set_trace()
        draw.plot_losses(losses, o)
        m.update_network(m.net) # TODO: verify if this step is necessary



