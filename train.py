import pdb
import sys
import tensorflow as tf


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
    # track loss 
    # rich resume 
    # set random seed 

    optimizer, lr = get_optimizer(m, o)

    saver = tf.train.Saver()

    # TODO: for accurate memory allocation, better to know a rough model size
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=o.gpu_frac)
    with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
            if o.gpu_manctrl else None) as sess:
        sess.run(tf.initialize_all_variables())
        #sess.run(tf.global_variables_initializer()) # hmm.. is this better?
        if o.resume:
            saver.restore(sess, o.resume_model)

        for ie in range(o.nepoch if not o.debugmode else 1):
            lr_epoch = o.lr*(0.1**np.floor(float(ie)/(o.nepoch/2))) \
                    if o.lr_update else o.lr # TODO: may need a different recipe 

            for ib in range(loader.ntr/o.batchsz if not o.debugmode else 100):
                batch, _ = loader.load_batch(ib, o, data_='train')
                #loader.run_sanitycheck(batch) # TODO: run if change dataset

                fdict = {
                        m.net['inputs']: batch['inputs'],
                        m.net['inputs_length']: batch['inputs_length'],
                        m.net['labels']: batch['labels'],
                        lr: lr_epoch
                        }
                _, loss = sess.run([optimizer, m.net['loss']], feed_dict=fdict)

                #print 'ep {0:d}/{1:d}, batch{2:5d}/{3:5d}] loss:{4:.3f}'.format(
                        #ie, o.nepoch, ib, loader.ntr/o.batchsz, loss)
                sys.stdout.write(
                        '\rep {0:d}/{1:d}, batch {2:5d}/{3:5d}] loss:{4:.3f}'\
                                .format(
                                    ie+1, o.nepoch, 
                                    ib+1, loader.ntr/o.batchsz, 
                                    loss))
                sys.stdout.flush()

            if not o.nosave:
                save_path = saver.save(sess, 
                        o.path_model+'/ep{}.ckpt'.format(ie))

        m.update_network(net) # TODO: verify if this step is necessary



