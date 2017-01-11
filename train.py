import pdb
import tensorflow as tf


def get_optimizer(o):
    lr = tf.placeholder(o.dtype, shape=[])
    if o.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(model.net['loss'])
    elif o.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(model.net['loss'])
    elif o.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(model.net['loss'])
    else:
        raise ValueError('optimizer not implemented or simply wrong.')
    return optimizer

def train(m, data_tr, data_va, o):

    # select optimizer
    # training summary (for what? find out - might be a big help later)
    # creates session?
    # update learning rate 
    # load batch (shuffle, augmentation, possibly data processing)
    # run 
    # track loss 
    # save model

    optimizer = get_optimizer(o)

    with tf.Session as sess:
        # if not loading a pre-trained model, initialize!
        sess.run(tf.initialize_all_variables())

        for ie in range(o.nepoch):
            # update learning rate
            lr_epoch = o.lr*(0.1**np.floor(float(ie)/(o.nepoch/2))) \
                    if o.lr_update else o.lr 

            # mini-batch training
            for b in range():

                loss = sess.run(
                        optimzier, m.net['loss'],
                        feed_dict={
                            m.net['inputs']: ,
                            m.net['inputs_length']: ,
                            m.net['targets']: ,
                            lr = lr_epoch
                            }
                        )
                print loss


    return net
