import pdb
import tensorflow as tf

from evaluate import evaluate


def test(m, loader, o):
    '''
    Note that it is considered that this wrapper serves a test routine with a 
    completely trained model. If you want a on-the-fly evaluations during 
    training, consider carefully which session you will use.
    '''

    saver = tf.train.Saver()

    with tf.Session(config=o.tfconfig) as sess:
        saver.restore(sess, o.restore_model)

        # TODO: need to have a separate evaluation routine
        evaluate(sess, m, loader, o, data_='test')


