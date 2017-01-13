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

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=o.gpu_frac)
    with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options)
            if o.gpu_manctrl else None) as sess:
        saver.restore(sess, o.restore_model)

        # TODO: need to have a separate evaluation routine
        evaluate(sess, m, loader, o, data_='test')


