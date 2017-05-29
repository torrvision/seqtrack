import unittest

import numpy as np
import os
from PIL import Image
import tempfile
import tensorflow as tf

import evaluate
import model as models
import train

class TestTrack(unittest.TestCase):

    def test_different_ntimesteps(self):
        frmsz = 241
        ntimesteps = [5, 20]
        sequence_len = 58

        tmp = tempfile.mkdtemp()
        model_file = os.path.join(tmp, 'model')
        sequence = random_sequence(dir_name=tmp, num_frames=sequence_len, frmsz=frmsz)

        trajectories = []
        for i in range(len(ntimesteps)):
            tf.reset_default_graph()
            example = train._make_placeholders(ntimesteps=ntimesteps[i], frmsz=frmsz, dtype=tf.float32)
            preproc = train._whiten(example, dtype=tf.float32, stat={'mean': 0.0, 'std': 1.0})
            # model = models.mlp(preproc, ntimesteps=ntimesteps[i], frmsz=frmsz)
            model = models.simple_search(preproc, ntimesteps=ntimesteps[i], frmsz=frmsz, use_rnn=False)
            # model = models.rnn_multi_res(preproc, ntimesteps=ntimesteps[i], frmsz=frmsz)
            saver = tf.train.Saver()
            init_op = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init_op)
                if i == 0:
                    saver.save(sess, model_file)
                else:
                    saver.restore(sess, model_file)
                traj = evaluate.track(sess, example, model, sequence)
                trajectories.append(traj)
                if i > 0:
                    for t in range(sequence_len-1):
                        print 'frame {}'.format(t)
                        print traj[t]
                        print trajectories[0][t]
                        np.testing.assert_allclose(traj[t], trajectories[0][t], rtol=1e-3,
                            err_msg='frame {}'.format(t))


def random_sequence(dir_name, num_frames, frmsz):
    seq = {
        'image_files':    [],
        'labels':         [],
        'label_is_valid': [],
        'original_image_size': (frmsz, frmsz),
    }
    for t in range(num_frames):
        im = np.uint8(np.random.random_integers(0, 255, size=(frmsz, frmsz, 3)))
        im = Image.fromarray(im)
        image_file = os.path.join(dir_name, '{:06d}.jpeg'.format(t))
        im.save(image_file)
        p0 = np.random.rand(2)
        p1 = np.random.rand(2)
        p0, p1 = np.minimum(p0, p1), np.maximum(p0, p1)
        label = np.concatenate([p0, p1])
        seq['image_files'].append(image_file)
        seq['labels'].append(label)
        seq['label_is_valid'].append(True)
    return seq
