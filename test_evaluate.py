import unittest

import functools
import numpy as np
import os
from PIL import Image
import shutil
import tempfile
import tensorflow as tf

import evaluate
import model as model_pkg
import opts
import train

class TestTrack(unittest.TestCase):

    def test_different_ntimesteps(self):
        '''This test may fail on GPU due to accumulation of errors.'''
        frmsz        = 241
        ntimesteps   = [4, 7]
        sequence_len = 1 + 25 # Includes first frame.
        # Which model to test?
        # TODO: Nice way to test all models?
        # model_fn = functools.partial(model_pkg.Nornn)
        model_fn = functools.partial(model_pkg.Nornn, rnn_num_layers=2)

        try:
            tmp = tempfile.mkdtemp()
            model_file = os.path.join(tmp, 'model')
            sequence = random_sequence(dir_name=tmp, num_frames=sequence_len, frmsz=frmsz)

            trajectories = []
            for i in range(len(ntimesteps)):
                o = opts.Opts()
                o.frmsz      = frmsz
                o.ntimesteps = ntimesteps[i]
                # o.initialize()
                tf.reset_default_graph()
                example = train._make_placeholders(o)
                example_white = train._whiten(example, o, stat={'mean': 0.0, 'std': 1.0})
                model = model_fn(example_white, o)
                saver = tf.train.Saver()
                init_op = tf.global_variables_initializer()

                with tf.Session() as sess:
                    sess.run(init_op)
                    if i == 0:
                        saver.save(sess, model_file)
                    else:
                        saver.restore(sess, model_file)
                    traj, _ = evaluate.track(sess, example, model, sequence, use_gt=False)
                    self.assertEqual(len(traj), sequence_len-1)
                    trajectories.append(traj)
                    if i > 0:
                        for t in range(sequence_len-1):
                            # print 'frame {}'.format(t)
                            # print traj[t]
                            # print trajectories[0][t]
                            np.testing.assert_allclose(traj[t], trajectories[0][t],
                                rtol=1e-4, atol=1e-6,
                                err_msg='different at frame {}'.format(t))
        finally:
            shutil.rmtree(tmp)


def random_sequence(dir_name, num_frames, frmsz):
    seq = {
        'image_files':    [],
        'labels':         [],
        'label_is_valid': [],
        'original_image_size': (frmsz, frmsz),
    }
    for t in range(num_frames):
        # TODO: Better to have "real-ish" images here?
        # Maybe it doesn't matter since the model is uninitialized.
        im = np.random.randint(256, size=(frmsz, frmsz, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        image_file = os.path.join(dir_name, '{:06d}.jpeg'.format(t))
        im.save(image_file)
        max_radius = 0.3
        size = np.random.uniform(max_radius, 2*max_radius, size=2)
        center = np.random.uniform(max_radius, 1-max_radius, size=2)
        label = np.concatenate([center-0.5*size, center+0.5*size])
        seq['image_files'].append(image_file)
        seq['labels'].append(label)
        seq['label_is_valid'].append(True)
    return seq
