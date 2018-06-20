from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from seqtrack import cnnutil

from seqtrack.models.feature_nets import resnet_v1
from nets import resnet_v1 as original


INPUT_SHAPE = (8, 512, 512, 3)


class TestResnetV1(tf.test.TestCase):

    def test_original_eval(self):
        '''Tests that original function can be evaluated.'''
        inputs = tf.random_normal(shape=INPUT_SHAPE, dtype=tf.float32, name='inputs')
        with slim.arg_scope(original.resnet_arg_scope()):
            outputs, _ = original.resnet_v1_50(inputs, global_pool=False)

        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            outputs.eval()

    def test_eval_padding_same(self):
        '''Tests that the new function can be evaluated using SAME padding.'''
        self._test_eval_padding(padding='SAME')

    def test_eval_padding_valid(self):
        '''Tests that the new function can be evaluated using VALID padding.'''
        self._test_eval_padding(padding='VALID')

    def _test_eval_padding(self, padding):
        inputs = tf.random_normal(shape=INPUT_SHAPE, dtype=tf.float32, name='inputs')
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            outputs, _ = resnet_v1.resnet_v1_50(inputs, padding=padding)

        init_op = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init_op)
            outputs.eval()

    def test_reuse_original_params(self):
        '''Tests that new function can reuse params from original function.'''
        inputs = tf.random_normal(shape=INPUT_SHAPE, dtype=tf.float32, name='inputs')

        with slim.arg_scope(original.resnet_arg_scope()):
            original.resnet_v1_50(inputs, global_pool=False, scope='resnet', reuse=False)
        # This should raise an exception if there are any new parameters.
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            resnet_v1.resnet_v1_50(inputs, scope='resnet', reuse=True)

    def test_valid_smaller_than_same(self):
        '''Tests that the output using VALID padding is smaller than using SAME.'''
        inputs = tf.random_normal(shape=INPUT_SHAPE, dtype=tf.float32, name='inputs')
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            outputs_valid, _ = resnet_v1.resnet_v1_50(inputs, padding='VALID',
                                                      scope='resnet_valid')
            outputs_same, _ = resnet_v1.resnet_v1_50(inputs, padding='SAME',
                                                     scope='resnet_same')

        self.assertLess(outputs_valid.shape[1], outputs_same.shape[1])
        self.assertLess(outputs_valid.shape[2], outputs_same.shape[2])

    def test_receptive_field_padding_same(self):
        '''Tests that the receptive field is correct using SAME padding.'''
        self._test_receptive_field_padding(padding='SAME')

    def test_receptive_field_padding_valid(self):
        '''Tests that the receptive field is correct using VALID padding.'''
        self._test_receptive_field_padding(padding='VALID')

    def _test_receptive_field_padding(self, padding):
        image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.float32, name='image')
        image = cnnutil.Tensor(image, rfs={'image': cnnutil.identity_rf()})
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            output, _ = resnet_v1.resnet_v1_50(image, padding=padding)

        if padding == 'VALID':
            self.assertTrue(all(output.rfs['image'].rect.min == 0))
        else:
            self.assertTrue(all(output.rfs['image'].rect.int_center() == 0))


if __name__ == '__main__':
    tf.test.main()
