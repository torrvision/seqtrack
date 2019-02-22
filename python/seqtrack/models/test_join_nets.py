from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets

import collections
import functools

from seqtrack import cnn
from seqtrack import receptive_field
from seqtrack.models import join_nets
from seqtrack.helpers import trySubTest


class TestFeatureNets(tf.test.TestCase):

    def test_instantiate(self):
        '''Instantiates the join functions.'''
        for join_arch in join_nets.SINGLE_JOIN_FNS:
            for num_batch_dims in [1, 2]:
                with trySubTest(self, join_arch=join_arch, num_batch_dims=num_batch_dims):
                    with tf.Graph().as_default():
                        join_fn = join_nets.BY_NAME[join_arch]
                        if join_arch in join_nets.FULLY_CONNECTED_FNS:
                            template_size = np.array([1, 1])
                        else:
                            template_size = np.array([4, 4])
                        search_size = np.array([10, 10])
                        template_shape = [None] + list(template_size) + [16]
                        search_shape = [None] * num_batch_dims + list(search_size) + [16]
                        template = tf.placeholder(tf.float32, template_shape, name='template')
                        search = tf.placeholder(tf.float32, search_shape, name='search')
                        is_training = tf.placeholder(tf.bool, (), name='is_training')
                        output = join_fn(template, search, is_training)
                        output_size = cnn.get_value(output).shape[-3:-1].as_list()
                        self.assertAllEqual(output_size, search_size - template_size + 1)
