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
            with trySubTest(self, join_arch=join_arch):
                with tf.Graph().as_default():
                    join_fn = join_nets.BY_NAME[join_arch]
                    if join_arch in join_nets.FULLY_CONNECTED_FNS:
                        template_size = np.array([1, 1])
                    else:
                        template_size = np.array([4, 4])
                    search_size = np.array([10, 10])
                    template_shape = [3] + list(template_size) + [16]
                    search_shape = [3, 2] + list(search_size) + [16]

                    template = tf.placeholder(tf.float32, template_shape, name='template')
                    search = tf.placeholder(tf.float32, search_shape, name='search')
                    is_training = tf.placeholder(tf.bool, (), name='is_training')
                    output = join_fn(template, search, is_training)
                    output = cnn.get_value(output)
                    output_size = output.shape[-3:-1].as_list()
                    self.assertAllEqual(output_size, search_size - template_size + 1)

                    init_op = tf.global_variables_initializer()
                    # with self.test_session() as sess:
                    with tf.Session() as sess:
                        sess.run(init_op)
                        sess.run(output, feed_dict={
                            template: np.random.normal(size=template_shape),
                            search: np.random.normal(size=search_shape),
                            is_training: False,
                        })
