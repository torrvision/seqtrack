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
from seqtrack.models import feature_nets


BATCH_LEN = 2


class TestFeatureNets(tf.test.TestCase):

    def test_unknown_size(self):
        '''Instantiates the network with unknown spatial dimensions.'''
        for feature_arch in feature_nets.NAMES:
            with self.subTest(feature_arch=feature_arch):
                feature_fn = feature_nets.BY_NAME[feature_arch]
                with tf.Graph().as_default():
                    image = tf.placeholder(tf.float32, (None, None, None, 32), name='image')
                    is_training = tf.placeholder(tf.bool, (), name='is_training')
                    _, _ = feature_fn(image, is_training)

    def test_desired_output_size_from_receptive_field(self):
        '''Uses the receptive field to get the input size for desired output size.'''
        for feature_arch in feature_nets.NAMES:
            with self.subTest(feature_arch=feature_arch), tf.Graph().as_default():
                feature_fn = feature_nets.BY_NAME[feature_arch]
                field = feature_nets.get_receptive_field(feature_fn)

                desired = np.array([10, 10])
                input_size = receptive_field.input_size(field, desired)
                input_shape = [None] + list(input_size) + [3]
                image = tf.placeholder(tf.float32, input_shape, name='image')
                is_training = tf.placeholder(tf.bool, (), name='is_training')
                feat, _ = feature_fn(image, is_training)
                output_size = feat.value.shape[1:3].as_list()
                self.assertAllEqual(output_size, desired)
                receptive_field.assert_center_alignment(input_size, output_size, field)

    def test_same_variables(self):
        '''Instantiates feature net using same scope as original function.'''
        for feature_arch in SLIM_ARCH_NAMES:
            with self.subTest(feature_arch=feature_arch), tf.Graph().as_default():
                original_fn = globals()[feature_arch]
                feature_fn = feature_nets.BY_NAME[feature_arch]

                image = tf.placeholder(tf.float32, (None, None, None, 3), name='image')
                with tf.variable_scope('net', reuse=False):
                    original_fn(image, is_training=True)
                with tf.variable_scope('net', reuse=True):
                    feature_fn(image, is_training=True)

    def test_no_padding_by_default(self):
        '''Tests that feature functions with default options have zero padding.'''
        for feature_arch in feature_nets.NAMES:
            with self.subTest(feature_arch=feature_arch), tf.Graph().as_default():
                feature_fn = feature_nets.BY_NAME[feature_arch]
                image = tf.placeholder(tf.float32, (None, None, None, 3), name='image')
                image = cnn.as_tensor(image, add_to_set=True)
                feat, _ = feature_fn(image, is_training=True)
                field = feat.fields[image.value]
                self.assertAllEqual(field.padding, [0, 0])

    def test_output_equal(self):
        '''Compares output to library implementation of networks.'''
        # The desired_size may need to be chosen such that original network structure is valid.
        TestCase = collections.namedtuple('TestCase', ['kwargs', 'desired_size', 'end_point'])
        cases = {
            'slim_alexnet_v2': TestCase(
                kwargs=dict(
                    output_layer='conv5',
                    output_act='relu',
                    conv_padding='SAME',
                    pool_padding='VALID'),
                desired_size=np.array([13, 13]),  # 3 + (6 - 1) * 2
                end_point='alexnet_v2/conv5',
            ),
            'slim_resnet_v1_50': TestCase(
                kwargs=dict(
                    num_blocks=4,
                    conv_padding='SAME',
                    pool_padding='SAME'),
                desired_size=np.array([3, 3]),
                end_point='resnet_v1_50/block4',
            ),
            'slim_vgg_16': TestCase(
                kwargs=dict(
                    output_layer='fc6',
                    output_act='relu',
                    conv_padding='SAME',
                    pool_padding='VALID'),
                desired_size=np.array([1, 1]),
                end_point='vgg_16/fc6',
            ),
        }

        for feature_arch, test_case in cases.items():
            graph = tf.Graph()
            with self.subTest(feature_arch=feature_arch), graph.as_default():
                original_fn = globals()[feature_arch]
                feature_fn = functools.partial(feature_nets.BY_NAME[feature_arch],
                                               **test_case.kwargs)
                field = feature_nets.get_receptive_field(feature_fn)
                input_size = receptive_field.input_size(field, test_case.desired_size)
                input_shape = [None] + list(input_size) + [3]

                image = tf.placeholder(tf.float32, input_shape, name='image')
                with tf.variable_scope('net', reuse=False):
                    _, end_points = original_fn(image, is_training=True)
                    try:
                        original = end_points['net/' + test_case.end_point]
                    except KeyError as ex:
                        raise ValueError('key not found ({}) in list: {}'.format(
                            ex, sorted(end_points.keys())))
                init_op = tf.global_variables_initializer()
                with tf.variable_scope('net', reuse=True):
                    ours, _ = feature_fn(image, is_training=True)
                    ours = cnn.get_value(ours)
                # self.assertEqual(original.shape.as_list(), ours.shape.as_list())

                with self.session(graph=graph) as sess:
                    sess.run(init_op)
                    want, got = sess.run((original, ours), feed_dict={
                        image: np.random.uniform(size=[BATCH_LEN] + input_shape[1:]),
                    })
                    self.assertAllClose(want, got)


# Functions that call original implementations from slim.nets:

SLIM_ARCH_NAMES = [
    'slim_alexnet_v2',
    'slim_resnet_v1_50',
    'slim_vgg_a',
    'slim_vgg_16',
]

def slim_alexnet_v2(inputs, is_training):
    with slim.arg_scope(slim_nets.alexnet.alexnet_v2_arg_scope()):
        return slim_nets.alexnet.alexnet_v2(inputs, is_training=is_training,
                                            spatial_squeeze=False)

def slim_resnet_v1_50(inputs, is_training):
    with slim.arg_scope(slim_nets.resnet_v1.resnet_arg_scope()):
        return slim_nets.resnet_v1.resnet_v1_50(inputs, is_training=is_training)

def slim_vgg_a(inputs, is_training):
    with slim.arg_scope(slim_nets.vgg.vgg_arg_scope()):
        return slim_nets.vgg.vgg_a(inputs, is_training=is_training,
                                    spatial_squeeze=False)

def slim_vgg_16(inputs, is_training):
    with slim.arg_scope(slim_nets.vgg.vgg_arg_scope()):
        return slim_nets.vgg.vgg_16(inputs, is_training=is_training,
                                    spatial_squeeze=False)
