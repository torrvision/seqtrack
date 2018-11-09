'''
Note that all functions are modified to use e.g. cnn.slim_conv2d() instead of slim.conv2d().
These functions use a cnn.Tensor to keep track of the receptive field.

You should not use slim.arg_scope() to modify the internal behaviour of these functions!
(Except through the provided e.g. alexnet.alexnet_v2_arg_scope().)
This is because a user does not know which functions are called internally.
Instead, these functions will expose all relevant options and use arg_scope internally.

We add parameter `variables_collections`.

Not yet: We add parameter `use_batch_norm`.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from seqtrack import cnn
from seqtrack import helpers

from . import util
from . import alexnet as alexnet_pkg
from . import vgg as vgg_pkg
from . import resnet_v1 as resnet_v1_pkg


# API of a feature function:
# 
# Takes an image and returns an output and a series of named intermediate endpoints (tensors).
# The intermediate endpoints may be used for multi-depth cross-correlation.

# TODO: Avoid duplication of default parameters here if possible?


def alexnet(x, is_training, trainable=True, variables_collections=None,
            weight_decay=0,
            output_layer='conv5',
            output_act='linear',
            freeze_until_layer=None,
            padding='VALID',
            enable_bnorm=True):
    with slim.arg_scope(feature_arg_scope(
            weight_decay=weight_decay, enable_bnorm=enable_bnorm, padding=padding)):
        return _alexnet_layers(x, is_training, trainable, variables_collections,
                               output_layer=output_layer,
                               output_activation_fn=helpers.get_act(output_act),
                               freeze_until_layer=freeze_until_layer)


def feature_arg_scope(weight_decay, enable_bnorm, padding):
    with slim.arg_scope(
            [cnn.slim_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay) if weight_decay else None,
            normalizer_fn=slim.batch_norm if enable_bnorm else None):
        with slim.arg_scope([cnn.slim_conv2d, cnn.slim_max_pool2d],
                            padding=padding) as arg_sc:
            return arg_sc


def _alexnet_layers(x, is_training, trainable=True, variables_collections=None,
                    output_layer='conv5',
                    output_activation_fn=None,
                    freeze_until_layer=None):
        # Should is_training be disabled with trainable=False?
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope([cnn.slim_conv2d, slim.batch_norm],
                              trainable=trainable,
                              variables_collections=variables_collections):
                # https://github.com/bertinetto/siamese-fc/blob/master/training/vid_create_net.m
                # https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
                # x = cnn.slim_conv2d(x, 96, [11, 11], 2, scope='conv1')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool1')
                # x = cnn.slim_conv2d(x, 256, [5, 5], scope='conv2')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool2')
                # x = cnn.slim_conv2d(x, 384, [3, 3], scope='conv3')
                # x = cnn.slim_conv2d(x, 384, [3, 3], scope='conv4')
                # x = cnn.slim_conv2d(x, 256, [3, 3], scope='conv5',
                #                     activation_fn=output_activation_fn, normalizer_fn=None)
                layers = [
                    ('conv1', util.partial(cnn.slim_conv2d, 96, [11, 11], 2)),
                    ('pool1', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv2', util.partial(cnn.slim_conv2d, 256, [5, 5])),
                    ('pool2', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv3', util.partial(cnn.slim_conv2d, 384, [3, 3])),
                    ('conv4', util.partial(cnn.slim_conv2d, 384, [3, 3])),
                    ('conv5', util.partial(cnn.slim_conv2d, 256, [3, 3])),
                ]
                return util.evaluate_until(
                    layers, x, output_layer,
                    output_kwargs=dict(
                        activation_fn=output_activation_fn,
                        normalizer_fn=None),
                    freeze_until_layer=freeze_until_layer)


def darknet(x, is_training, trainable=True, variables_collections=None,
            weight_decay=0,
            output_layer='conv5',
            output_act='linear',
            freeze_until_layer=None,
            padding='VALID',
            enable_bnorm=True):
    with slim.arg_scope(feature_arg_scope(
            weight_decay=weight_decay, enable_bnorm=enable_bnorm, padding=padding)):
        return _darknet_layers(x, is_training, trainable, variables_collections,
                               output_layer=output_layer,
                               output_activation_fn=helpers.get_act(output_act),
                               freeze_until_layer=freeze_until_layer)


def _darknet_layers(x, is_training, trainable=True, variables_collections=None,
                    output_layer='conv5',
                    output_activation_fn=None,
                    freeze_until_layer=None):
    # Should is_training be disabled with trainable=False?
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope([cnn.slim_conv2d, slim.batch_norm],
                          trainable=trainable,
                          variables_collections=variables_collections):
            # https://github.com/pjreddie/darknet/blob/master/cfg/darknet.cfg
            with slim.arg_scope([cnn.slim_conv2d], activation_fn=helpers.leaky_relu):
                # x = cnn.slim_conv2d(x, 16, [3, 3], 1, scope='conv1')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool1')
                # x = cnn.slim_conv2d(x, 32, [3, 3], 1, scope='conv2')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool2')
                # x = cnn.slim_conv2d(x, 64, [3, 3], 1, scope='conv3')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool3')
                # x = cnn.slim_conv2d(x, 128, [3, 3], 1, scope='conv4')
                # x = cnn.slim_max_pool2d(x, [3, 3], 2, scope='pool4')
                # x = cnn.slim_conv2d(x, 256, [3, 3], 1, scope='conv5',
                #                     activation_fn=output_activation_fn, normalizer_fn=None)
                layers = [
                    ('conv1', util.partial(cnn.slim_conv2d, 16, [3, 3], 1)),
                    ('pool1', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv2', util.partial(cnn.slim_conv2d, 32, [3, 3], 1)),
                    ('pool2', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv3', util.partial(cnn.slim_conv2d, 64, [3, 3], 1)),
                    ('pool3', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv4', util.partial(cnn.slim_conv2d, 128, [3, 3], 1)),
                    ('pool4', util.partial(cnn.slim_max_pool2d, [3, 3], 2)),
                    ('conv5', util.partial(cnn.slim_conv2d, 256, [3, 3], 1)),
                ]
                return util.evaluate_until(
                    layers, x, output_layer,
                    output_kwargs=dict(
                        activation_fn=output_activation_fn,
                        normalizer_fn=None),
                    freeze_until_layer=freeze_until_layer)


def slim_alexnet_v2(x, is_training, trainable=True, variables_collections=None,
                    weight_decay=0.0005,
                    conv_padding='VALID',
                    pool_padding='VALID',
                    conv1_stride=4,
                    output_layer='conv5',
                    output_act='linear',
                    freeze_until_layer=None):
    if not trainable:
        raise NotImplementedError('trainable not supported')
    # TODO: Support variables_collections.

    with slim.arg_scope(alexnet_pkg.alexnet_v2_arg_scope(
            weight_decay=weight_decay,
            conv_padding=conv_padding,
            pool_padding=pool_padding)):
        return alexnet_pkg.alexnet_v2(
            x,
            is_training=is_training,
            conv1_stride=conv1_stride,
            output_layer=output_layer,
            output_activation_fn=helpers.get_act(output_act),
            freeze_until_layer=freeze_until_layer)


def slim_vgg_a(x, is_training, trainable=True, variables_collections=None,
               weight_decay=0.0005,
               conv_padding='VALID',
               pool_padding='VALID',
               output_layer='conv5/conv5_2',
               output_act='linear',
               freeze_until_layer=None):
    if not trainable:
        raise NotImplementedError('trainable not supported')
    # TODO: Support variables_collections.

    with slim.arg_scope(vgg_pkg.vgg_arg_scope(
            weight_decay=weight_decay,
            conv_padding=conv_padding,
            pool_padding=pool_padding)):
        return vgg_pkg.vgg_a(
            x, is_training=is_training,
            output_layer=output_layer,
            output_activation_fn=helpers.get_act(output_act),
            freeze_until_layer=freeze_until_layer)


def slim_vgg_16(x, is_training, trainable=True, variables_collections=None,
                weight_decay=0.0005,
                conv_padding='VALID',
                pool_padding='VALID',
                output_layer='conv5/conv5_3',
                output_act='linear',
                freeze_until_layer=None):
    if not trainable:
        raise NotImplementedError('trainable not supported')
    # TODO: Support variables_collections.

    with slim.arg_scope(vgg_pkg.vgg_arg_scope(
            weight_decay=weight_decay,
            conv_padding=conv_padding,
            pool_padding=pool_padding)):
        return vgg_pkg.vgg_16(
            x, is_training=is_training,
            output_layer=output_layer,
            output_activation_fn=helpers.get_act(output_act),
            freeze_until_layer=freeze_until_layer)


def slim_resnet_v1_50(x, is_training, trainable=True, variables_collections=None,
                      weight_decay=0.0001,
                      use_batch_norm=True,
                      # reuse=None,
                      # scope='resnet_v1_50',
                      conv_padding='VALID',
                      pool_padding='VALID',
                      conv1_stride=2,
                      pool1_stride=2,
                      num_blocks=4,
                      block1_stride=2,
                      block2_stride=2,
                      block3_stride=2):
    if not trainable:
        raise NotImplementedError('trainable not supported')
    with slim.arg_scope(resnet_v1_pkg.resnet_arg_scope(
            weight_decay=weight_decay,
            use_batch_norm=use_batch_norm,
            pool_padding=pool_padding,
            variables_collections=variables_collections)):
        return resnet_v1_pkg.resnet_v1_50(
            x,
            is_training=is_training,
            # reuse=None,
            # scope='resnet_v1_50',
            conv_padding=conv_padding,
            conv1_stride=conv1_stride,
            pool1_stride=pool1_stride,
            num_blocks=num_blocks,
            block1_stride=block1_stride,
            block2_stride=block2_stride,
            block3_stride=block3_stride)


NAMES = [
    'alexnet',
    'darknet',
    'slim_alexnet_v2',
    'slim_resnet_v1_50',
    'slim_vgg_a',
    'slim_vgg_16',
]

BY_NAME = {name: globals()[name] for name in NAMES}


def get_receptive_field(feature_fn):
    graph = tf.Graph()
    with graph.as_default():
        image = tf.placeholder(tf.float32, (None, None, None, 3), name='image')
        is_training = tf.placeholder(tf.bool, (), name='is_training')
        image = cnn.as_tensor(image, add_to_set=True)
        feat, _ = feature_fn(image, is_training)
        return feat.fields[image.value]
