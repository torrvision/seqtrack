from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import math
import pprint

from seqtrack import cnn


def conv2d(template, search,
           learn_spatial_weight=False,
           reduce_channels=True,
           use_mean=False,
           use_batch_norm=False,
           learn_gain=False,
           gain_init=1,
           scope='conv2d'):
    '''
    If use_batch_norm is true, then an output gain will always be incorporated.
    Otherwise, it will only be incorporated if learn_gain is true.

    It is not necessary to use the mean if using batch norm.
    '''
    with tf.variable_scope(scope, 'conv2d'):
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)
        if learn_spatial_weight:
            # TODO: Could have different spatial weight for each channel?
            template_size = template.shape[1:3].as_list()
            spatial_weight = tf.get_variable('spatial_weight', template_size, tf.float32,
                                             initializer=tf.ones_initializer())
            template *= tf.expand_dims(spatial_weight, -1)

        dot = cnn.diag_xcorr(search, template)
        dot = cnn.channel_sum(dot)
        if use_mean:
            num_elems = np.prod(template.shape[1:].as_list())
            dot = cnn.pixelwise(lambda dot: (1 / tf.to_float(num_elems)) * dot, dot)
        # Take mean over channels.
        return _calibrate(dot, use_batch_norm, learn_gain, gain_init)


def cosine(template, search,
           use_batch_norm=False,
           gain_init=True,
           eps=1e-3,
           scope='cosine'):
    with tf.variable_scope(scope, 'cosine'):
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)

        num_channels = template.shape[-1].value
        template_size = template.shape[1:3].as_list()
        ones = tf.ones(template_size + [num_channels, 1], tf.float32)

        dot_xy = cnn.channel_sum(cnn.diag_xcorr(search, template, padding='VALID'))
        dot_xx = tf.reduce_sum(tf.square(template), axis=(-3, -2, -1), keepdims=True)
        dot_yy = cnn.nn_conv2d(cnn.pixelwise(tf.square, search), ones,
                               strides=[1, 1, 1, 1], padding='VALID')

        denom = cnn.pixelwise(lambda dot_yy: tf.sqrt(dot_xx * dot_yy), dot_yy)
        similarity = cnn.pixelwise_binary(
            lambda dot_xy, denom: dot_xy / (denom + eps), dot_xy, denom)
        # Gain is necessary here because similarity is always in [-1, 1].
        return _calibrate(similarity, use_batch_norm, learn_gain=True, gain_init=gain_init)


def distance(template, search,
             use_mean=False,
             use_batch_norm=False,
             learn_gain=False,
             gain_init=1,
             scope='distance'):
    with tf.variable_scope(scope, 'distance'):
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)

        num_channels = template.shape[-1].value
        template_size = template.shape[1:3].as_list()
        ones = tf.ones(template_size + [num_channels, 1], tf.float32)

        dot_xy = cnn.diag_xcorr(search, template)
        dot_xx = tf.reduce_sum(tf.square(template), axis=(-3, -2, -1), keepdims=True)
        dot_yy = cnn.nn_conv2d(cnn.pixelwise(tf.square, search), ones,
                               strides=[1, 1, 1, 1], padding='VALID')
        # (x - y)**2 = x**2 - 2 x y + y**2
        # sq_dist = dot_xx - 2 * dot_xy + dot_yy
        sq_dist = cnn.pixelwise_binary(
            lambda dot_xy, dot_yy: dot_xx - 2 * dot_xy + dot_yy, dot_xy, dot_yy)
        sq_dist = cnn.pixelwise(
            lambda sq_dist: tf.reduce_sum(sq_dist, axis=-1, keepdims=True), sq_dist)
        if use_mean:
            # Take root-mean-square of difference.
            num_elems = np.prod(template.shape[1:].as_list())
            sq_dist = cnn.pixelwise(lambda sq_dist: (1 / tf.to_float(num_elems)) * sq_dist, sq_dist)
        dist = cnn.pixelwise(tf.sqrt, sq_dist)
        return _calibrate(dist, use_batch_norm, learn_gain, gain_init)


def depthwise_conv2d(template, search,
                     learn_spatial_weight=False,
                     use_mean=False,
                     use_batch_norm=False,
                     scope='depthwise_conv2d'):
    '''Computes the cross-correlation of each channel independently.'''
    with tf.variable_scope(scope, 'depthwise_conv2d'):
        template = cnn.get_value(template)
        template_size = template.shape[1:3].as_list()

        if learn_spatial_weight:
            # TODO: Could have different spatial weight for each channel?
            spatial_weight = tf.get_variable('spatial_weight', template_size, tf.float32,
                                             initializer=tf.ones_initializer())
            template *= tf.expand_dims(spatial_weight, -1)

        output = cnn.diag_xcorr(search, template)
        if use_mean:
            num_elems = np.prod(template_size)
            output = cnn.pixelwise(lambda output: (1 / tf.to_float(num_elems)) * output, output)
        if use_batch_norm:
            output = slim.batch_norm(output.value)
        return output


def all_pixel_pairs(template, search, scope='all_pixel_pairs'):
    '''
    Args:
        template: cnn.Tensor with shape [n, h_t, w_t, c]
        search: cnn.Tensor with shape [n, h_s, w_s, c]

    Returns:
        cnn.Tensor with shape [n, h_s, w_s, h_t * w_t]
    '''
    with tf.variable_scope(scope, 'all_pixel_pairs'):
        # Do not worry about receptive field for template.
        template = cnn.get_value(template)

        template_size = template.shape[1:3].as_list()
        num_channels = template.shape[-1].value

        # Break template into 1x1 patches.
        # Then "convolve" (multiply) each with the search image.
        t = template.value
        s = search.value
        # template becomes: [n,   1,   1, h_t, w_t, c]
        # search becomes:   [n, h_s, w_s,   1,   1, c]
        t = helpers.expand_dims_n(t, 1, 2)
        s = helpers.expand_dims_n(s, 3, 2)
        p = t * s
        p = tf.reduce_sum(p, axis=-1, keepdims=False)
        # Merge the spatial dimensions of the template into features.
        # response becomes: [n, h_s, w_s, h_t * w_t]
        p, _ = helpers.merge_dims(3, 5)
        pairs = cnn.Tensor(p, search.field)

        # weights has shape [h_t, w_t, h_t, w_t]
        weights = tf.get_variable('weights', template_size + template_size, tf.float32,
                                  initializer=tf.ones_initializer())
        weights, _ = tf.merge_dims(weights, 2, 4)
        weights = tf.expand_dims(weights, 0)
        response = cnn.nn_conv2d(pairs, weights,
                                 strides=[1, 1, 1, 1], padding='VALID')
        return response


def abs_diff(template, search,
             reduce_channels=True,
             use_mean=False,
             use_batch_norm=False,
             scope='abs_diff'):
    '''
    Requires that template is 1x1.
    '''
    with tf.variable_scope(scope, 'abs_diff'):
        template = cnn.get_value(template)
        template_size = template.shape[1:3].as_list()
        if template_size != [1, 1]:
            raise ValueError('template shape is not [1, 1]: {}'.format(template_size))
        # Use broadcasting to perform element-wise operation.
        delta = cnn.pixelwise(lambda x: tf.abs(x - template), search)
        if reduce_channels:
            delta = cnn.channel_sum(delta)
            if use_mean:
                num_channels = template.shape[-1].value
                delta = cnn.pixelwise(lambda x: (1 / tf.to_float(num_channels)) * x, delta)
        # TODO: No bias if attaching more layers?
        return _calibrate(delta, use_batch_norm, learn_gain=False, gain_init=1)


def _calibrate(response, use_batch_norm, learn_gain, gain_init):
    '''
    Either adds batch_norm (with center and scale) or a scalar bias with optional gain.
    '''
    if use_batch_norm:
        output = cnn.pixelwise(slim.batch_norm, response, center=True, scale=True)
    else:
        # Add bias (cannot be represented by dot product) and optional gain.
        bias = tf.get_variable('bias', [], tf.float32, initializer=tf.zeros_initializer())
        if learn_gain:
            gain = tf.get_variable('gain', [], tf.float32,
                                   initializer=tf.constant_initializer(gain_init))
            output = cnn.pixelwise(lambda x: gain * x + bias, response)
        else:
            output = cnn.pixelwise(lambda x: x + bias, response)
    return output


def mlp(template, search,
        num_layers,
        num_hidden,
        join_name='depthwise_conv2d',
        join_params=None,
        scope='mlp_join'):
    with tf.variable_scope(scope, 'mlp_join'):
        join_params = join_params or {}
        join_fn = BY_NAME[join_name]
        similarity = join_fn(template, search, **join_params)
        response = cnn.mlp(similarity, num_layers=num_layers,
                           num_hidden=num_hidden, num_outputs=1)
        return response


NAMES = [
    'conv2d',
    'distance',
    'cosine',
    'abs_diff',
    'depthwise_conv2d',
]

BY_NAME = {name: globals()[name] for name in NAMES}
