from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from seqtrack import cnn
from seqtrack import helpers


def xcorr(template, search, is_training,
          enable_pre_conv=False,
          pre_conv_params=None,
          learn_spatial_weight=False,
          reduce_channels=True,
          use_mean=False,
          use_batch_norm=False,
          learn_gain=False,
          gain_init=1,
          scope='xcorr'):
    '''
    If use_batch_norm is true, then an output gain will always be incorporated.
    Otherwise, it will only be incorporated if learn_gain is true.

    It is not necessary to use the mean if using batch norm.
    '''
    return _xcorr_general(
        template, search, is_training,
        enable_pre_conv=enable_pre_conv,
        pre_conv_params=pre_conv_params,
        learn_spatial_weight=learn_spatial_weight,
        reduce_channels=reduce_channels,
        use_mean=use_mean,
        use_batch_norm=use_batch_norm,
        learn_gain=learn_gain,
        gain_init=gain_init,
        scope=scope)


def _xcorr_general(template, search, is_training,
                   enable_pre_conv=False,
                   pre_conv_params=None,
                   learn_spatial_weight=False,
                   reduce_channels=True,
                   use_mean=False,
                   use_batch_norm=False,
                   learn_gain=False,
                   gain_init=1,
                   scope='xcorr'):
    '''Convolves template with search.

    If use_batch_norm is true, then an output gain will always be incorporated.
    Otherwise, it will only be incorporated if learn_gain is true.

    It is not necessary to use the mean if using batch norm.
    '''
    with tf.variable_scope(scope, 'xcorr'):
        pre_conv_params = pre_conv_params or {}
        template_size = template.shape[1:3].as_list()

        if enable_pre_conv:
            template = _pre_conv(template, is_training, scope='pre', reuse=False, **pre_conv_params)
            search = _pre_conv(search, is_training, scope='pre', reuse=True, **pre_conv_params)
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)

        spatial_normalizer = (1 / np.prod(template_size)) if use_mean else 1
        if learn_spatial_weight:
            # Initialize with spatial normalizer.
            spatial_weight = tf.get_variable(
                'spatial_weight', template_size, tf.float32,
                initializer=tf.constant_initializer(spatial_normalizer))
            template *= tf.expand_dims(spatial_weight, -1)
        dot = cnn.diag_xcorr(search, template)
        if not learn_spatial_weight:
            dot = cnn.pixelwise(lambda dot: spatial_normalizer * dot, dot)

        if reduce_channels:
            dot = cnn.channel_mean(dot) if use_mean else cnn.channel_sum(dot)
        return _calibrate(dot, use_batch_norm, learn_gain, gain_init)


def _pre_conv(x, is_training,
              num_outputs=None,
              kernel_size=1,
              stride=1,
              padding='VALID',
              activation='linear',
              scope='preconv',
              reuse=None):
    '''
    Args:
        num_outputs: If num_outputs is None, the input dimension is used.
    '''
    x = cnn.as_tensor(x)
    if not num_outputs:
        num_outputs = x.value.shape[-1].value
    return cnn.slim_conv2d(x, num_outputs, kernel_size,
                           stride=stride,
                           padding=padding,
                           activation_fn=helpers.get_act(activation),
                           normalizer_fn=None,  # No batch-norm.
                           scope=scope,
                           reuse=reuse)


def cosine(template, search, is_training,
           use_batch_norm=False,
           gain_init=1,
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


def distance(template, search, is_training,
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


def depthwise_xcorr(template, search, is_training,
                    learn_spatial_weight=False,
                    use_mean=False,
                    use_batch_norm=False,
                    learn_gain=False,
                    gain_init=1,
                    scope='depthwise_xcorr'):
    '''Computes the cross-correlation of each channel independently.'''
    return _xcorr_general(
        template, search, is_training,
        enable_pre_conv=False,
        learn_spatial_weight=learn_spatial_weight,
        reduce_channels=False,
        use_mean=use_mean,
        use_batch_norm=use_batch_norm,
        learn_gain=learn_gain,
        gain_init=gain_init,
        scope=scope)


def all_pixel_pairs(template, search, is_training,
                    operation='mul',
                    reduce_channels=True,
                    use_mean=True,
                    scope='all_pixel_pairs'):
    '''
    Args:
        template: cnn.Tensor with shape [n, h_t, w_t, c]
        search: cnn.Tensor with shape [n, h_s, w_s, c]

    Returns:
        cnn.Tensor with shape [n, h_s, w_s, h_t * w_t]
    '''
    with tf.variable_scope(scope, 'all_pixel_pairs'):
        template = cnn.as_tensor(template)
        search = cnn.as_tensor(search)
        template_size = template.value.shape[1:3].as_list()
        num_channels = template.value.shape[-1].value

        # Break template into 1x1 patches.
        # Then "convolve" (multiply) each with the search image.
        t = template.value
        s = search.value
        # template becomes: [n,   1,   1, h_t, w_t, c]
        # search becomes:   [n, h_s, w_s,   1,   1, c]
        t = helpers.expand_dims_n(t, 1, 2)
        s = helpers.expand_dims_n(s, 3, 2)
        if operation == 'mul':
            p = t * s
        elif operation == 'abs_diff':
            p = tf.abs(t - s)
        else:
            raise ValueError('unknown operation: "{}"'.format(operation))

        # if reduce_channels:
        #     if use_mean:
        #         p = tf.reduce_mean(p, axis=-1, keepdims=True)
        #     else:
        #         p = tf.reduce_sum(p, axis=-1, keepdims=True)
        # Merge the spatial dimensions of the template into features.
        # response becomes: [n, h_s, w_s, h_t * w_t * c]
        p, _ = helpers.merge_dims(p, 3, 6)
        pairs = cnn.Tensor(p, search.fields)

        # TODO: This initialization could be too small?
        normalizer = 1 / (np.prod(template_size) ** 2 * num_channels) if use_mean else 1
        weights_shape = template_size + [np.prod(template_size) * num_channels, 1]
        weights = tf.get_variable('weights', weights_shape, tf.float32,
                                  initializer=tf.constant_initializer(normalizer))
        # TODO: Support depthwise_conv2d (keep channels).
        response = cnn.nn_conv2d(pairs, weights, strides=[1, 1, 1, 1], padding='VALID')
        return response


def abs_diff(template, search, is_training,
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


def mlp(template, search, is_training,
        num_layers,
        num_hidden,
        join_name='depthwise_xcorr',
        join_params=None,
        scope='mlp_join'):
    with tf.variable_scope(scope, 'mlp_join'):
        join_params = join_params or {}
        join_fn = BY_NAME[join_name]
        similarity = join_fn(template, search, **join_params)
        response = cnn.mlp(similarity, num_layers=num_layers,
                           num_hidden=num_hidden, num_outputs=1)
        return response


def multi_xcorr(template, search, is_training,
                layer_names, template_layers, search_layers, search_image,
                enable_final_conv=False,
                final_conv_params=None,
                hidden_conv_num_outputs=None,
                hidden_conv_activation='linear',
                use_batch_norm=False,
                use_mean=False,
                scope='multi_xcorr'):
    '''
    Args:
        template_layers: Dict that maps names to tensors.
        search_layers: Dict that maps names to tensors.
    '''
    with tf.variable_scope(scope, 'multi_xcorr'):
        template_layers = template_layers or {}
        search_layers = search_layers or {}
        final_conv_params = final_conv_params or {}
        assert 'final' not in layer_names

        scores = {}
        scores['final'] = _xcorr_general(template, search, is_training,
                                         enable_pre_conv=enable_final_conv,
                                         pre_conv_params=final_conv_params,
                                         use_mean=use_mean,
                                         use_batch_norm=use_batch_norm,
                                         scope='final_xcorr')
        final_conv = cnn.channel_sum(cnn.diag_xcorr(search, template))
        for name in layer_names:
            template_layer = template_layers[name]
            search_layer = search_layers[name]
            # TODO: Add batch-norm to each cross-correlation?
            # Must be a 1x1 convolution to ensure that receptive fields of different layers align.
            scores[name] = _xcorr_general(template_layers[name], search_layers[name], is_training,
                                          enable_pre_conv=True,
                                          pre_conv_params=dict(
                                              num_outputs=hidden_conv_num_outputs,
                                              kernel_size=1,
                                              stride=1,
                                              activation=hidden_conv_activation),
                                          use_mean=use_mean,
                                          use_batch_norm=use_batch_norm,
                                          scope=name + '_xcorr')

        # Upsample all to minimum stride.
        # Then take center-crop of minimum size.
        field_strides = {name: _unique(score.fields[cnn.get_value(search_image)].stride)
                         for name, score in scores.items()}
        min_stride = min(field_strides.values())
        for name in ['final'] + layer_names:
            stride = field_strides[name]
            if stride != min_stride:
                assert stride % min_stride == 0
                relative = stride // min_stride
                scores[name] = cnn.upsample(scores[name], relative,
                                            method=tf.image.ResizeMethod.BILINEAR)

        sizes = {name: _unique(score.value.shape[1:3].as_list()) for name, score in scores.items()}
        min_size = min(sizes.values())
        for name in ['final'] + layer_names:
            size = sizes[name]
            if (size - min_size) % 2 != 0:
                raise ValueError('remainder is not even: {} within {}'.format(min_size, size))
            margin = (size - min_size) // 2
            scores[name] = cnn.spatial_trim(scores[name], margin, margin)

        # TODO: How to handle calibration here?
        total = scores['final']
        for name in layer_names:
            total += scores[name]
        return total


def _unique(elems):
    assert len(elems) > 0
    val = elems[0]
    assert all(x == val for x in elems)
    return val


'''Functions that take two inputs.
'''
SINGLE_JOIN_FNS = [
    'xcorr',
    'distance',
    'cosine',
    'abs_diff',
    'depthwise_xcorr',
    'all_pixel_pairs',
    'abs_diff',
]

'''Functions that require 1x1 template.
'''
FULLY_CONNECTED_FNS = [
    'abs_diff',
]

'''Functions that use features from multiple layers.
'''
MULTI_JOIN_FNS = [
    'multi_xcorr',
]

BY_NAME = {name: globals()[name] for name in SINGLE_JOIN_FNS + MULTI_JOIN_FNS}
