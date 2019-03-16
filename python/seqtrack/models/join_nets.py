from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import functools
partial = functools.partial

from seqtrack import cnn
from seqtrack import helpers


def concat_fc(template, search, is_training,
              trainable=True,
              join_dim=128,
              mlp_num_outputs=1,
              mlp_num_layers=2,
              mlp_num_hidden=128,
              mlp_kwargs=None,
              scope=None):
    '''
    Args:
        template: [b, h, w, c]
        search: [b, s, h, w, c]
    '''
    with tf.variable_scope(scope, 'concat_fc'):
        template = cnn.as_tensor(template)
        search = cnn.as_tensor(search)

        # Instead of sliding-window concat, we do separate conv and sum the results.
        # Disable activation and normalizer. Perform these after the sum.
        kernel_size = template.value.shape[-3:-1].as_list()
        conv_kwargs = dict(
            padding='VALID',
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=None,  # Disable bias because bnorm is performed later.
        )
        with tf.variable_scope('template'):
            template = cnn.slim_conv2d(template, join_dim, kernel_size,
                                       scope='fc', **conv_kwargs)
        with tf.variable_scope('search'):
            search, restore = cnn.merge_batch_dims(search)
            search = cnn.slim_conv2d(search, join_dim, kernel_size,
                                     scope='fc', **conv_kwargs)
            search = restore(search)

        template = cnn.get_value(template)
        template = tf.expand_dims(template, 1)
        # This is a broadcasting addition. Receptive field in template not tracked.
        output = cnn.pixelwise(lambda search: search + template, search)
        output = cnn.pixelwise(partial(slim.batch_norm, is_training=is_training), output)
        output = cnn.pixelwise(tf.nn.relu, output)

        mlp_kwargs = mlp_kwargs or {}
        output, restore = cnn.merge_batch_dims(output)
        output = cnn.mlp(output,
                         num_layers=mlp_num_layers,
                         num_hidden=mlp_num_hidden,
                         num_outputs=mlp_num_outputs,
                         trainable=trainable, **mlp_kwargs)
        output = restore(output)
        return output


def xcorr(template, search, is_training,
          trainable=True,
          use_pre_conv=False,
          pre_conv_params=None,
          learn_spatial_weight=False,
          weight_init_method='ones',
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
        trainable=trainable,
        use_pre_conv=use_pre_conv,
        pre_conv_params=pre_conv_params,
        learn_spatial_weight=learn_spatial_weight,
        weight_init_method=weight_init_method,
        reduce_channels=reduce_channels,
        use_mean=use_mean,
        use_batch_norm=use_batch_norm,
        learn_gain=learn_gain,
        gain_init=gain_init,
        scope=scope)


def _xcorr_general(template, search, is_training,
                   trainable=True,
                   use_pre_conv=False,
                   pre_conv_params=None,
                   learn_spatial_weight=False,
                   weight_init_method='ones',
                   reduce_channels=True,
                   use_mean=False,
                   use_batch_norm=False,
                   learn_gain=False,
                   gain_init=1,
                   scope='xcorr'):
    '''Convolves template with search.

    Args:
        template: [b, h, w, c]
        search: [b, s, h, w, c]

    If use_batch_norm is true, then an output gain will always be incorporated.
    Otherwise, it will only be incorporated if learn_gain is true.

    When `learn_spatial_weight` is false:
    If `use_batch_norm` is true, `use_mean` should have no effect.
    When `learn_spatial_weight` is true:
    The `use_mean` parameter also controls the initialization of the spatial weights.
    This may have an effect on gradient descent, even if `use_batch_norm` is true.
    '''
    with tf.variable_scope(scope, 'xcorr'):
        pre_conv_params = pre_conv_params or {}

        if use_pre_conv:
            template = _pre_conv(template, is_training, trainable=trainable,
                                 scope='pre', reuse=False, **pre_conv_params)
            search = _pre_conv(search, is_training, trainable=trainable,
                               scope='pre', reuse=True, **pre_conv_params)
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)
        template_size = template.shape[-3:-1].as_list()

        # There are two separate issues here:
        # 1. Whether to make the initial output equal to the mean?
        # 2. How to share this between a constant multiplier and initialization?
        spatial_normalizer = 1 / np.prod(template_size)
        if learn_spatial_weight:
            if weight_init_method == 'mean':
                weight_init = spatial_normalizer
            elif weight_init_method == 'ones':
                weight_init = 1
            else:
                raise ValueError('unknown weight init method: "{}"'.format(weight_init_method))
        else:
            weight_init = 1
        if use_mean:
            # Maintain property:
            # normalize_factor * weight_init = spatial_normalizer
            normalize_factor = spatial_normalizer / weight_init
        else:
            normalize_factor = 1

        if learn_spatial_weight:
            # Initialize with spatial normalizer.
            spatial_weight = tf.get_variable(
                'spatial_weight', template_size, tf.float32,
                initializer=tf.constant_initializer(weight_init),
                trainable=trainable)
            template *= tf.expand_dims(spatial_weight, -1)
        dot = cnn.diag_xcorr(search, template)
        dot = cnn.pixelwise(lambda dot: normalize_factor * dot, dot)
        if reduce_channels:
            dot = cnn.channel_mean(dot) if use_mean else cnn.channel_sum(dot)
        return _calibrate(dot, is_training, use_batch_norm, learn_gain, gain_init,
                          trainable=trainable)


def _pre_conv(x, is_training,
              num_outputs=None,
              kernel_size=1,
              stride=1,
              padding='VALID',
              activation='linear',
              trainable=True,
              scope='preconv',
              reuse=None):
    '''
    Args:
        num_outputs: If num_outputs is None, the input dimension is used.
    '''
    # TODO: Support multi-scale.
    x = cnn.as_tensor(x)
    if not num_outputs:
        num_outputs = x.value.shape[-1].value
    return cnn.slim_conv2d(x, num_outputs, kernel_size,
                           stride=stride,
                           padding=padding,
                           activation_fn=helpers.get_act(activation),
                           normalizer_fn=None,  # No batch-norm.
                           trainable=trainable,
                           scope=scope,
                           reuse=reuse)


def cosine(template, search, is_training,
           trainable=True,
           use_batch_norm=False,
           gain_init=1,
           eps=1e-3,
           scope='cosine'):
    '''
    Args:
        template: [b, h, w, c]
        search: [b, s, h, w, c]
    '''
    search = cnn.as_tensor(search)
    num_search_dims = len(search.value.shape)
    if num_search_dims != 5:
        raise ValueError('search should have 5 dims: {}'.format(num_search_dims))

    with tf.variable_scope(scope, 'cosine'):
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)

        dot_xy = cnn.channel_sum(cnn.diag_xcorr(search, template, padding='VALID'))
        dot_xx = tf.reduce_sum(tf.square(template), axis=(-3, -2, -1), keepdims=True)

        sq_search = cnn.pixelwise(tf.square, search)
        ones = tf.ones_like(template)  # TODO: Faster and less memory to use sum.
        dot_yy = cnn.channel_sum(cnn.diag_xcorr(sq_search, ones, padding='VALID'))
        # num_channels = template.shape[-1].value
        # template_size = template.shape[-3:-1].as_list()
        # ones = tf.ones(template_size + [num_channels, 1], tf.float32)
        # sq_search, restore = cnn.merge_batch_dims(sq_search)
        # dot_yy = cnn.nn_conv2d(sq_search, ones, strides=[1, 1, 1, 1], padding='VALID')
        # dot_yy = restore(dot_yy)

        dot_xx = tf.expand_dims(dot_xx, 1)
        assert_ops = [tf.assert_non_negative(dot_xx, message='assert dot_xx non negative'),
                      tf.assert_non_negative(dot_yy.value, message='assert dot_yy non negative')]
        with tf.control_dependencies(assert_ops):
            denom = cnn.pixelwise(lambda dot_yy: tf.sqrt(dot_xx * dot_yy), dot_yy)
        similarity = cnn.pixelwise_binary(
            lambda dot_xy, denom: dot_xy / (denom + eps), dot_xy, denom)
        # Gain is necessary here because similarity is always in [-1, 1].
        return _calibrate(similarity, is_training, use_batch_norm,
                          learn_gain=True,
                          gain_init=gain_init,
                          trainable=trainable)


def distance(template, search, is_training,
             trainable=True,
             use_mean=False,
             use_batch_norm=False,
             learn_gain=False,
             gain_init=1,
             scope='distance'):
    '''
    Args:
        template: [b, h, w, c]
        search: [b, s, h, w, c]
    '''
    search = cnn.as_tensor(search)
    num_search_dims = len(search.value.shape)
    if num_search_dims != 5:
        raise ValueError('search should have 5 dims: {}'.format(num_search_dims))

    with tf.variable_scope(scope, 'distance'):
        search = cnn.as_tensor(search)
        # Discard receptive field of template and get underlying tf.Tensor.
        template = cnn.get_value(template)

        num_channels = template.shape[-1].value
        template_size = template.shape[-3:-1].as_list()
        ones = tf.ones(template_size + [num_channels, 1], tf.float32)

        dot_xy = cnn.diag_xcorr(search, template)
        dot_xx = tf.reduce_sum(tf.square(template), axis=(-3, -2, -1), keepdims=True)
        if len(search.value.shape) == 5:
            dot_xx = tf.expand_dims(dot_xx, 1)
        sq_search = cnn.pixelwise(tf.square, search)
        sq_search, restore = cnn.merge_batch_dims(sq_search)
        dot_yy = cnn.nn_conv2d(sq_search, ones, strides=[1, 1, 1, 1], padding='VALID')
        dot_yy = restore(dot_yy)
        # (x - y)**2 = x**2 - 2 x y + y**2
        # sq_dist = dot_xx - 2 * dot_xy + dot_yy
        sq_dist = cnn.pixelwise_binary(
            lambda dot_xy, dot_yy: dot_xx - 2 * dot_xy + dot_yy, dot_xy, dot_yy)
        sq_dist = cnn.pixelwise(
            lambda sq_dist: tf.reduce_sum(sq_dist, axis=-1, keepdims=True), sq_dist)
        if use_mean:
            # Take root-mean-square of difference.
            num_elems = np.prod(template.shape[-3:].as_list())
            sq_dist = cnn.pixelwise(lambda sq_dist: (1 / tf.to_float(num_elems)) * sq_dist, sq_dist)
        dist = cnn.pixelwise(tf.sqrt, sq_dist)
        return _calibrate(dist, is_training, use_batch_norm, learn_gain, gain_init,
                          trainable=trainable)


def depthwise_xcorr(template, search, is_training,
                    trainable=True,
                    learn_spatial_weight=False,
                    use_mean=False,
                    use_batch_norm=False,
                    learn_gain=False,
                    gain_init=1,
                    scope='depthwise_xcorr'):
    '''Computes the cross-correlation of each channel independently.'''
    return _xcorr_general(
        template, search, is_training,
        trainable=trainable,
        use_pre_conv=False,
        learn_spatial_weight=learn_spatial_weight,
        reduce_channels=False,
        use_mean=use_mean,
        use_batch_norm=use_batch_norm,
        learn_gain=learn_gain,
        gain_init=gain_init,
        scope=scope)


def all_pixel_pairs(template, search, is_training,
                    trainable=True,
                    operation='mul',
                    reduce_channels=True,
                    use_mean=True,
                    use_batch_norm=False,
                    learn_gain=False,
                    gain_init=1,
                    scope='all_pixel_pairs'):
    '''
    Args:
        template: cnn.Tensor with shape [n, h_t, w_t, c]
        search: cnn.Tensor with shape [n, s, h_s, w_s, c]

    Returns:
        cnn.Tensor with shape [n, h_s, w_s, h_t * w_t]
    '''
    with tf.variable_scope(scope, 'all_pixel_pairs'):
        template = cnn.as_tensor(template)
        search = cnn.as_tensor(search)
        template_size = template.value.shape[-3:-1].as_list()
        num_channels = template.value.shape[-1].value

        # Break template into 1x1 patches.
        # Then "convolve" (multiply) each with the search image.
        t = template.value
        s = search.value
        # template becomes: [n, 1, ...,   1,   1, h_t, w_t, c]
        # search becomes:   [n, s, ..., h_s, w_s,   1,   1, c]
        t = tf.expand_dims(t, 1)
        t = helpers.expand_dims_n(t, -4, 2)
        s = helpers.expand_dims_n(s, -2, 2)
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
        # response becomes: [n, ..., h_s, w_s, h_t * w_t * c]
        p, _ = helpers.merge_dims(p, -3, None)
        pairs = cnn.Tensor(p, search.fields)

        # TODO: This initialization could be too small?
        normalizer = 1 / (np.prod(template_size) ** 2 * num_channels) if use_mean else 1
        weights_shape = template_size + [np.prod(template_size) * num_channels, 1]
        weights = tf.get_variable('weights', weights_shape, tf.float32,
                                  initializer=tf.constant_initializer(normalizer),
                                  trainable=trainable)
        # TODO: Support depthwise_conv2d (keep channels).
        pairs, restore = cnn.merge_batch_dims(pairs)
        response = cnn.nn_conv2d(pairs, weights, strides=[1, 1, 1, 1], padding='VALID')
        response = restore(response)

        return _calibrate(response, is_training, use_batch_norm, learn_gain, gain_init,
                          trainable=trainable)


def abs_diff(template, search, is_training,
             trainable=True,
             use_pre_conv=True,
             pre_conv_output_dim=256,
             reduce_channels=True,
             use_mean=False,
             use_batch_norm=False,
             scope='abs_diff'):
    '''
    Requires that template is 1x1.

    Args:
        template: [b, ht, wt, c]
        search: [b, s, hs, ws, c]
    '''
    with tf.variable_scope(scope, 'abs_diff'):
        template = cnn.as_tensor(template)
        search = cnn.as_tensor(search)

        if use_pre_conv:
            # Reduce template to 1x1.
            kernel_size = template.value.shape[-3:-1].as_list()

            def pre_conv(x):
                x = cnn.pixelwise(partial(slim.batch_norm, is_training=is_training), x)
                x = cnn.pixelwise(tf.nn.relu, x)
                x, restore = cnn.merge_batch_dims(x)
                x = cnn.slim_conv2d(x, pre_conv_output_dim, kernel_size,
                                    padding='VALID',
                                    activation_fn=None,
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params=dict(is_training=is_training),
                                    scope='conv')
                x = restore(x)
                return x

            # Perform pre-activation because the output layer did not have activations.
            with tf.variable_scope('pre_conv', reuse=False):
                template = pre_conv(template)
            with tf.variable_scope('pre_conv', reuse=True):
                search = pre_conv(search)

        template = cnn.get_value(template)
        template_size = template.shape[-3:-1].as_list()
        if template_size != [1, 1]:
            raise ValueError('template shape is not [1, 1]: {}'.format(template_size))
        # Use broadcasting to perform element-wise operation.
        template = tf.expand_dims(template, 1)
        delta = cnn.pixelwise(lambda x: tf.abs(x - template), search)
        if reduce_channels:
            delta = cnn.channel_sum(delta)
            if use_mean:
                num_channels = template.shape[-1].value
                delta = cnn.pixelwise(lambda x: (1 / tf.to_float(num_channels)) * x, delta)
        # TODO: No bias if attaching more layers?
        return _calibrate(delta, is_training, use_batch_norm, learn_gain=False, gain_init=1,
                          trainable=trainable)


def _calibrate(response, is_training, use_batch_norm, learn_gain, gain_init, trainable=True):
    '''
    Either adds batch_norm (with center and scale) or a scalar bias with optional gain.
    '''
    if use_batch_norm:
        output = cnn.pixelwise(slim.batch_norm, response, center=True, scale=True,
                               is_training=is_training, trainable=trainable)
    else:
        # Add bias (cannot be represented by dot product) and optional gain.
        bias = tf.get_variable('bias', [], tf.float32,
                               initializer=tf.zeros_initializer(),
                               trainable=trainable)
        if learn_gain:
            gain = tf.get_variable('gain', [], tf.float32,
                                   initializer=tf.constant_initializer(gain_init),
                                   trainable=trainable)
            output = cnn.pixelwise(lambda x: gain * x + bias, response)
        else:
            output = cnn.pixelwise(lambda x: x + bias, response)
    return output


def mlp(template, search, is_training,
        trainable=True,
        num_layers=2,
        num_hidden=128,
        join_arch='depthwise_xcorr',
        join_params=None,
        scope='mlp_join'):
    '''Applies an MLP after another join function.

    Args:
        template: [b, ht, wt, c]
        search: [b, s, hs, ws, c]
    '''
    with tf.variable_scope(scope, 'mlp_join'):
        join_params = join_params or {}
        join_fn = BY_NAME[join_arch]
        similarity = join_fn(template, search,
                             is_training=is_training,
                             trainable=trainable,
                             **join_params)
        # similarity: [b, s, h, w, c]
        similarity, restore = cnn.merge_batch_dims(similarity)
        response = cnn.mlp(similarity,
                           num_layers=num_layers,
                           num_hidden=num_hidden,
                           num_outputs=1,
                           is_training=is_training,
                           trainable=trainable)
        return restore(response)


def multi_xcorr(template, search,
                layer_names, template_layers, search_layers, search_image,
                is_training,
                trainable=True,
                use_final_conv=False,
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
                                         trainable=trainable,
                                         use_pre_conv=use_final_conv,
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
                                          trainable=trainable,
                                          use_pre_conv=True,
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

        sizes = {name: _unique(score.value.shape[-3:-1].as_list()) for name, score in scores.items()}
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
    'concat_fc',
    'xcorr',
    'distance',
    'cosine',
    'abs_diff',
    'depthwise_xcorr',
    'all_pixel_pairs',
    'mlp',
]

'''Functions that require 1x1 template.
'''
FULLY_CONNECTED_FNS = [
]

'''Functions that use features from multiple layers.
'''
MULTI_JOIN_FNS = [
    'multi_xcorr',
]

BY_NAME = {name: globals()[name] for name in SINGLE_JOIN_FNS + MULTI_JOIN_FNS}
