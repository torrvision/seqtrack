'''Wraps standard conv-net functions with ability to compute receptive field.

While tf.contrib.receptive_field can be used if the network comprises only standard layers,
it cannot support operations like:
  - cross-correlation (required by convolutional siamese net)
  - cropping (required by ResNet without padding)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import functools
import operator

from seqtrack import receptive_field
from seqtrack import helpers

from tensorflow.contrib.layers.python.layers import utils as layer_utils


# We use this class to perform our custom receptive field computation.
# Each tensor maintains its receptive fields with respect to inputs of interest.
# We could alternatively define a graph, however this seems more involved.

class Tensor(object):
    '''Describes a tensor that tracks receptive fields.'''

    def __init__(self, value=None, fields=None):
        '''
        Args:
            fields: Dictionary that maps tf.Tensor to receptive_field.ReceptiveField.
        '''
        self.value = value
        self.fields = fields or {}

    def get_shape(self):
        return self.value.get_shape()

    def add_to_set(self):
        self.fields[self.value] = receptive_field.identity()

    # TODO: How to support 1 + x as well as x + 1?

    def __add__(a, b):
        return pixelwise_binary(operator.__add__, a, b)

    def __sub__(a, b):
        return pixelwise_binary(operator.__sub__, a, b)

    def __mul__(a, b):
        return pixelwise_binary(operator.__mul__, a, b)


def get_value(x):
    if isinstance(x, Tensor):
        x = x.value
    assert isinstance(x, (tf.Tensor, tf.Variable))
    return x


def as_tensor(x, add_to_set=False):
    if not isinstance(x, Tensor):
        assert isinstance(x, (tf.Tensor, tf.Variable))
        x = Tensor(x)
    if add_to_set:
        x.add_to_set()
    return x


def pixelwise(func, x, *args, **kwargs):
    '''Calls a unary function that operates on each spatial location independently.

    This does not affect the receptive fields.
    '''
    x = as_tensor(x)
    return Tensor(func(x.value, *args, **kwargs), x.fields)


def partial_pixelwise(func, **kwargs):
    '''Partial evaluation of pixelwise().

    Useful for defining pixelwise functions.
    '''
    return functools.partial(pixelwise, func, **kwargs)


nn_relu = partial_pixelwise(tf.nn.relu)
nn_relu6 = partial_pixelwise(tf.nn.relu6)
nn_softmax = partial_pixelwise(tf.nn.softmax)
clip_by_value = partial_pixelwise(tf.clip_by_value)

slim_dropout = partial_pixelwise(slim.dropout)
# slim_softmax = partial_pixelwise(slim.softmax)

channel_sum = partial_pixelwise(
    lambda x, **kwargs: tf.reduce_sum(x, axis=-1, keepdims=True, **kwargs))
channel_mean = partial_pixelwise(
    lambda x, **kwargs: tf.reduce_mean(x, axis=-1, keepdims=True, **kwargs))


def pixelwise_binary(func, a, b):
    a = as_tensor(a)
    b = as_tensor(b)
    _assert_is_image(a)
    _assert_is_image(b)
    # TODO: Support broadcasting across space.
    _assert_equal_spatial_dim(a, b)
    return Tensor(func(a.value, b.value), receptive_field.merge_dicts(a.fields, b.fields))


def nn_conv2d(input, filter, strides, padding, **kwargs):
    # Assumes data_format == 'NHWC'
    # Assumes dilations == [1, 1, 1, 1]
    input = as_tensor(input)
    filter = as_tensor(filter)
    assert len(input.value.shape) == 4
    assert len(filter.value.shape) == 4
    kernel_size = filter.value.shape[0:2].as_list()
    # Check that strides are [1, ..., 1]
    assert len(strides) == 4
    assert strides[0] == 1
    assert strides[3] == 1
    spatial_stride = strides[1:3]

    # if padding != 'VALID' and tuple(kernel_size) != (1, 1):
    #     raise ValueError('padding must be VALID: {}'.format(padding))

    output = Tensor()
    output.value = tf.nn.conv2d(input.value, filter.value,
                                strides=strides, padding=padding, **kwargs)
    # Update receptive fields.
    relative = receptive_field.conv2d(kernel_size, spatial_stride, 'VALID')
    output.fields = {k: receptive_field.compose(v, relative) for k, v in input.fields.items()}
    return output


@slim.add_arg_scope
def slim_conv2d(inputs, num_outputs, kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                **kwargs):
    kernel_size = layer_utils.n_positive_integers(2, kernel_size)
    # if padding != 'VALID' and tuple(kernel_size) != (1, 1):
    #     raise ValueError('padding must be VALID: {}'.format(padding))
    # return _slim_conv2d_valid(inputs, num_outputs, kernel_size,
    #                           stride=stride, **kwargs)
    if rate != 1:
        raise ValueError('dilation rate not supported yet: {}'.format(rate))
    inputs = as_tensor(inputs)
    outputs = Tensor()
    outputs.value = slim.conv2d(inputs.value, num_outputs, kernel_size,
                                stride=stride, padding=padding, **kwargs)
    # Update receptive fields.
    relative = receptive_field.conv2d(kernel_size, stride, padding)
    outputs.fields = {k: receptive_field.compose(v, relative) for k, v in inputs.fields.items()}
    return outputs


@slim.add_arg_scope
def slim_max_pool2d(inputs, kernel_size, stride=2, padding='VALID', **kwargs):
    # if padding != 'VALID':
    #     raise ValueError('padding must be VALID: {}'.format(padding))
    inputs = as_tensor(inputs)
    outputs = Tensor()
    outputs.value = slim.max_pool2d(inputs.value, kernel_size,
                                    stride=stride, padding=padding, **kwargs)
    relative = receptive_field.conv2d(kernel_size, stride, padding)
    outputs.fields = {k: receptive_field.compose(v, relative) for k, v in inputs.fields.items()}
    return outputs


def diag_xcorr(input, filter, stride=1, padding='VALID', name='diag_xcorr'):
    '''
    Args:
        input: [b, ..., h, w, c]
        filter: [b, fh, fw, c]
        stride: Either an integer or length 2 (as for slim operations).
    '''
    if padding != 'VALID':
        raise ValueError('padding must be VALID: {}'.format(padding))
    input = as_tensor(input)
    filter = as_tensor(filter)
    _assert_is_image(input)
    _assert_is_image(filter)
    kernel_size = filter.value.shape.as_list()[-3:-1]
    assert all(kernel_size)  # Must not be None or 0.

    output = Tensor()
    output.value = helpers.diag_xcorr(input.value, filter.value,
                                      stride=stride, padding=padding, name=name)
    # TODO: Incorporate receptive field of filter (fully-connected).
    field_output_input = receptive_field.conv2d(kernel_size, stride, padding)
    output.fields = {k: receptive_field.compose(v, field_output_input)
                     for k, v in input.fields.items()}
    return output


# def xcorr(input, filter, **kwargs):
#     '''
#     Args:
#         kwargs: For depthwise_xcorr2d.
#     '''
#     return channel_sum(diag_xcorr(input, filter, **kwargs))


def concat():
    raise NotImplementedError('concat')


def spatial_trim(x, first, last):
    x = as_tensor(x)
    first = np.array(layer_utils.two_element_tuple(first))
    last = np.array(layer_utils.two_element_tuple(last))
    stop = [-amount if amount else None for amount in last]
    value = x.value[:, first[0]:stop[0], first[1]:stop[1], :]
    relative = receptive_field.ReceptiveField(size=1, stride=1, padding=-first)
    fields = {k: receptive_field.compose(v, relative) for k, v in x.fields.items()}
    return Tensor(value, fields)


# We probably won't use this because it violates convolutionality?
# def spatial_pad(x, first, last):
#     x = as_tensor(x)
#     first = np.array(layer_utils.two_element_tuple(first))
#     last = np.array(layer_utils.two_element_tuple(last))
#     value = tf.pad(x.value, [[0, 0], [first[0], last[0]], [first[1], last[1]], [0, 0]])
#     relative = receptive_field.ReceptiveField(size=1, stride=1, padding=first)
#     fields = {k: receptive_field.compose(v, relative) for k, v in x.fields.items()}
#     return Tensor(value, fields)


# TODO: Support SAME convolution through explicit padding.
# def conv2d_explicit_pad(inputs, num_outputs, kernel_size,
#                         stride=1, padding='SAME', **kwargs):
#     '''
#     Args:
#         kwargs: For slim.conv2d
#     '''
#     if padding == 'SAME':
#         # Check that kernel size is odd.
#         kernel_size = np.asarray(layer_utils.n_positive_integers(2, kernel_size))
#         if not all(kernel_size % 2 == 1):
#             raise ValueError('kernel_size must be odd for SAME padding: {}'.format(kernel_size))
#         # Pad with half of kernel size.
#         half = (kernel_size - 1) // 2
#         inputs = pad2d(inputs, paddings=[[x, x] for x in half])
# 
#     return conv2d_valid(inputs, num_outputs, kernel_size,
#                         stride=stride, **kwargs)


def _assert_is_image(x):
    x = get_value(x)
    if not len(x.shape) >= 3:
        raise ValueError('tensor is not an image: num dims {}'.format(len(x.shape)))


def _assert_equal_spatial_dim(x, y):
    x = get_value(x)
    y = get_value(y)
    x_size = x.shape[1:3].as_list()
    y_size = y.shape[1:3].as_list()
    # Do not permit unknown spatial dimension.
    # if any(dim is None for dim in x_shape[1:3]):
    #     raise ValueError('spatial dim of x is not known: {}'.format(x.shape))
    # if any(dim is None for dim in y_shape[1:3]):
    #     raise ValueError('spatial dim of y is not known: {}'.format(y.shape))
    # Protect against broadcasting.
    if x_size == [1, 1] and y_size != [1, 1]:
        raise ValueError('broadcasting not supported')
    if y_size == [1, 1] and x_size != [1, 1]:
        raise ValueError('broadcasting not supported')
    if x_size != y_size:
        raise ValueError('spatial dims not equal: {}'.format(x.shape, y.shape))


@partial_pixelwise
def mlp(net, num_layers, num_hidden, num_outputs,
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu,
        output_normalizer_fn=None,
        output_activation_fn=None,
        scope='mlp'):
    with tf.variable_scope(scope, 'mlp'):
        for i in range(num_layers - 1):
            net = slim.conv2d(net, num_hidden, kernel_size=1, stride=1, padding='VALID',
                              normalizer_fn=normalizer_fn,
                              activation_fn=activation_fn,
                              scope='fc{}'.format(i + 1))
        net = slim.conv2d(net, num_outputs, kernel_size=1, stride=1, padding='VALID',
                          normalizer_fn=output_normalizer_fn,
                          activation_fn=output_activation_fn,
                          scope='fc{}'.format(num_layers))
        return net


def upsample(x, rate, method=0):
    x = as_tensor(x)
    # Upsampling (with align_corners true) does not modify the size and padding of field.
    # This is not entirely true e.g. bilinear uses adjacent pixels too.
    # TODO: Do we want to represent this or not?
    input_size = np.array(x.value.shape[1:3].as_list())
    # For example: reshape 11 => 31 with rate 3.
    output_size = (input_size - 1) * rate + 1
    output_value = tf.image.resize_images(x.value, output_size, method=method, align_corners=True)
    output_fields = {}
    for key, field in x.fields.items():
        assert np.all(field.stride % rate == 0)
        output_fields[key] = receptive_field.ReceptiveField(
            size=field.size, stride=field.stride // rate, padding=field.padding)
    return Tensor(output_value, output_fields)


def merge_batch_dims(x):
    '''Merges all dimensions except the last three.

    Returns:
        (merged, restore_fn)
    '''
    x = as_tensor(x)
    ndim = len(x.value.shape)
    assert ndim >= 4
    if ndim == 4:
        return x, _identity
    # Merge all dimensions except last three.
    value, restore_fn = helpers.merge_dims(x.value, None, -3)
    y = Tensor(value, x.fields)
    return y, partial_pixelwise(restore_fn, axis=0)


def _identity(x):
    return x
