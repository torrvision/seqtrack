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

    def __add__(a, b):
        return _call_binary_elementwise(operator.__add__, a, b)

    def __sub__(a, b):
        return _call_binary_elementwise(operator.__sub__, a, b)

    def __mul__(a, b):
        return _call_binary_elementwise(operator.__mul__, a, b)


def get_value(x):
    if isinstance(x, Tensor):
        x = x.value
    assert isinstance(x, tf.Tensor)
    return x


def as_tensor(x, add_to_set=False):
    if not isinstance(x, Tensor):
        assert isinstance(x, tf.Tensor)
        x = Tensor(x)
    if add_to_set:
        x.add_to_set()
    return x


def _call_elementwise(func, x):
    x = as_tensor(x)
    return Tensor(func(x.value), x.fields)


def _elementwise(func):
    return functools.partial(_call_elementwise, func)


nn_relu = _elementwise(tf.nn.relu)
nn_relu6 = _elementwise(tf.nn.relu6)
nn_softmax = _elementwise(tf.nn.softmax)
clip_by_value = _elementwise(tf.clip_by_value)

slim_dropout = _elementwise(slim.dropout)
# slim_softmax = _elementwise(slim.softmax)

channel_sum = _elementwise(lambda x, **kwargs: tf.reduce_sum(x, axis=3, keepdims=True, **kwargs))


def _call_binary_elementwise(func, a, b):
    a = as_tensor(a)
    b = as_tensor(b)
    _assert_is_image(a)
    _assert_is_image(b)
    # TODO: Support broadcasting across space.
    _assert_equal_spatial_dim(a, b)
    return Tensor(func(a.value, b.value), receptive_field.merge_dicts(a.fields, b.fields))


@slim.add_arg_scope
def slim_conv2d(inputs, num_outputs, kernel_size,
                stride=1, padding='SAME', **kwargs):
    kernel_size = layer_utils.n_positive_integers(2, kernel_size)
    if padding != 'VALID' and tuple(kernel_size) != (1, 1):
        raise ValueError('padding must be VALID: {}'.format(padding))
    return _slim_conv2d_valid(inputs, num_outputs, kernel_size,
                              stride=stride, **kwargs)


def _slim_conv2d_valid(inputs, num_outputs, kernel_size,
                       stride=1,
                       data_format=None,
                       rate=1,
                       **kwargs):
    '''
    Args:
        inputs: May be either tf.Tensor or cnn.Tensor.
            The output type will match.
    '''
    if rate != 1:
        raise ValueError('dilation rate not supported yet: {}'.format(rate))
    inputs = as_tensor(inputs)
    outputs = Tensor()
    outputs.value = slim.conv2d(inputs.value, num_outputs, kernel_size,
                                stride=stride, padding='VALID', **kwargs)
    # Update receptive fields.
    relative = receptive_field.conv2d(kernel_size, stride, 'VALID')
    outputs.fields = {k: receptive_field.compose(v, relative) for k, v in inputs.fields.items()}
    return outputs


@slim.add_arg_scope
def slim_max_pool2d(inputs, kernel_size, stride=2, padding='VALID', **kwargs):
    if padding != 'VALID':
        raise ValueError('padding must be VALID: {}'.format(padding))
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
    stride = layer_utils.n_positive_integers(2, stride)
    nhwc_strides = [1, stride[0], stride[1], 1]
    output.value = helpers.diag_xcorr(input.value, filter.value,
                                      strides=nhwc_strides, padding=padding,
                                      name=name)
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
    if len(x.shape) != 4:
        raise ValueError('tensor is not an image: num dims {}'.format(len(x.shape)))


def _assert_equal_spatial_dim(x, y):
    x = get_value(x)
    y = get_value(y)
    x_shape = x.shape.as_list()
    y_shape = y.shape.as_list()
    if any(dim is None for dim in x_shape[1:3]):
        raise ValueError('spatial dim of x is not known: {}'.format(x.shape))
    if any(dim is None for dim in y_shape[1:3]):
        raise ValueError('spatial dim of y is not known: {}'.format(y.shape))
    if x_shape[1:3] != y_shape[1:3]:
        raise ValueError('spatial dims not equal: {}'.format(x.shape, y.shape))
