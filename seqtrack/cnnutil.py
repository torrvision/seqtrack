from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers
from tensorflow.contrib.layers.python.layers.utils import two_element_tuple


class IntRect:
    '''Describes a rectangle.

    The elements of the rectangle satisfy min <= u < max.
    '''

    def __init__(self, min=(0, 0), max=(0, 0)):
        # Numbers should be integers, but it is useful to have +inf and -inf.
        self.min = np.array(min)
        self.max = np.array(max)

    def __eq__(a, b):
        return np.array_equal(a.min, b.min) and np.array_equal(a.max, b.max)

    def __str__(self):
        return '{}-{}'.format(tuple(self.min), tuple(self.max))

    def empty(self):
        return not all(self.min < self.max)

    def size(self):
        return self.max - self.min

    def int_center(self):
        '''Finds the center pixel of a region.'''
        s = self.min + self.max - 1
        if not all(s % 2 == 0):
            raise ValueError('rectangle does not have integer center')
        return s // 2

    def intersect(a, b):
        return IntRect(np.maximum(a.min, b.min), np.minimum(a.max, b.max))

    def union(a, b):
        return IntRect(np.minimum(a.min, b.min), np.maximum(a.max, b.max))


class ReceptiveField:
    '''Describes the receptive fields (in an earlier layer) of all pixels in a later layer.

    If y has receptive field rf in x, then pixel y[v] depends on pixels x[u] where
        v*rf.stride + rf.rect.min <= u < v*rf.stride + rf.rect.max
    '''

    def __init__(self, rect=IntRect(), stride=(0, 0)):
        self.rect = rect
        self.stride = np.array(stride)

    def __eq__(a, b):
        return a.rect == b.rect and a.stride == b.stride

    def __str__(self):
        return '{}@{}'.format(self.rect, tuple(self.stride))

    def union(a, b, assert_aligned=False):
        '''Computes the receptive field of the next layer.

        Given relative receptive field in prev of pixels in curr.
        input -> ... -> prev -> curr
        '''
        if not np.array_equal(a.stride, b.stride):
            raise RuntimeError('strides are different: {}, {}'.format(a.stride, b.stride))
        if assert_aligned:
            a_center = a.rect.int_center()
            b_center = b.rect.int_center()
            if not np.array_equal(a_center, b_center):
                raise RuntimeError('centers not equal: {}, {}'.format(a_center, b_center))
        return ReceptiveField(rect=a.rect.union(b.rect), stride=a.stride)


def identity_rf():
    return ReceptiveField(rect=IntRect((0, 0), (1, 1)), stride=(1, 1))


def infinite_rect():
    return IntRect(min=(float('-inf'), float('-inf')), max=(float('+inf'), float('+inf')))


def compose_rf(prev_rf, rel_rf):
    '''Computes the receptive field of the next layer.

    Given relative receptive field in prev of pixels in curr.
    input -> ... -> prev -> curr
    '''
    # curr[v] depends on prev[u] for
    #   v*rel_rf.stride + rel_rf.rect.min <= u <= v*rel_rf.stride + rel_rf.rect.max - 1
    # and prev[u] depends on input[t] for
    #   u*prev_rf.stride + prev_rf.rect.min <= t <= u*prev_rf.stride + prev_rf.rect.max - 1
    #
    # Therefore, curr[v] depends on input[t] for t between
    #   (v*rel_rf.stride + rel_rf.rect.min) * prev_rf.stride + prev_rf.rect.min
    #   (v*rel_rf.stride + rel_rf.rect.max - 1) * prev_rf.stride + prev_rf.rect.max - 1
    # or equivalently
    #   v*(rel_rf.stride*prev_rf.stride) + (rel_rf.rect.min*prev_rf.stride + prev_rf.rect.min)
    #   v*(rel_rf.stride*prev_rf.stride) + ((rel_rf.rect.max-1)*prev_rf.stride + prev_rf.rect.max) - 1
    stride = prev_rf.stride * rel_rf.stride
    min = prev_rf.rect.min + prev_rf.stride * rel_rf.rect.min
    max = prev_rf.rect.max + prev_rf.stride * (rel_rf.rect.max - 1)
    return ReceptiveField(IntRect(min, max), stride)


def rf_centers_in_input(output_size, rf):
    '''Gives the center pixels of the receptive fields of the corners of the activation map.'''
    output_size = np.asarray(n_positive_integers(2, output_size))
    center_min = rf.rect.int_center()
    center_max = center_min + (output_size - 1) * rf.stride
    return center_min, center_max


def assert_center_alignment(input_size, output_size, rf):
    '''
    Args:
        input_size: (height, width)
        output_size: (height, width)
        rf: cnnutil.ReceptiveField, which uses (height, width)
    '''
    input_size = np.array(n_positive_integers(2, input_size))
    output_size = np.array(n_positive_integers(2, output_size))

    min_pt = rf.rect.int_center()
    max_pt = min_pt + rf.stride * (output_size - 1) + 1
    gap_before = min_pt
    gap_after = input_size - max_pt
    # If gap_before is equal to gap_after, then center of response map
    # corresponds to center of search image.
    np.testing.assert_array_equal(gap_before, gap_after)


def conv2d_rf(inputs, input_rfs, num_outputs, kernel_size, stride=1, padding='SAME', **kwargs):
    '''Wraps slim.conv2d to include receptive field calculation.

    input_rfs['var_name'] is the receptive field of input w.r.t. var_name.
    output_rfs['var_name'] is the receptive field of output w.r.t. var_name.
    '''
    if input_rfs is None:
        input_rfs = {}
    if inputs is not None:
        assert len(inputs.shape) == 4  # otherwise slim.conv2d does higher-dim convolution
        outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding, **kwargs)
    else:
        outputs = None
    rel_rf = _filter_rf(kernel_size, stride, padding)
    output_rfs = {k: compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def max_pool2d_rf(inputs, input_rfs, kernel_size, stride=2, padding='VALID', **kwargs):
    '''Wraps slim.max_pool2d to include receptive field calculation.'''
    if input_rfs is None:
        input_rfs = {}
    if inputs is not None:
        outputs = slim.max_pool2d(inputs, kernel_size, stride, padding, **kwargs)
    else:
        outputs = None
    rel_rf = _filter_rf(kernel_size, stride, padding)
    output_rfs = {k: compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def _filter_rf(kernel_size, stride, padding):
    '''Computes the receptive field of a filter.'''
    kernel_size = np.array(n_positive_integers(2, kernel_size))
    stride = np.array(n_positive_integers(2, stride))
    # Get relative receptive field.
    if padding == 'SAME':
        assert np.all(kernel_size % 2 == 1)
        half = (kernel_size - 1) // 2
        rect = IntRect(-half, half + 1)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    elif padding == 'VALID':
        rect = IntRect(np.zeros_like(kernel_size), kernel_size)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    else:
        raise ValueError('invalid padding: {}'.format(padding))
    return rel_rf


class Tensor(object):
    '''Describes a tensor that can track receptive fields.'''

    def __init__(self, value, rfs):
        self.value = value
        self.rfs = rfs

    def get_shape(self):
        return self.value.get_shape()


def _elementwise(func):
    return functools.partial(_call_elementwise, func)


def _call_elementwise(func, inputs, *args, **kwargs):
    if isinstance(inputs, Tensor):
        output_value = func(inputs.value, *args, **kwargs)
        return Tensor(output_value, rfs=inputs.rfs)
    else:
        return func(inputs, *args, **kwargs)


dropout = _elementwise(slim.dropout)
softmax = _elementwise(slim.softmax)
relu = _elementwise(tf.nn.relu)
relu6 = _elementwise(tf.nn.relu6)
clip_by_value = _elementwise(tf.clip_by_value)


def _with_optional_rf(func):
    return functools.partial(_call_with_optional_rf, func)


def _call_with_optional_rf(func, inputs, *args, **kwargs):
    if isinstance(inputs, Tensor):
        output_value, output_rfs = func(inputs.value, inputs.rfs, *args, **kwargs)
        return Tensor(output_value, rfs=output_rfs)
    else:
        output_value, _ = func(inputs, None, *args, **kwargs)
        return output_value


def spatial_trim_rf(x, input_rfs, first, last):
    input_rfs = input_rfs or {}
    first = np.array(two_element_tuple(first))
    last = np.array(two_element_tuple(last))
    start = [amount if amount else None for amount in first]
    stop = [-amount if amount else None for amount in last]
    y = x[:, start[0]:stop[0], start[1]:stop[1], :]
    rel_rf = ReceptiveField(rect=IntRect(min=first, max=first + 1), stride=1)
    output_rfs = {k: compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return y, output_rfs


def spatial_pad_rf(x, input_rfs, first, last):
    input_rfs = input_rfs or {}
    first = np.array(two_element_tuple(first))
    last = np.array(two_element_tuple(last))
    y = tf.pad(x, [[0, 0], [first[0], last[0]], [first[1], last[1]], [0, 0]])
    rel_rf = ReceptiveField(rect=IntRect(min=-first, max=-first + 1), stride=1)
    output_rfs = {k: compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return y, output_rfs


spatial_trim = _with_optional_rf(spatial_trim_rf)
spatial_pad = _with_optional_rf(spatial_pad_rf)


@tf.contrib.framework.add_arg_scope
def conv2d(inputs, *args, **kwargs):
    '''Drop-in replacement for slim.conv2d that supports Tensor objects.'''
    # if isinstance(inputs, Tensor):
    #     output_value, output_rfs = conv2d_rf(inputs.value, inputs.rfs, *args, **kwargs)
    #     return Tensor(output_value, rfs=output_rfs)
    # else:
    #     output_value, _ = conv2d_rf(inputs, None, *args, **kwargs)
    #     return output_value
    return _call_with_optional_rf(conv2d_rf, inputs, *args, **kwargs)


@tf.contrib.framework.add_arg_scope
def max_pool2d(inputs, *args, **kwargs):
    # if isinstance(inputs, Tensor):
    #     output_value, output_rfs = max_pool2d_rf(inputs.value, inputs.rfs, *args, **kwargs)
    #     return Tensor(output_value, rfs=output_rfs)
    # else:
    #     output_value, _ = max_pool2d_rf(inputs, None, *args, **kwargs)
    #     return output_value
    return _call_with_optional_rf(max_pool2d_rf, inputs, *args, **kwargs)


def add_rf(x, x_rfs, y, y_rfs, assert_aligned=False):
    x_rfs = x_rfs or {}
    y_rfs = y_rfs or {}

    # Do not allow broadcasting.
    # assert x.shape == y.shape, 'shapes not equal: {}, {}'.format(x.shape, y.shape)
    z = x + y

    keys = set(itertools.chain(x_rfs.keys(), y_rfs.keys()))
    z_rfs = {}
    for k in keys:
        if k in x_rfs and k in y_rfs:
            z_rfs[k] = x_rfs[k].union(y_rfs[k], assert_aligned=assert_aligned)
        elif k in x_rfs:
            z_rfs[k] = x_rfs[k]
        elif k in y_rfs:
            z_rfs[k] = y_rfs[k]
    return z, z_rfs


def add(x, y, **kwargs):
    assert isinstance(x, Tensor) == isinstance(y, Tensor)
    if isinstance(x, Tensor):
        z_value, z_rfs = add_rf(x.value, x.rfs, y.value, y.rfs, **kwargs)
        return Tensor(z_value, rfs=z_rfs)
    else:
        z, _ = add_rf(x, None, y, None, **kwargs)
        return z
