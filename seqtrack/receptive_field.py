'''

Uses same ReceptiveField type as tf.contrib.receptive_field.

Note that tf.contrib needs to know the image size to perform 'SAME' convolution.
This is necessary to match the implementation of tf.nn.conv2d.
We instead take 'SAME' convolution to mean that the receptive field centers are aligned.
(However, we always adopt 'VALID' convolution, so this is usually not an issue.)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers.utils import n_positive_integers
from tensorflow.contrib.layers.python.layers.utils import two_element_tuple

from tensorflow.contrib.receptive_field.python.util.receptive_field import ReceptiveField


# Not sure how bad this is?
ReceptiveField.__str__ = (
    lambda self: '[size={:s} stride={:s} padding={:s}]'.format(
        self.size, self.stride, self.padding))


def identity():
    return ReceptiveField(size=(1, 1), stride=(1, 1), padding=(0, 0))


def compose(yx, zy):
    '''Obtains the composition of two receptive fields.

    x -> y -> z

    Careful: The order is from the inside out, not outside in!
    That is, if z = f(g(x)), then use compose(g, f).

    Args:
        yx: Receptive field of y with respect to x.
        zy: Receptive field of z with respect to y.

    Returns:
        Receptive field of z with respect to x.
    '''
    assert isinstance(yx, ReceptiveField), type(yx)
    assert isinstance(zy, ReceptiveField), type(zy)
    # Translation of 1 pixel in z corresponds to translation of zy.stride pixels in y.
    # Translation of 1 pixel in y corresponds to translation of yx.stride pixels in x.
    zx_stride = zy.stride * yx.stride
    # 1 pixel in z corresponds to zy.size pixels in y.
    # 1 pixel in y corresponds to yx.size pixels in x.
    # n pixels in y corresponds to yx.size + (n - 1) * yx.stride pixels in x.
    zx_size = yx.size + (zy.size - 1) * yx.stride
    # Let r.offset be defined as -r.padding.
    # Pixel 0 in z corresponds to rect starting at zy.offset in y.
    # Pixel 0 in y corresponds to rect starting at yx.offset in x.
    # Pixel i in y corresponds to rect starting at yx.offset + i * yx.stride in x.
    # Therefore:
    # zx_offset = yx_offset + (zy_offset * yx.stride)
    # -zx_padding = -yx.padding + (-zy.padding * yx.stride)
    # zx_padding = yx.padding + zy.padding * yx.stride
    # TODO: Support None in only some dimensions?
    if not _any_is_none(yx.padding) and not _any_is_none(zy.padding):
        zx_padding = yx.padding + zy.padding * yx.stride
    else:
        zx_padding = [None, None]
    return ReceptiveField(size=zx_size, stride=zx_stride, padding=zx_padding)


def conv2d(kernel_size, stride, padding):
    '''Returns the receptive field of a convolutional layer.'''
    kernel_size = np.array(n_positive_integers(2, kernel_size))
    stride = np.array(n_positive_integers(2, stride))
    # Get relative receptive field.
    if padding == 'SAME':
        assert np.all(kernel_size % 2 == 1), 'kernel size must be odd with SAME convolution'
        half_kernel = (kernel_size - 1) // 2
        padding_amount = -half_kernel
    elif padding == 'VALID':
        padding_amount = np.zeros_like(kernel_size)
    else:
        raise ValueError('unknown padding: {}'.format(padding))
    return ReceptiveField(size=kernel_size, stride=stride, padding=padding_amount)


# def local_op(kernel_size, stride, padding):
#     '''Computes the receptive field of a filter.'''
#     kernel_size = np.array(n_positive_integers(2, kernel_size))
#     stride = np.array(n_positive_integers(2, stride))
#     # Get relative receptive field.
#     if padding == 'SAME':
#         assert np.all(kernel_size % 2 == 1)
#         half = (kernel_size - 1) // 2
#         rect = IntRect(-half, half + 1)
#         rel_rf = ReceptiveField(rect=rect, stride=stride)
#     elif padding == 'VALID':
#         rect = IntRect(np.zeros_like(kernel_size), kernel_size)
#         rel_rf = ReceptiveField(rect=rect, stride=stride)
#     else:
#         raise ValueError('invalid padding: {}'.format(padding))
#     return rel_rf


def merge_dicts(a, b):
    keys = set(list(a.keys()) + list(b.keys()))
    c = {}
    for k in keys:
        c[k] = merge(a.get(k, None), b.get(k, None))
    return c


def merge(a, b, require_center_alignment=True):
    '''Merges two receptive fields.'''
    if b is None:
        return a
    if a is None:
        return b
    assert not _any_is_none(a.padding)
    assert not _any_is_none(b.padding)
    if not np.array_equal(a.stride, b.stride):
        raise ValueError('strides are not equal: {}, {}'.format(a.stride, b.stride))
    stride = a.stride
    # Integer rectangle goes from -padding to -padding + size - 1 (inclusive).
    # This means the center is -padding + (size - 1) / 2.
    # To avoid the divide by 2, we can examine -2 * padding + (size - 1).
    if require_center_alignment:
        a_center = -2 * a.padding + a.size - 1
        b_center = -2 * b.padding + b.size - 1
        if not np.array_equal(a_center, b_center):
            raise ValueError('centers are not aligned: {:s}, {:s}'.format(a, b))
    # Take union of rectangles.
    a_min, a_max = -a.padding, -a.padding + a.size
    b_min, b_max = -b.padding, -b.padding + b.size
    r_min = np.minimum(a_min, b_min)
    r_max = np.maximum(a_max, b_max)
    size = r_max - r_min
    padding = -r_min
    return ReceptiveField(size=size, stride=stride, padding=padding)


def _any_is_none(x):
    return any(elem is None for elem in x)


# class IntRect:
#     '''Describes a rectangle.
# 
#     The elements of the rectangle satisfy min <= u < max.
#     '''
# 
#     def __init__(self, min=(0, 0), max=(0, 0)):
#         # Numbers should be integers, but it is useful to have +inf and -inf.
#         self.min = np.array(min)
#         self.max = np.array(max)
# 
#     def __eq__(a, b):
#         return np.array_equal(a.min, b.min) and np.array_equal(a.max, b.max)
# 
#     def __str__(self):
#         return '{}-{}'.format(tuple(self.min), tuple(self.max))
# 
#     def empty(self):
#         return not all(self.min < self.max)
# 
#     def size(self):
#         return self.max - self.min
# 
#     def int_center(self):
#         '''Finds the center pixel of a region.'''
#         s = self.min + self.max - 1
#         if not all(s % 2 == 0):
#             raise ValueError('rectangle does not have integer center')
#         return s // 2
# 
#     def intersect(a, b):
#         return IntRect(np.maximum(a.min, b.min), np.minimum(a.max, b.max))
# 
#     def union(a, b):
#         return IntRect(np.minimum(a.min, b.min), np.maximum(a.max, b.max))


# def infinite_rect():
#     return IntRect(min=(float('-inf'), float('-inf')), max=(float('+inf'), float('+inf')))


# class ReceptiveField:
#     '''Describes the receptive fields (in an earlier layer) of all pixels in a later layer.
# 
#     If y has receptive field rf in x, then pixel y[v] depends on pixels x[u] where
#         v*rf.stride + rf.rect.min <= u < v*rf.stride + rf.rect.max
#     '''
# 
#     def __init__(self, rect=IntRect(), stride=(0, 0)):
#         self.rect = rect
#         self.stride = np.array(stride)
# 
#     def __eq__(a, b):
#         return a.rect == b.rect and a.stride == b.stride
# 
#     def __str__(self):
#         return '{}@{}'.format(self.rect, tuple(self.stride))
# 
#     def union(a, b, assert_aligned=False):
#         '''Computes the receptive field of the next layer.
# 
#         Given relative receptive field in prev of pixels in curr.
#         input -> ... -> prev -> curr
#         '''
#         if not np.array_equal(a.stride, b.stride):
#             raise RuntimeError('strides are different: {}, {}'.format(a.stride, b.stride))
#         if assert_aligned:
#             a_center = a.rect.int_center()
#             b_center = b.rect.int_center()
#             if not np.array_equal(a_center, b_center):
#                 raise RuntimeError('centers not equal: {}, {}'.format(a_center, b_center))
#         return ReceptiveField(rect=a.rect.union(b.rect), stride=a.stride)


# def identity():
#     return ReceptiveField(rect=IntRect((0, 0), (1, 1)), stride=(1, 1))


# def compose(prev_rf, rel_rf):
#     '''Computes the receptive field of the next layer.
# 
#     Given relative receptive field in prev of pixels in curr.
#     input -> ... -> prev -> curr
#     '''
#     # curr[v] depends on prev[u] for
#     #   v*rel_rf.stride + rel_rf.rect.min <= u <= v*rel_rf.stride + rel_rf.rect.max - 1
#     # and prev[u] depends on input[t] for
#     #   u*prev_rf.stride + prev_rf.rect.min <= t <= u*prev_rf.stride + prev_rf.rect.max - 1
#     #
#     # Therefore, curr[v] depends on input[t] for t between
#     #   (v*rel_rf.stride + rel_rf.rect.min) * prev_rf.stride + prev_rf.rect.min
#     #   (v*rel_rf.stride + rel_rf.rect.max - 1) * prev_rf.stride + prev_rf.rect.max - 1
#     # or equivalently
#     #   v*(rel_rf.stride*prev_rf.stride) + (rel_rf.rect.min*prev_rf.stride + prev_rf.rect.min)
#     #   v*(rel_rf.stride*prev_rf.stride) + ((rel_rf.rect.max-1)*prev_rf.stride + prev_rf.rect.max) - 1
#     stride = prev_rf.stride * rel_rf.stride
#     min = prev_rf.rect.min + prev_rf.stride * rel_rf.rect.min
#     max = prev_rf.rect.max + prev_rf.stride * (rel_rf.rect.max - 1)
#     return ReceptiveField(IntRect(min, max), stride)


def assert_center_alignment(input_size, output_size, field):
    '''
    Args:
        input_size: (height, width)
        output_size: (height, width)
        field: ReceptiveField, which uses (height, width)
    '''
    input_size = np.array(n_positive_integers(2, input_size))
    output_size = np.array(n_positive_integers(2, output_size))
    # First receptive field spans {first_min, ..., first_max - 1}
    first_min = field.padding
    first_max = first_min + field.size
    # Last receptive field spans {last_min, ..., last_max - 1}
    last_min = field.padding + (output_size - 1) * field.stride
    last_max = last_min + field.size
    # If gap_before is equal to gap_after, then center of response map
    # corresponds to center of search image.
    gap_before = first_min
    gap_after = input_size - last_max
    if not np.array_equal(gap_before, gap_after):
        raise AssertionError('centers are not aligned: before {}, after {}'.format(
            gap_before, gap_after))


# def centers_in_input(output_size, rf):
#     '''Gives the center pixels of the receptive fields of the corners of the activation map.'''
#     output_size = np.asarray(n_positive_integers(2, output_size))
#     center_min = rf.rect.int_center()
#     center_max = center_min + (output_size - 1) * rf.stride
#     return center_min, center_max
