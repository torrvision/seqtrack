from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from seqtrack import geom
from seqtrack.helpers import most_static_shape
from seqtrack.helpers import expand_dims_n
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers


def foreground_labels_grid(im_size, rect, name='fg_labels_grid', **kwargs):
    with tf.name_scope(name) as scope:
        position = make_grid_centers(im_size)
        return foreground_labels(position, rect, name=scope, **kwargs)


def make_grid_centers(im_size, name='make_grid_centers'):
    '''Make grid of center positions of each pixel.

    Args:
        im_size: (height, width)

    Returns:
        Tensor grid of size [height, width, 2] as (x, y).
    '''
    with tf.name_scope(name) as scope:
        size_y, size_x = n_positive_integers(2, im_size)
        range_y = (tf.to_float(tf.range(size_y)) + 0.5) / float(size_y)
        range_x = (tf.to_float(tf.range(size_x)) + 0.5) / float(size_x)
        grid_y, grid_x = tf.meshgrid(range_y, range_x, indexing='ij')
        # grid = tf.stack((tf.tile(tf.expand_dims(range_x, 0), [size_y, 1]),
        #                  tf.tile(tf.expand_dims(range_y, 1), [1, size_x])), axis=-1)
        return tf.stack((grid_x, grid_y), axis=-1, name=scope)


def foreground_labels(position, rect, shape='rect', sigma=0.3, name='foreground_labels'):
    '''
    Args:
        position: [h, w, 2]
        rect: [b, 4]
        shape: gaussian, rect

    Returns: labels, has_label
        labels: Tensor with shape [b, h, w]
        has_label: Tensor of bools with shape [b, h, w]
    '''
    with tf.name_scope(name) as scope:
        label_shape = most_static_shape(rect)[0:1] + most_static_shape(position)[0:2]

        rect = expand_dims_n(rect, -2, n=2)  # [b, 4] -> [b, 1, 1, 4]
        min_pt, max_pt = geom.rect_min_max(rect)
        center, size = 0.5 * (min_pt + max_pt), max_pt - min_pt

        has_label = tf.ones(label_shape, tf.bool)  # [b, h, w]
        if shape == 'rect':
            is_pos = tf.logical_and(tf.reduce_all(min_pt <= position, axis=-1),
                                    tf.reduce_all(position <= max_pt, axis=-1))
            labels = tf.to_float(is_pos)
        elif shape == 'gaussian':
            relative_error = (position - center) / size
            labels = tf.exp(-0.5 * tf.reduce_sum(tf.square(relative_error / sigma), axis=-1))
        else:
            raise ValueError('unknown shape: {}'.format(shape))
        return labels, has_label


def translation_labels(position, rect, shape, radius_pos=0.3, radius_neg=0.3, sigma=0.3,
                       name='translation_labels'):
    '''
    Args:
        position: Tensor with shape [h, w, 2]
        rect: Tensor of shape [b, 4]
        shape: gaussian, threshold

    Returns:
        labels, has_label

    Parameters radius and sigma are relative to search region not object!

    reduce_sum(weight, axis=(-2, -1)) = 1
    '''
    with tf.name_scope(name) as scope:
        label_shape = most_static_shape(rect)[0:1] + most_static_shape(position)[-3:-1]

        rect = expand_dims_n(rect, -2, n=2)  # [b, 4] -> [b, 1, 1, 4]
        min_pt, max_pt = geom.rect_min_max(rect)
        center = 0.5 * (min_pt + max_pt)

        error = position - center
        sqr_dist = tf.reduce_sum(tf.square(error), axis=-1)
        dist = tf.norm(error, axis=-1)

        has_label = tf.ones(label_shape, tf.bool)  # [b, h, w]
        if shape == 'gaussian':
            labels = tf.exp(-0.5 * sqr_dist / tf.square(float(sigma)))
        elif shape == 'threshold':
            is_pos = dist <= radius_pos
            is_neg = radius_neg <= dist
            has_label = tf.logical_or(is_pos, is_neg)
            is_pos = tf.to_float(is_pos)
            is_neg = tf.to_float(is_neg)
            labels = is_pos / (is_pos + is_neg)
            labels = tf.where(has_label, labels, 0.5 * tf.ones_like(labels))
        else:
            raise ValueError('shape not supported: {}'.format(shape))
        return labels, has_label


def normalized_sigmoid_cross_entropy_with_logits(
        targets, logits, weights, pos_weight=1.0, balanced=False, axis=None,
        name='normalized_sigmoid_cross_entropy_with_logits'):
    '''
    Supports broadcasting for `weights` but not `targets` and `logits`.
    '''
    with tf.name_scope(name) as scope:
        weights = tf.to_float(weights)
        sum_p = tf.reduce_sum(weights * targets, axis=axis, keepdims=True)
        sum_not_p = tf.reduce_sum(weights * (1 - targets), axis=axis, keepdims=True)
        if balanced:
            assert_p = tf.Assert(
                tf.reduce_all(sum_p > 0), [sum_p], summarize=10, name='assert_p')
            assert_not_p = tf.Assert(
                tf.reduce_all(sum_not_p > 0), [sum_not_p], summarize=10, name='assert_not_p')
            with tf.control_dependencies([assert_p, assert_not_p]):
                gamma_base = sum_not_p / sum_p
        else:
            gamma_base = 1
        gamma = pos_weight * gamma_base
        cross_ent = (2 / (gamma + 1)) * tf.nn.weighted_cross_entropy_with_logits(
            targets=targets, logits=logits, pos_weight=gamma)
        # Find constant alpha that normalizes mass to one.
        beta = gamma / (gamma + 1)
        alpha = 1 / (2 * beta * sum_p + 2 * (1 - beta) * sum_not_p)
        return tf.reduce_sum(alpha * weights * cross_ent, axis=axis, keepdims=False)
