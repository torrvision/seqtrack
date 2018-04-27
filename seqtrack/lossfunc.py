import tensorflow as tf
import tensorflow.contrib.slim as slim

from seqtrack import geom
from seqtrack.helpers import most_static_shape, expand_dims_n


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
            labels = tf.exp(-0.5 * sqr_dist / tf.square(sigma))
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


def make_balanced_weights(labels, has_label, axis=None, name='make_balanced_weights'):
    '''
    Caution: Enforces minimum mass of 1 per label.
    If there is almost zero weight for a label, this will make it effectively zero.
    '''
    with tf.name_scope(name) as scope:
        mass_pos = tf.where(has_label, labels, tf.zeros_like(labels))
        mass_neg = tf.where(has_label, 1. - labels, tf.zeros_like(labels))
        total_mass_pos = tf.maximum(1., tf.reduce_sum(mass_pos, axis=axis, keep_dims=True))
        total_mass_neg = tf.maximum(1., tf.reduce_sum(mass_neg, axis=axis, keep_dims=True))
        weights_pos = mass_pos / total_mass_pos
        weights_neg = mass_neg / total_mass_neg
        weights = 0.5 * weights_pos + 0.5 * weights_neg
        return weights


def make_uniform_weights(has_label, axis=None, name='make_uniform_weights'):
    with tf.name_scope(name) as scope:
        mass = tf.to_float(has_label)
        total_mass = tf.reduce_sum(mass, axis=axis, keep_dims=True)
        weights = mass / total_mass
        weights = tf.where(has_label, weights, tf.zeros_like(weights))
        return weights
