import tensorflow as tf

from helpers import merge_dims

EPSILON = 1e-3


def crop_rect(rects, window_rect, name='crop_rect'):
    '''Returns each rectangle relative to a window.

    Args:
        rects -- [..., 4]
        window_rect -- [..., 4]
    '''
    with tf.name_scope(name) as scope:
        window_min, window_max = rect_min_max(window_rect)
        window_size = window_max - window_min
        window_size = tf.sign(window_size) * tf.maximum(tf.abs(window_size), EPSILON)
        rects_min, rects_max = rect_min_max(rects)
        out_min = (rects_min - window_min) / window_size
        out_max = (rects_max - window_min) / window_size
        return make_rect(out_min, out_max, name=scope)


def crop_inverse(rect, name='crop_inverse'):
    '''Returns the rectangle that reverses the crop.

    If q = crop_inverse(r), then crop(crop(im, r), q) restores the image.
    That is, cropping is a group operation with an inverse.

    CAUTION: Epsilon means that inverse is not exact?
    '''
    with tf.name_scope(name) as scope:
        # x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
        rect_min, rect_max = rect_min_max(rect)
        # TODO: Support reversed rectangle.
        rect_size = tf.maximum(tf.abs(rect_max - rect_min), EPSILON)
        # x_size = tf.maximum(tf.abs(x_max - x_min), EPSILON)
        # y_size = tf.maximum(tf.abs(y_max - y_min), EPSILON)
        inv_min = -rect_min / rect_size
        # u_min = -x_min / x_size
        # v_min = -y_min / y_size
        inv_max = (1 - rect_min) / rect_size
        # inv_max = inv_min + 1 / rect_size
        # u_max = u_min + 1 / x_size
        # v_max = v_min + 1 / y_size
        return make_rect(inv_min, inv_max, name=scope)


def rect_min_max(rect, name='rect_min_max'):
    with tf.name_scope(name) as scope:
        x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
        min_pt = tf.stack([x_min, y_min], axis=-1)
        max_pt = tf.stack([x_max, y_max], axis=-1)
    return min_pt, max_pt


def make_rect(min_pt, max_pt, name='make_rect'):
    with tf.name_scope(name) as scope:
        x_min, y_min = tf.unstack(min_pt, axis=-1)
        x_max, y_max = tf.unstack(max_pt, axis=-1)
        return tf.stack([x_min, y_min, x_max, y_max], axis=-1, name=scope)


def rect_iou(a_rect, b_rect, name='rect_iou'):
    '''Supports broadcasting.'''
    # Assumes that rectangles are valid (min <= max).
    # This is important if the function is going to be differentiated.
    # Otherwise, there will be no gradient for invalid rectangles.
    # TODO: Add assertion?
    with tf.name_scope(name) as scope:
        intersect_min, intersect_max = rect_min_max(rect_intersect(a_rect, b_rect))
        intersect_area = tf.reduce_prod(tf.maximum(0.0, intersect_max - intersect_min), axis=-1)
        a_min, a_max = rect_min_max(a_rect)
        b_min, b_max = rect_min_max(b_rect)
        a_area = tf.reduce_prod(tf.maximum(0.0, a_max - a_min), axis=-1)
        b_area = tf.reduce_prod(tf.maximum(0.0, b_max - b_min), axis=-1)
        union_area = a_area + b_area - intersect_area
        return intersect_area / tf.maximum(union_area, EPSILON)


def rect_intersect(a_rect, b_rect, name='rect_intersect'):
    # Assumes that rectangles are valid (min <= max).
    with tf.name_scope(name) as scope:
        a_min, a_max = rect_min_max(a_rect)
        b_min, b_max = rect_min_max(b_rect)
        intersect_min = tf.maximum(a_min, b_min)
        intersect_max = tf.minimum(a_max, b_max)
        return make_rect(intersect_min, intersect_max)


def rect_to_tf_box(rect, name='rect_to_tf_box'):
    with tf.name_scope(name) as scope:
        x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
        return tf.stack([y_min, x_min, y_max, x_max], axis=-1, name=scope)


def rect_translate(rect, delta, name='rect_translate'):
    with tf.name_scope(name) as scope:
        min_pt, max_pt = rect_min_max(rect)
        return make_rect(min_pt + delta, max_pt + delta)


def rect_translate_random(rect, limit, name='rect_translate_random'):
    with tf.name_scope(name) as scope:
        min_pt, max_pt = rect_min_max(rect)
        rect_size = max_pt - min_pt
        diam = tf.reduce_mean(rect_size, axis=-1)
        delta = diam * tf.random_uniform(shape=[tf.shape(rect)[0]], minval=-limit, maxval=limit)
        return rect_translate(rect, delta)


def rect_mul(rect, scale, name='rect_mul'):
    with tf.name_scope(name) as scope:
        min_pt, max_pt = rect_min_max(rect)
        return make_rect(min_pt * scale, max_pt * scale)


def warp_anchor(anchor, warp, name='warp_anchor'):
    # Supports broadcasting.
    with tf.name_scope(name) as scope:
        warp_offset_x, warp_offset_y, warp_scale_x, warp_scale_y = tf.unstack(warp, axis=-1)
        warp_offset = tf.stack([warp_offset_x, warp_offset_y], axis=-1)
        warp_scale = tf.stack([warp_scale_x, warp_scale_y], axis=-1)

        anchor_min, anchor_max = rect_min_max(anchor)
        anchor_size = anchor_max - anchor_min
        anchor_center = 0.5 * (anchor_min + anchor_max) # Expect zero.

        rect_size = tf.exp(warp_scale) * anchor_size
        rect_center = anchor_center + warp_offset
        return make_rect(rect_center - 0.5*rect_size, rect_center + 0.5*rect_size)


def unit_rect(dtype=tf.float32):
    min_pt = tf.constant([0, 0], dtype=dtype)
    max_pt = tf.constant([1, 1], dtype=dtype)
    return make_rect(min_pt, max_pt)
