import tensorflow as tf

from helpers import merge_dims

EPSILON = 1e-3


def crop_example(example, window, crop_size, pad_value=None, name='crop_example'):
    '''
    Args:
        example -- Dictionary.
            example['x'] -- [n, t, h, w, c]
            example['y'] -- [n, t, 4] (optional)
        window_rect -- [n, t, 4] or [n, 4]
    '''
    with tf.name_scope(name):
        out = {}
        out['x'] = crop_image_sequence(example['x'], window,
            crop_size=crop_size,
            pad_value=pad_value,
        )
        if 'y' in example:
            out['y'] = crop_rect(example['y'], window)
        # TODO: Modify y_valid if object is out of window?
        if 'y_is_valid' in example:
            out['y_is_valid'] = example['y_is_valid']
    return out


def crop_example_frame(example, window_rect, crop_size, pad_value=None,
                       name='crop_example_frame'):
    '''
    Args:
        example -- Dictionary.
            example['x'] -- [n, h, w, c]
            example['y'] -- [n, 4] (optional)
        window_rect -- [n, 4]
    '''
    with tf.name_scope(name):
        out = {}
        out['x'] = crop_image(example['x'], window_rect, crop_size=crop_size, pad_value=pad_value)
        if 'y' in example:
            out['y'] = crop_rect(example['y'], window_rect)
        # TODO: Modify y_valid if object is out of window?
        if 'y_is_valid' in example:
            out['y_is_valid'] = example['y_is_valid']
    return out


# def crop_prediction(pred, window_rect, crop_size):
#     '''
#     Args:
#         pred -- Dictionary.
#             pred['y'] -- [n, t, 4]
#             pred['hmap_softmax'] -- [n, t, h, w, 2]
#             All other fields are kept intact.
#         window_rect -- [n, 4]
#     '''
#     out = dict(pred)
#     out['y'] = crop_rect_sequence(pred['y'], window_rect)
#     if 'hmap_softmax' in pred:
#         out['hmap_softmax'] = crop_image_sequence(pred['hmap_softmax'], window_rect,
#             crop_size=crop_size,
#         )
#     return out


def object_centric_window(obj_rect, relative_size=4.0, name='object_centric_window'):
    with tf.name_scope(name) as scope:
        obj_min, obj_max = rect_min_max(obj_rect)
        obj_size = tf.maximum(0.0, obj_max - obj_min)
        center = 0.5 * (obj_min + obj_max)
        obj_diam = tf.exp(tf.reduce_mean(tf.log(tf.maximum(obj_size, EPSILON)), axis=-1))
        window_diam = relative_size * obj_diam
        window_min = center - 0.5*tf.expand_dims(window_diam, -1)
        window_max = center + 0.5*tf.expand_dims(window_diam, -1)
        return make_rect(window_min, window_max, name=scope)


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


# def crop_rect_sequence(rects, window_rect):
#     '''Returns each rectangle relative to a window.
#     
#     Args:
#         rects -- [n, t, 4]
#         window_rect -- [n, t, 4] or [n, 4]
#     '''
#     assert len(rects.shape) == 3
#     if len(window_rect.shape) == 2:
#         # Same rectangle in every image.
#         sequence_len = tf.shape(rects)[1]
#         window_rect = tf.expand_dims(window_rect, 1)
#         window_rect = tf.tile(window_rect, [1, sequence_len, 1])
#     assert len(window_rect.shape) == 3
#     # Now dimensions match.
#     return crop_rect(rects, window_rect)


def crop_image(ims, window_rect, crop_size, pad_value=None, name='crop_image'):
    '''
    Crops a window from each image in a batch.

    Args:
        ims -- [n, h, w, c]
        window_rect -- [n, 4]
    '''
    with tf.name_scope(name) as scope:
        assert len(ims.shape) == 4
        assert len(window_rect.shape) == 2
        boxes = rect_to_tf_box(window_rect)
        n = tf.shape(ims)[0]
        return tf.image.crop_and_resize(ims, boxes, box_ind=tf.range(n),
            crop_size=crop_size,
            method='bilinear',
            extrapolation_value=pad_value,
            name=scope)


def crop_image_sequence(ims, window_rect, crop_size, pad_value=None,
                        name='crop_image_sequence'):
    '''
    Extracts 

    Args:
        ims -- [n, t, h, w, c]
        window_rect -- [n, t, 4] or [n, 4]
    '''
    with tf.name_scope(name) as scope:
        assert len(ims.shape) == 5
        if len(window_rect.shape) == 2:
            # Same rectangle in every image.
            sequence_len = tf.shape(ims)[1]
            window_rect = tf.expand_dims(window_rect, 1)
            window_rect = tf.tile(window_rect, [1, sequence_len, 1])
        assert len(window_rect.shape) == 3
        # Flatten.
        ims, unmerge = merge_dims(ims, 0, 2)
        window_rect, _ = merge_dims(window_rect, 0, 2)
        boxes = rect_to_tf_box(window_rect)
        num_images = tf.shape(ims)[0]
        crop_ims = tf.image.crop_and_resize(ims, boxes, box_ind=tf.range(num_images),
            crop_size=crop_size,
            method='bilinear',
            extrapolation_value=pad_value)
        # Un-flatten.
        return unmerge(crop_ims, 0, name=scope)


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
        return make_rect(inv_min, inv_max, name='scope')


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
