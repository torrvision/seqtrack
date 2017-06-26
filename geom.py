import tensorflow as tf

def crop_example(example, window_rect, first_only=False):
    '''
    Args:
        example -- Dictionary.
            example['x0'] -- [n, h, w, c]
            example['y0'] -- [n, 4]
            example['x'] -- [n, t, h, w, c]
            example['y'] -- [n, t, 4] (optional)
            All other fields are kept intact.
        window_rect -- [n, 4]
    '''
    out = dict(example)

    xs = tf.expand_dims(example['x0'], 1)
    if not first_only:
        # Re-combine into a single sequence.
        xs = tf.concat([xs, example['x']], axis=1)
    im_size = xs.shape.as_list()[2:4] # Require static for now.
    xs = crop_image_sequence(xs, window_rect, crop_size=im_size)
    out['x0'] = xs[:, 0]
    if not first_only:
        out['x'] = xs[:, 1:]

    ys = tf.expand_dims(example['y0'], 1)
    if not first_only and 'y' in example:
        # Re-combine into a single sequence.
        ys = tf.concat([ys, example['y'], axis=1)
    ys = crop_rect_sequence(ys, window_rect)
    out['y0'] = ys[:, 0]
    if not first_only and 'y' in example:
        out['y'] = ys[:, 1:]
    
    # TODO: Modify y_valid if object is out of window?
    return out


def crop_pred(pred, window_rect):
    '''
    Args:
        pred -- Dictionary.
            pred['y'] -- [n, t, 4]
            pred['hmap'] -- [n, t, h, w, 2]
            All other fields are kept intact.
        window_rect -- [n, 4]
    '''
    out = dict(pred)
    out['y'] = crop_rect_sequence(pred['y'], window_rect)
    if 'hmap' in pred:
        out['hmap'] = crop_image_sequence(pred['hmap'], window_rect)


def object_centric_window(obj_rect, relative_size=4.0):
    eps = 0.01
    obj_min, obj_max = rect_min_max(obj_rect)
    obj_size = tf.maximum(0.0, obj_max - obj_min)
    center = 0.5 * (obj_min + obj_max)
    obj_diam = tf.exp(tf.reduce_mean(tf.log(obj_size + eps), axis=-1))
    context_diam = relative_size * obj_diam
    window_min = center - 0.5*tf.expand_dims(context_diam, -1)
    window_max = center + 0.5*tf.expand_dims(context_diam, -1)
    return make_rect(window_min, window_max)

def crop_rects(rects, window_rect):
    '''Returns each rectangle relative to a window.
    
    Args:
        rects -- [..., 4]
        window_rect -- [..., 4]
    '''
    eps = 0.01
    window_min, window_max = rect_min_max(window_rect)
    window_size = window_max - window_min
    window_size = tf.sign(window_size) * (tf.abs(window_size) + eps)
    rects_min, rects_max = rect_min_max(rects)
    out_min = (rects_min - window_min) / window_size
    out_max = (rects_max - window_min) / window_size
    return make_rect(out_min, out_max)

def crop_rect_sequence(rects, window_rect):
    '''Returns each rectangle relative to a window.
    
    Args:
        rects -- [n, t, 4]
        window_rect -- [n, 4]
    '''
    # TODO: Support window_rect of size [n, t, 4] as well.
    # Same rectangle in every image.
    sequence_len = tf.shape(rects)[1]
    window_rect = tf.expand_dims(window_rect, 1)
    window_rect = tf.tile(window_rect, [1, sequence_len, 1])
    # Now dimensions match.
    return crop_rects(rects, window_rect)

def crop_image_sequence(ims, window_rect, crop_size, pad_value=None):
    '''
    Extracts 

    Args:
        ims -- [n, t, h, w, c]
        window_rect -- [n, 4]
    '''
    # Same rectangle in every image.
    sequence_len = tf.shape(ims)[1]
    window_rect = tf.expand_dims(window_rect, 1)
    window_rect = tf.tile(window_rect, [1, sequence_len, 1])
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
    crop_ims = unmerge(crop_ims, 0)
    return crop_ims

def crop_inverse(rect):
    '''Returns the rectangle that reverses the crop.

    If q = crop_inverse(r), then crop(crop(im, r), q) restores the image.
    That is, cropping is a group operation with an inverse.

    CAUTION: Epsilon means that inverse is not exact?
    '''
    eps = 0.01
    # x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    rect_min, rect_max = rect_min_max(rect)
    # TODO: Support reversed rectangle.
    rect_size = tf.abs(rect_max - rect_min) + eps
    # x_size = tf.abs(x_max - x_min) + eps
    # y_size = tf.abs(y_max - y_min) + eps
    inv_min = -rect_min / rect_size
    # u_min = -x_min / x_size
    # v_min = -y_min / y_size
    inv_max = (1 - rect_min) / rect_size
    # inv_max = inv_min + 1 / rect_size
    # u_max = u_min + 1 / x_size
    # v_max = v_min + 1 / y_size
    return make_rect(inv_min, inv_max)

def rect_min_max(rect):
    x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    min_pt = tf.stack([x_min, y_min], axis=-1)
    max_pt = tf.stack([x_max, y_max], axis=-1)
    return min_pt, max_pt

def make_rect(min_pt, max_pt):
    x_min, y_min = tf.unstack(min_pt, axis=-1)
    x_max, y_max = tf.unstack(max_pt, axis=-1)
    return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

def rect_to_tf_box(rect):
    x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)
