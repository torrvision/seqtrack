import tensorflow as tf

import geom


def score_logistic(mask, score_pred, rect_gt, radius_pos=0.2, radius_neg=0.5,
              image_summaries_collections=None):
    '''
    Args:
        mask       -- [b, t]
        score_pred -- [b, t, h, w, 1]
        rect_gt    -- [b, t, 4]
    '''
    # TODO: Use y_is_valid!
    # Compare sliding-window score map to ground truth.
    # score_pred
    # rect_gt
    with tf.name_scope('score_logistic'):
        score_dim = score_pred.shape.as_list()[-3:-1][::-1]
        label_map = _make_label_map(mask, rect_gt, score_dim, radius_pos, radius_neg)
        loss_raw = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(tf.equal(label_map, 1)),
            logits=score_pred)

        # Normalize labels per image, then take mean over all images.
        is_pos = tf.equal(label_map, 1)
        is_neg = tf.equal(label_map, -1)
        num_pos = tf.reduce_sum(tf.to_int32(is_pos), axis=(-2, -1), keep_dims=True)
        num_neg = tf.reduce_sum(tf.to_int32(is_neg), axis=(-2, -1), keep_dims=True)
        # Compute weighted loss per image.
        loss_balanced = tf.reduce_sum(
            tf.add(tf.multiply(tf.to_float(is_pos), loss_raw) / tf.to_float(num_pos + 1),
                   tf.multiply(tf.to_float(is_neg), loss_raw) / tf.to_float(num_neg + 1)),
            axis=(-2, -1))

        with tf.name_scope('summary'):
            tf.summary.image('label_map',
                tf.image.convert_image_dtype(
                    0.5 * (1.0 + tf.to_float(label_map[0])),
                    tf.uint8, saturate=True),
                collections=image_summaries_collections)
            tf.summary.image('score',
                tf.image.convert_image_dtype(
                    tf.sigmoid(score_pred[0]),
                    tf.uint8, saturate=True),
                collections=image_summaries_collections)

        # Take mean over images, excluding images without labels.
        # NOTE: Batch must contain at least 1 valid example!
        return tf.reduce_mean(tf.boolean_mask(loss_balanced, mask))


def _make_label_map(mask, rect, dim, radius_pos, radius_neg):
    '''
    Args:
        mask -- [b, t]
        rect -- Ground truth rectangle. [b, t, 4]
        dim -- Size of score map. Tensor of length 2.

    Returns:
        [b, t, h, w, 1]

    It is assumed that the center of the first and last pixel in the score map
    aligns with the first and last pixel in the (cropped) image.
    '''
    eps = 1e-3
    u, v = tf.meshgrid(tf.to_float(tf.range(dim[0])) / tf.to_float(dim[0] - 1),
                       tf.to_float(tf.range(dim[1])) / tf.to_float(dim[1] - 1),
                       indexing='xy')
    coords = tf.stack([u, v], axis=-1)
    # Use same rectangle for all spatial positions.
    # rect [n, t, 4] -> [n, t, 1, 1, 4]
    rect = tf.expand_dims(tf.expand_dims(rect, 2), 2)
    rect_min, rect_max = geom.rect_min_max(rect)
    rect_center = 0.5 * (rect_min + rect_max)
    rect_size = tf.maximum(rect_max - rect_min, 0) + eps
    rect_diam = tf.exp(tf.reduce_mean(tf.log(rect_size), axis=-1, keep_dims=True))
    dist = tf.norm(coords - rect_center, axis=-1, keep_dims=True)
    rel_dist = dist / rect_diam
    # Ensure that at least the closest element is a positive!
    # (Only if it is not too far in *absolute* distance.)
    min_dist = tf.reduce_min(dist, axis=(-2, -1), keep_dims=True)
    pos = tf.logical_or(rel_dist <= radius_pos,
                        tf.logical_and(dist <= min_dist, min_dist <= 0.1))
    neg = tf.logical_and(tf.logical_not(pos),
                         rel_dist > radius_neg)
    # Exclude elements that are invalid.
    mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
    pos = tf.logical_and(pos, mask)
    neg = tf.logical_and(neg, mask)
    label = tf.cast(pos, tf.int32) - tf.cast(neg, tf.int32)
    return label
