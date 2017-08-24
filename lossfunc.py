import tensorflow as tf

import geom
import helpers


def balanced_mean_logistic(labels, logits, axis=None, name='balanced_mean_logistic'):
    '''
    Args:
        labels is an integer in {-1, 0, 1}
        Shape of labels and logits must match.
    '''
    with tf.name_scope(name) as scope:
        is_pos = tf.greater(labels, 0)
        is_neg = tf.less(labels, 0)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(is_pos),
            logits=logits)
        # Normalize labels per image, then take mean over all images.
        num_pos = tf.reduce_sum(tf.to_int32(is_pos), axis=axis, keep_dims=True)
        num_neg = tf.reduce_sum(tf.to_int32(is_neg), axis=axis, keep_dims=True)
        # Compute weighted loss.
        return tf.reduce_sum(
            tf.add(tf.multiply(tf.to_float(is_pos), loss) / tf.to_float(num_pos + 1),
                   tf.multiply(tf.to_float(is_neg), loss) / tf.to_float(num_neg + 1)),
            axis=axis)


def label_map_anchor_l1(anchor_map, rect_gt,
        iou_positive_anchor=0.5,
        image_summaries_collections=None):
    '''
    Args:
        anchor_map -- [h, w, 4]
        rect_gt    -- [b, t, 4]

    Returns:
        [b, t, h, w]
    '''
    with tf.name_scope('positive_anchors'):
        # Find anchor with greatest IOU.
        # anchor_iou is [b, t, h, w]
        anchor_iou = geom.rect_iou(
            anchor_map,
            tf.expand_dims(tf.expand_dims(rect_gt, -2), -2))
        # Find maximum IOU over spatial dimensions.
        max_anchor_iou = tf.reduce_max(anchor_iou, axis=(2, 3), keep_dims=True)

        # Take all rectangles with considerable IOU.
        # TODO: Assert that at least one rectangle has non-zero IOU.
        is_pos = tf.equal(anchor_iou, max_anchor_iou)
        is_pos = tf.logical_or(
            tf.equal(anchor_iou, max_anchor_iou), # Best anchor.
            tf.greater_equal(anchor_iou, iou_positive_anchor)) # Anchor within threshold.
        is_neg = tf.logical_not(is_pos)
        return tf.to_int32(is_pos) - tf.to_int32(is_neg)


def rect_map_l1(is_pos, rect_map, rect_gt, normalize=False):
    '''
    Args:
        is_pos   -- [b, t, h, w]
        rect_map -- [b, t, h, w, 4]
        rect_gt  -- [b, t, 4]

    Returns:
        [b, t]
    '''
    with tf.name_scope('rect_map_l1'):
        loss_raw = _l1(rect_map,
                       tf.expand_dims(tf.expand_dims(rect_gt, -2), -2),
                       normalize=normalize)
        num_pos = tf.reduce_sum(tf.to_int32(is_pos), axis=(2, 3), keep_dims=True)
        return tf.reduce_sum(
            tf.multiply(tf.to_float(is_pos), loss_raw) / tf.to_float(num_pos + 1),
            axis=(2, 3))


def make_label_map_dist(mask, rect, dim, radius_pos, radius_neg):
    '''
    Args:
        mask -- [b, t]
        rect -- Ground truth rectangle. [b, t, 4]
        dim -- Size of score map. Tensor of length 2.

    Returns:
        [b, t, h, w]

    It is assumed that the center of the first and last pixel in the score map
    aligns with the first and last pixel in the (cropped) image.
    '''
    eps = 1e-3
    dim_y, dim_x = dim[0], dim[1]
    coords_x, coords_y = tf.meshgrid(
        tf.to_float(tf.range(dim_x)) / tf.to_float(dim_x - 1),
        tf.to_float(tf.range(dim_y)) / tf.to_float(dim_y - 1))
    coords = tf.stack([coords_x, coords_y], axis=-1)
    # Use same rectangle for all spatial positions.
    # rect [n, t, 4] -> [n, t, 1, 1, 4]
    rect = tf.expand_dims(tf.expand_dims(rect, 2), 2)
    rect_min, rect_max = geom.rect_min_max(rect)
    rect_center = 0.5 * (rect_min + rect_max)
    rect_size = tf.maximum(rect_max - rect_min, 0) + eps
    rect_diam = tf.exp(tf.reduce_mean(tf.log(rect_size), axis=-1))
    dist = tf.norm(coords - rect_center, axis=-1)
    rel_dist = dist / rect_diam
    # Ensure that at least the closest element is a positive!
    # (Only if it is not too far in *absolute* distance.)
    min_dist = tf.reduce_min(dist, axis=(-2, -1), keep_dims=True)
    # pos = rel_dist <= radius_pos
    pos = tf.logical_or(rel_dist <= radius_pos,
                        tf.logical_and(dist <= min_dist, min_dist <= 0.1))
    neg = tf.logical_and(tf.logical_not(pos), rel_dist > radius_neg)
    # Exclude elements that are invalid.
    mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
    pos = tf.logical_and(pos, mask)
    neg = tf.logical_and(neg, mask)
    label = tf.to_int32(pos) - tf.to_int32(neg)
    return label


def _l1(rect_pred, rect_gt, normalize=False):
    if normalize:
        return _l1_norm(rect_pred, rect_gt)
    else:
        return _l1_unnorm(rect_pred, rect_gt)


def _l1_unnorm(rect_pred, rect_gt):
    return tf.reduce_mean(tf.abs(rect_pred - rect_gt), axis=-1)


def _l1_norm(rect_pred, rect_gt):
    eps = 0.05
    err = _l1_unnorm(rect_pred, rect_gt)
    gt_min, gt_max = geom.rect_min_max(rect_gt)
    gt_size = tf.abs(gt_max - gt_min)
    return err / (tf.reduce_mean(gt_size, axis=-1) + eps)
