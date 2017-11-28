import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from seqtrack import cnnutil
from seqtrack import geom

from seqtrack.cnnutil import ReceptiveField, IntRect
from seqtrack.helpers import merge_dims, grow_rect, modify_aspect_ratio, diag_xcorr
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers


def crop(im, rect, im_size, name='crop'):
    with tf.name_scope(name) as scope:
        batch_len = tf.shape(im)[0]
        return tf.image.crop_and_resize(im, geom.rect_to_tf_box(rect),
                                        box_ind=tf.range(batch_len),
                                        crop_size=n_positive_integers(2, im_size),
                                        extrapolation_value=128)


def crop_pyr(im, rect, im_size, scales, name='crop_pyr'):
    '''
    Args:
        im: [b, h, w, 3]
        rect: [b, 4]
        scales: [s]

    Returns:
        [b, s, h, w, 3]
    '''
    with tf.name_scope(name) as scope:
        # [b, s, 4]
        rects = grow_rect(tf.expand_dims(scales, -1), tf.expand_dims(rect, -2))
        # Extract multiple rectangles from each image.
        batch_len = tf.shape(im)[0]
        num_scales, = tf.unstack(tf.shape(scales))
        box_ind = tf.tile(tf.expand_dims(tf.range(batch_len), 1), [1, num_scales])
        # [b, s, ...] -> [b*s, ...]
        rects, restore = merge_dims(rects, 0, 2)
        box_ind, _ = merge_dims(box_ind, 0, 2)
        crop = tf.image.crop_and_resize(im, geom.rect_to_tf_box(rects),
                                        box_ind=box_ind,
                                        crop_size=n_positive_integers(2, im_size),
                                        extrapolation_value=128)
        # [b*s, ...] -> [b, s, ...]
        crop = restore(crop, 0)
        return crop, rects


def scale_range(num, step, name='scale_range'):
    '''Creates a geometric progression with 1 at the center.'''
    with tf.name_scope(name) as scope:
        assert isinstance(num, tf.Tensor)
        with tf.control_dependencies(
                [tf.assert_equal(num % 2, 1, message='number of scales must be odd')]):
            half = (num - 1) / 2
        log_step = tf.abs(tf.log(step))
        return tf.exp(log_step * tf.to_float(tf.range(-half, half+1)))


def conv2d_rf(inputs, input_rfs, num_outputs, kernel_size, stride=1, padding='SAME', **kwargs):
    '''Wraps slim.conv2d to include receptive field calculation.
    
    input_rfs['var_name'] is the receptive field of input w.r.t. var_name.
    output_rfs['var_name'] is the receptive field of output w.r.t. var_name.
    '''
    if input_rfs is None:
        input_rfs = {}
    assert len(inputs.shape) == 4 # otherwise slim.conv2d does higher-dim convolution
    outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding, **kwargs)
    rel_rf = _filter_rf(kernel_size, stride, padding)
    output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def max_pool2d_rf(inputs, input_rfs, kernel_size, stride=2, padding='VALID', **kwargs):
    '''Wraps slim.max_pool2d to include receptive field calculation.'''
    if input_rfs is None:
        input_rfs = {}
    outputs = slim.max_pool2d(inputs, kernel_size, stride, padding, **kwargs)
    rel_rf = _filter_rf(kernel_size, stride, padding)
    output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def diag_xcorr_rf(input, filter, input_rfs, stride=1, padding='VALID', name='diag_xcorr'):
    '''
    Args:
        input: [b, ..., h, w, c]
        filter: [b, fh, fw, c]
        stride: Either an integer or length 2 (as for slim operations).
    '''
    stride = n_positive_integers(2, stride)
    kernel_size = filter.shape.as_list()[-3:-1]
    assert all(kernel_size) # Must not be None or 0.
    rel_rf = _filter_rf(kernel_size, stride, padding)
    output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}

    nhwc_strides = [1, stride[0], stride[1], 1]
    output = diag_xcorr(input, filter, strides=nhwc_strides, padding=padding, name=name)
    return output, output_rfs


# def xcorr_rf(input, filter, input_rfs, padding='VALID'):
#     '''Wraps tf.nn.conv2d to include receptive field calculation.
# 
#     Args:
#         input: [b, h, w, c]
#         filter: [b, h, w, c]
#     '''
#     # TODO: Support depth-wise/diagonal convolution.
#     if input_rfs is None:
#         input_rfs = {}
#     kernel_size = filter.shape.as_list()[-4:-2]
#     rel_rf = _filter_rf(kernel_size, stride, padding)
#     output = tf.nn.conv2d(input, filter, padding=padding)
#     output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
#     return output, output_rfs


def _filter_rf(kernel_size, stride, padding):
    '''Computes the receptive field of a filter.'''
    kernel_size = np.array(n_positive_integers(2, kernel_size))
    stride = np.array(n_positive_integers(2, stride))
    # Get relative receptive field.
    if padding == 'SAME':
        assert np.all(kernel_size % 2 == 1)
        half = (kernel_size - 1) / 2
        rect = IntRect(-half, half+1)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    elif padding == 'VALID':
        rect = IntRect(np.zeros_like(kernel_size), kernel_size)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    else:
        raise ValueError('invalid padding: {}'.format(padding))
    return rel_rf


def coerce_aspect(target, im_aspect, aspect_method='stretch', name='coerce_aspect'):
    with tf.name_scope(name) as scope:
        # Modify aspect ratio of target.
        # Necessary to account for aspect ratio of reference frame.
        if aspect_method != 'stretch':
            # Transform (1, 1) to (im_aspect, 1) so that width/height = im_aspect.
            # stretch = tf.stack([tf.pow(im_aspect, 0.5), tf.pow(im_aspect, -0.5)], axis=-1)
            stretch = tf.stack([im_aspect, tf.ones_like(im_aspect)], axis=-1)
            target = geom.rect_mul(target, stretch)
            target = modify_aspect_ratio(target, aspect_method)
            target = geom.rect_mul(target, 1./stretch)
        return target


def find_center_in_scoremap(scoremap, threshold=0.95):
    assert len(scoremap.shape.as_list()) == 4

    max_val = tf.reduce_max(scoremap, axis=(1,2), keep_dims=True)
    with tf.control_dependencies([tf.assert_greater_equal(scoremap, 0.0)]):
        max_loc = tf.greater_equal(scoremap, max_val*threshold) # values over 95% of max.

    spatial_dim = scoremap.shape.as_list()[1:3]
    assert all(spatial_dim) # Spatial dimension must be static.
    # Compute center of each pixel in [0, 1] in search area.
    dim_y, dim_x = spatial_dim[0], spatial_dim[1]
    centers_x, centers_y = tf.meshgrid(
        tf.to_float(tf.range(dim_x)) / tf.to_float(dim_x - 1),
        tf.to_float(tf.range(dim_y)) / tf.to_float(dim_y - 1))
    centers = tf.stack([centers_x, centers_y], axis=-1)
    max_loc = tf.to_float(max_loc)
    center = tf.divide(
        tf.reduce_sum(centers * max_loc, axis=(1, 2)),
        tf.reduce_sum(max_loc, axis=(1, 2)))

    EPSILON = 1e-4
    with tf.control_dependencies([tf.assert_greater_equal(center, -EPSILON),
                                  tf.assert_less_equal(center, 1.0+EPSILON)]):
        center = tf.identity(center)
    return center


def displacement_from_center(im_size, name='displacement_grid'):
    '''
    Args:
        im_size: (height, width)

    Returns:
        Tensor grid of size [height, width, 2] as (x, y).
    '''
    with tf.name_scope(name) as scope:
        # Get the translation from the center.
        im_size = np.asarray(im_size)
        assert all(im_size % 2 == 1)
        center = (im_size - 1) / 2
        size_y, size_x = im_size
        center_y, center_x = center
        grid_y = tf.range(size_y) - center_y
        grid_x = tf.range(size_x) - center_x
        grid = tf.stack((tf.tile(tf.expand_dims(grid_x, 0), [size_y, 1]),
                         tf.tile(tf.expand_dims(grid_y, 1), [1, size_x])), axis=-1)
        return grid
