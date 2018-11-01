from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import functools
import matplotlib.cm

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom
from seqtrack import lossfunc
from seqtrack import receptive_field

from seqtrack.helpers import merge_dims
from seqtrack.helpers import modify_aspect_ratio
# from seqtrack.helpers import diag_xcorr
from seqtrack.helpers import expand_dims_n
from seqtrack.helpers import weighted_mean
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers


def crop(im, rect, im_size, pad_value=0, feather=False, feather_margin=0.05, name='crop'):
    '''
    Args:
        im: [b, h, w, c]
        rect: [b, 4] or [4]
        im_size: (height, width)
        pad_value: Either scalar constant or
            tf.Tensor that is broadcast-compatible with image.
    '''
    with tf.name_scope(name) as scope:
        if isinstance(pad_value, tf.Tensor):
            # TODO: This operation seems slow!
            im -= pad_value
            im = crop(im, rect, im_size, pad_value=0,
                      feather=feather, feather_margin=feather_margin, name=name)
            im += pad_value
            return im

        if feather:
            im = feather_image(im, margin=feather_margin, background_value=pad_value)
        batch_len = tf.shape(im)[0]
        # Make rect broadcast.
        rect = rect + tf.zeros([batch_len, 1], tf.float32)
        return tf.image.crop_and_resize(im, geom.rect_to_tf_box(rect),
                                        box_ind=tf.range(batch_len),
                                        crop_size=n_positive_integers(2, im_size),
                                        extrapolation_value=pad_value)


def crop_pyr(im, rect, im_size, scales, pad_value=0, feather=False, feather_margin=0.05, name='crop_pyr'):
    '''
    Args:
        im: [b, h, w, 3]
        rect: [b, 4]
        im_size: (height, width)
        scales: [s]
        pad_value: Either scalar constant or
            tf.Tensor that is broadcast-compatible with image.

    Returns:
        [b, s, h, w, 3]
    '''
    with tf.name_scope(name) as scope:
        if isinstance(pad_value, tf.Tensor):
            # TODO: This operation seems slow!
            im -= pad_value
            crop_ims, rects = crop_pyr(im, rect, im_size, scales, pad_value=0,
                                       feather=feather, feather_margin=feather_margin, name=name)
            crop_ims += tf.expand_dims(pad_value, 1)
            return crop_ims, rects

        if feather:
            im = feather_image(im, margin=feather_margin, background_value=pad_value)
        # [b, s, 4]
        rects = geom.grow_rect(tf.expand_dims(scales, -1), tf.expand_dims(rect, -2))
        # Extract multiple rectangles from each image.
        batch_len = tf.shape(im)[0]
        num_scales, = tf.unstack(tf.shape(scales))
        box_ind = tf.tile(tf.expand_dims(tf.range(batch_len), 1), [1, num_scales])
        # [b, s, ...] -> [b*s, ...]
        rects, restore = merge_dims(rects, 0, 2)
        box_ind, _ = merge_dims(box_ind, 0, 2)
        crop_ims = tf.image.crop_and_resize(im, geom.rect_to_tf_box(rects),
                                            box_ind=box_ind,
                                            crop_size=n_positive_integers(2, im_size),
                                            extrapolation_value=pad_value)
        # [b*s, ...] -> [b, s, ...]
        crop_ims = restore(crop_ims, 0)
        return crop_ims, rects


def feather_image(im, margin, background_value=0, name='feather_image'):
    with tf.name_scope(name) as scope:
        im_size = im.shape.as_list()[-3:-1]
        assert all(im_size)
        mask = feather_mask(im_size, margin)
        if background_value == 0:
            im *= mask
        else:
            im = mask * im + (1 - mask) * background_value
        return im


def feather_mask(im_size, margin, name='feather_mask'):
    '''
    Args:
        im_size: (height, width)
        pad_value: Either scalar constant or
            tf.Tensor that is broadcast-compatible with image.
    '''
    # TODO: Should account for aspect ratio.
    with tf.name_scope(name) as scope:
        grid = lossfunc.make_grid_centers(im_size)
        # Distance from interior on either side.
        lower, upper = margin, 1 - margin
        below_lower = tf.clip_by_value((lower - grid) / margin, 0., 1.)
        above_upper = tf.clip_by_value((grid - upper) / margin, 0., 1.)
        delta = tf.maximum(below_lower, above_upper)
        dist = tf.clip_by_value(tf.norm(delta, axis=-1, keep_dims=True), 0., 1.)
        mask = 1. - dist
        return mask


def scale_range(num, step, name='scale_range'):
    '''Creates a geometric progression with 1 at the center.'''
    with tf.name_scope(name) as scope:
        assert isinstance(num, tf.Tensor)
        with tf.control_dependencies(
                [tf.assert_equal(num % 2, 1, message='number of scales must be odd')]):
            half = (num - 1) / 2
        log_step = tf.abs(tf.log(step))
        return tf.exp(log_step * tf.to_float(tf.range(-half, half + 1)))


# def conv2d_rf(inputs, input_rfs, num_outputs, kernel_size, stride=1, padding='SAME', **kwargs):
#     '''Wraps slim.conv2d to include receptive field calculation.
# 
#     input_rfs['var_name'] is the receptive field of input w.r.t. var_name.
#     output_rfs['var_name'] is the receptive field of output w.r.t. var_name.
#     '''
#     if input_rfs is None:
#         input_rfs = {}
#     if inputs is not None:
#         assert len(inputs.shape) == 4  # otherwise slim.conv2d does higher-dim convolution
#         outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding, **kwargs)
#     else:
#         outputs = None
#     rel_rf = _filter_rf(kernel_size, stride, padding)
#     output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
#     return outputs, output_rfs


# def max_pool2d_rf(inputs, input_rfs, kernel_size, stride=2, padding='VALID', **kwargs):
#     '''Wraps slim.max_pool2d to include receptive field calculation.'''
#     if input_rfs is None:
#         input_rfs = {}
#     if inputs is not None:
#         outputs = slim.max_pool2d(inputs, kernel_size, stride, padding, **kwargs)
#     else:
#         outputs = None
#     rel_rf = _filter_rf(kernel_size, stride, padding)
#     output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
#     return outputs, output_rfs


# def diag_xcorr_rf(input, filter, input_rfs, stride=1, padding='VALID', name='diag_xcorr'):
#     '''
#     Args:
#         input: [b, ..., h, w, c]
#         filter: [b, fh, fw, c]
#         stride: Either an integer or length 2 (as for slim operations).
#     '''
#     stride = n_positive_integers(2, stride)
#     kernel_size = filter.shape.as_list()[-3:-1]
#     assert all(kernel_size)  # Must not be None or 0.
#     rel_rf = _filter_rf(kernel_size, stride, padding)
#     output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
# 
#     nhwc_strides = [1, stride[0], stride[1], 1]
#     output = diag_xcorr(input, filter, strides=nhwc_strides, padding=padding, name=name)
#     return output, output_rfs


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


# def _filter_rf(kernel_size, stride, padding):
#     '''Computes the receptive field of a filter.'''
#     kernel_size = np.array(n_positive_integers(2, kernel_size))
#     stride = np.array(n_positive_integers(2, stride))
#     # Get relative receptive field.
#     if padding == 'SAME':
#         assert np.all(kernel_size % 2 == 1)
#         half = (kernel_size - 1) / 2
#         rect = IntRect(-half, half + 1)
#         rel_rf = ReceptiveField(rect=rect, stride=stride)
#     elif padding == 'VALID':
#         rect = IntRect(np.zeros_like(kernel_size), kernel_size)
#         rel_rf = ReceptiveField(rect=rect, stride=stride)
#     else:
#         raise ValueError('invalid padding: {}'.format(padding))
#     return rel_rf


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
            target = geom.rect_mul(target, 1. / stretch)
        return target


def find_center_in_scoremap(scoremap, threshold=0.95):
    assert len(scoremap.shape.as_list()) == 4

    max_val = tf.reduce_max(scoremap, axis=(1, 2), keep_dims=True)
    with tf.control_dependencies([tf.assert_greater_equal(scoremap, 0.0)]):
        max_loc = tf.greater_equal(scoremap, max_val * threshold)  # values over 95% of max.

    spatial_dim = scoremap.shape.as_list()[1:3]
    assert all(spatial_dim)  # Spatial dimension must be static.
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
                                  tf.assert_less_equal(center, 1.0 + EPSILON)]):
        center = tf.identity(center)
    return center


def is_peak(response, axis=None, eps_rel=0.0, eps_abs=0.0, name='is_peak'):
    '''
    Args:
        response: [b, s, h, w, 1]
    '''
    with tf.name_scope(name) as scope:
        # Find arg max over all scales.
        response = tf.verify_tensor_all_finite(response, 'response is not finite')
        max_val = tf.reduce_max(response, axis=axis, keep_dims=True)
        is_max = tf.logical_or(
            tf.greater_equal(response, max_val - eps_rel*tf.abs(max_val)),
            tf.greater_equal(response, max_val - eps_abs))
        return is_max


def find_peak_pyr(response, scales, eps_rel=0.0, eps_abs=0.0, name='find_peak_pyr'):
    '''
    Args:
        response: [b, s, h, w, 1]
        scales: [s]

    Assumes that response is centered and at same stride as search image.
    '''
    with tf.name_scope(name) as scope:
        response = tf.squeeze(response, axis=-1)
        upsample_size = response.shape.as_list()[-2:]
        assert all(upsample_size)
        upsample_size = np.array(upsample_size)
        # Find arg max over all scales.
        response = tf.verify_tensor_all_finite(response, 'response is not finite')
        max_val = tf.reduce_max(response, axis=(-3, -2, -1), keep_dims=True)
        with tf.control_dependencies([tf.assert_non_negative(response)]):
            # is_max = tf.to_float(response >= (1.0 - eps_rel) * max_val)
            is_max = tf.logical_or(tf.greater_equal(response, max_val - eps_rel * tf.abs(max_val)),
                                   tf.greater_equal(response, max_val - eps_abs))
        is_max = tf.to_float(is_max)

        grid = tf.to_float(displacement_from_center(upsample_size))

        # Grid now has translation from center in search image co-ords.
        # Transform into co-ordinate frame of each scale.
        grid = tf.multiply(grid,                           # [h, w, 2]
                           expand_dims_n(scales, -1, n=3))  # [s, 1, 1, 1]

        translation = weighted_mean(grid,                       # [s, h, w, 2]
                                    tf.expand_dims(is_max, -1),  # [b, s, h, w] -> [b, s, h, w, 1]
                                    axis=(-4, -3, -2))          # [b, s, h, w, 2] -> [b, 2]
        scale = weighted_mean(expand_dims_n(scales, -1, n=2),  # [b, s] -> [b, s, 1, 1]
                              is_max,                         # [b, s, h, w]
                              axis=(-3, -2, -1))              # [b, s, h, w] -> [b]
        return translation, scale


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
        # assert all(im_size % 2 == 1), 'image size not odd: {}'.format(str(im_size))
        if not all(im_size % 2 == 1):
            logger.warning('use center of image with non-odd size %s', str(im_size))
        center = (im_size - 1) / 2
        size_y, size_x = im_size
        center_y, center_x = center
        grid_y = tf.range(size_y) - center_y
        grid_x = tf.range(size_x) - center_x
        grid = tf.stack((tf.tile(tf.expand_dims(grid_x, 0), [size_y, 1]),
                         tf.tile(tf.expand_dims(grid_y, 1), [1, size_x])), axis=-1)
        return grid


def rect_grid(response_size, rf, search_size, rect_size, name='rect_grid'):
    '''Obtains the rectangles for a translation scoremap.

    Args:
        response_size -- Dimension of scoremap. (height, width)
        rf -- Receptive field of scoremap.
        search_size -- Dimension of search image in pixels. (height, width)
        rect_size -- Size of rectangle in normalized co-ords in search image. [b, 2]
    '''
    with tf.name_scope(name) as scope:
        # Assert that receptive fields are centered.
        receptive_field.assert_center_alignment(search_size, response_size, rf)
        # Obtain displacement from center of search image.
        # Not necessary to use receptive field offset because it is centered.
        disp = displacement_from_center(response_size)
        disp = tf.to_float(disp) * rf.stride / search_size
        # Get centers of receptive field of each pixel.
        centers = 0.5 + disp
        # centers is [h, w, 2]
        rect_size = expand_dims_n(rect_size, -2, 2)  # [b, 2] -> [b, 1, 1, 2]
        return geom.make_rect_center_size(centers, rect_size)


def rect_grid_pyr(response_size, rf, search_size, rect_size, scales, name='rect_grid_pyr'):
    '''Obtains the rectangles for a translation and scale scoremap.

    Args:
        response_size -- Dimension of scoremap. (height, width)
        rf -- Receptive field of scoremap.
        search_size -- Dimension of search image in pixels. (height, width)
        rect_size -- Size of rectangle in normalized co-ords in search image. [b, 2]
    '''
    with tf.name_scope(name) as scope:
        # Assert that receptive fields are centered.
        receptive_field.assert_center_alignment(search_size, response_size, rf)
        # Obtain displacement from center of search image.
        # Not necessary to use receptive field offset because it is centered.
        disp = displacement_from_center(response_size)
        disp = tf.to_float(disp) * rf.stride / search_size
        disp = tf.multiply(tf.expand_dims(disp, -4),     # [b, h, w, 2] -> [b, 1, h, w, 2]
                           expand_dims_n(scales, -1, 3))  # [s] -> [s, 1, 1, 1]
        # Get centers of receptive field of each pixel.
        centers = 0.5 + disp
        rect_size = tf.multiply(tf.expand_dims(rect_size, -2),  # [b, 2] -> [b, 1, 2]
                                tf.expand_dims(scales, -1))    # [s] -> [s, 1]
        rect_size = expand_dims_n(rect_size, -2, 2)  # [b, s, 2] -> [b, s, 1, 1, 2]
        return geom.make_rect_center_size(centers, rect_size)


def colormap(x, cmap_name, name='colormap'):
    with tf.name_scope(name) as scope:
        x_shape = x.shape.as_list()
        assert x_shape[-1] == 1
        y = tf.py_func(functools.partial(_colormap_np, cmap_name), [x], tf.float32)
        y.set_shape(x_shape[:-1] + [4])
        return y


def _colormap_np(cmap_name, x):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    x = np.squeeze(x, axis=-1)
    x = cmap(x)
    x = np.asfarray(x, np.float32)
    return x
