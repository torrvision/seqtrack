import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import math

from seqtrack import cnnutil
from seqtrack import geom
from seqtrack import lossfunc
from seqtrack import models
from seqtrack.models import interface as models_interface
from seqtrack.models import util

from seqtrack.helpers import merge_dims, expand_dims_n, get_act
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

_COLORMAP = 'viridis'


class SiamFC(models_interface.IterModel):

    def __init__(
            self,
            template_size=127,
            search_size=255,
            template_scale=2,
            aspect_method='perimeter', # TODO: Equivalent to SiamFC?
            pad_with_mean=False, # Use mean of first image for padding.
            feather=True,
            feather_margin=0.05,
            center_input_range=True, # Make range [-0.5, 0.5] instead of [0, 1].
            keep_uint8_range=False, # Use input range of 255 instead of 1.
            feature_padding='VALID',
            xcorr_padding='VALID',
            bnorm_after_xcorr=False,
            # Tracking parameters:
            num_scales=5,
            scale_step=1.03,
            scale_update_rate=0.6,
            report_square=False,
            enable_hann=True,
            # Loss parameters:
            sigma=0.3,
            balance_classes=False,
            loss_coeffs=None):
        self._template_size = template_size
        self._search_size = search_size
        self._aspect_method = aspect_method
        self._pad_with_mean = pad_with_mean
        self._feather = feather
        self._feather_margin = feather_margin
        self._keep_uint8_range = keep_uint8_range
        self._center_input_range = center_input_range
        self._template_scale = template_scale
        self._search_scale = float(search_size) / template_size * template_scale
        self._feature_padding = feature_padding
        self._xcorr_padding = xcorr_padding
        self._bnorm_after_xcorr = bnorm_after_xcorr
        self._num_scales = num_scales
        self._scale_step = scale_step
        self._scale_update_rate = scale_update_rate
        self._report_square = report_square
        self._enable_hann = enable_hann
        self._sigma = sigma
        self._balance_classes = balance_classes
        self._loss_coeffs = loss_coeffs or {}

        self._num_frames = 0
        # For summaries in end():
        self._info = {k: [] for k in [
            'search', 'response', 'labels', 'response_in_search',
            'response_final', 'response_final_in_image',
        ]}
        self._losses = {}

    def start(self, frame, aspect, run_opts, enable_loss,
              image_summaries_collections=None, name='start'):
        with tf.name_scope(name) as scope:
            self._aspect = aspect
            self._is_training = run_opts['is_training']
            self._is_tracking = run_opts['is_tracking']
            self._enable_loss = enable_loss
            # self._init_rect = frame['y']
            self._image_summaries_collections = image_summaries_collections

            mean_color = tf.reduce_mean(frame['x'], axis=(-3, -2), keep_dims=True)
            # TODO: frame['image'] and template_im have a viewport
            rect_square = util.coerce_aspect(frame['y'], im_aspect=self._aspect,
                                             aspect_method=self._aspect_method)
            template_rect = geom.grow_rect(self._template_scale, rect_square)
            template_im = util.crop(
                frame['x'], template_rect, self._template_size,
                pad_value=mean_color if self._pad_with_mean else 0.5,
                feather=self._feather, feather_margin=self._feather_margin)
            template_input = self._preproc(template_im)
            with tf.variable_scope('feature_net', reuse=False):
                template_feat, _ = _feature_net(template_input, padding=self._feature_padding,
                                                is_training=self._is_training)

            with tf.name_scope('summary'):
                tf.summary.image('template', _to_uint8(template_im[0:1]),
                                 max_outputs=1, collections=self._image_summaries_collections)

            # TODO: Avoid passing template_feat to and from GPU (or re-computing).
            state = {
                'y':             tf.identity(frame['y']),
                'template_feat': tf.identity(template_feat),
                'mean_color':    tf.identity(mean_color),
            }
            return state

    def next(self, frame, prev_state, name='timestep'):
        with tf.name_scope(name) as scope:
            # During training, let the "previous location" be the current true location
            # so that the object is in the center of the search area (like SiamFC).
            gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_state['y'])
            prev_rect = tf.cond(self._is_tracking, lambda: prev_state['y'], lambda: gt_rect)
            # Coerce the aspect ratio of the rectangle to construct the search area.
            prev_rect_square = util.coerce_aspect(
                prev_rect, im_aspect=self._aspect, aspect_method=self._aspect_method)
            if self._report_square:
                prev_rect = prev_rect_square
            search_rect = geom.grow_rect(self._search_scale, prev_rect_square)

            # Extract an image pyramid (use 1 scale when not in tracking mode).
            num_scales = tf.cond(self._is_tracking, lambda: self._num_scales, lambda: 1)
            mid_scale = (num_scales - 1) / 2
            scales = util.scale_range(num_scales, self._scale_step)
            search_ims, search_rects = util.crop_pyr(
                frame['x'], search_rect, self._search_size, scales,
                pad_value=prev_state['mean_color'] if self._pad_with_mean else 0.5,
                feather=self._feather, feather_margin=self._feather_margin)
            self._info['search'].append(_to_uint8(search_ims[:, mid_scale]))

            # Extract features, perform search, get receptive field of response wrt image.
            rfs = {'search': cnnutil.identity_rf()}
            search_input = self._preproc(search_ims)
            with tf.variable_scope('feature_net', reuse=True):
                search_feat, rfs = _feature_net(search_input, rfs, padding=self._feature_padding,
                                                is_training=self._is_training)
            response, rfs = util.diag_xcorr_rf(
                input=search_feat, filter=prev_state['template_feat'], input_rfs=rfs,
                padding=self._xcorr_padding)
            response = tf.reduce_sum(response, axis=-1, keep_dims=True)
            with tf.variable_scope('output', reuse=(self._num_frames > 0)):
                if self._bnorm_after_xcorr:
                    response = slim.batch_norm(response, scale=True, is_training=self._is_training)
                else:
                    gain = tf.get_variable('gain', shape=[], dtype=tf.float32)
                    bias = tf.get_variable('bias', shape=[], dtype=tf.float32)
                    response = gain * response + bias
            response_size = response.shape.as_list()[-3:-1]
            assert_center_alignment(self._search_size, response_size, rfs['search'])
            response_cmap = util.colormap(tf.sigmoid(response), _COLORMAP)
            self._info['response'].append(_to_uint8(response_cmap[:, mid_scale]))

            # TODO: Simplify!
            losses = {}
            if self._enable_loss:
                # Obtain displacement from center of search image.
                disp = util.displacement_from_center(response_size)
                disp = tf.to_float(disp) * rfs['search'].stride / self._search_size
                grid = 0.5 + disp

                gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
                # labels, has_label = lossfunc.translation_labels(grid, gt_rect_in_search, **kwargs)
                labels, has_label = lossfunc.translation_labels(
                    grid, gt_rect_in_search, shape='gaussian', sigma=self._sigma/self._search_scale)
                labels_cmap = util.colormap(tf.expand_dims(labels, -1), _COLORMAP)
                self._info['labels'].append(_to_uint8(labels_cmap))
                if self._balance_classes:
                    weights = lossfunc.make_balanced_weights(labels, has_label, axis=(-2, -1))
                else:
                    weights = lossfunc.make_uniform_weights(has_label, axis=(-2, -1))
                # Note: This squeeze will fail if there is an image pyramid (is_tracking=True).
                losses['ce'] = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels,                            # [b, h, w]
                    logits=tf.squeeze(response, axis=(1, 4))) # [b, s, h, w, 1] -> [b, h, w]
                # TODO: Is this the best way to handle y_is_valid?
                losses['ce'] = tf.where(frame['y_is_valid'], losses['ce'], tf.zeros_like(losses['ce']))
                losses['ce'] = tf.reduce_mean(tf.reduce_sum(weights * losses['ce'], axis=(1, 2)))

            for k in losses:
                self._losses.setdefault(k, []).append(losses[k])
            loss = tf.add_n([float(self._loss_coeffs[k]) * v for k, v in losses.items()])

            # Get relative translation and scale from response.
            response_final = _finalize_scores(
                response, rfs['search'].stride, enable_hann=self._enable_hann)
            upsample_size = response_final.shape.as_list()[-2:]
            translation_in_search, scale = _find_peak(response_final, self._search_size, scales)

            # Draw coarse response over search image.
            rf_centers = _find_rf_centers(self._search_size, response_size, rfs['search'])
            response_in_search = _align_corner_centers(rf_centers, response_size)
            self._info['response_in_search'].append(_to_uint8(paste_image_at_rect(
                search_ims[:, mid_scale], response_cmap[:, mid_scale], response_in_search, alpha=0.5)))
            # Draw upsample, regularized response over original image.
            response_final_cmap = util.colormap(tf.expand_dims(response_final, -1), _COLORMAP)
            self._info['response_final'].append(_to_uint8(response_final_cmap[:, mid_scale]))
            response_final_in_search = _align_corner_centers(rf_centers, upsample_size)
            response_final_in_image = geom.crop_rect(response_final_in_search,
                                                     geom.crop_inverse(search_rect))
            # TODO: How to visualize multi-scale responses?
            vis = paste_image_at_rect(frame['x'], response_final_cmap[:, mid_scale],
                                      response_final_in_image, alpha=0.5)
            self._info['response_final_in_image'].append(_to_uint8(vis))

            # Damp the scale update towards 1.
            scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.
            # Get rectangle in search image.
            prev_target_in_search = geom.crop_rect(prev_rect, search_rect)
            pred_in_search = _rect_translate_scale(prev_target_in_search, translation_in_search,
                                                   tf.expand_dims(scale, -1))
            # Move from search back to original image.
            # TODO: Test that this is equivalent to scaling translation?
            pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            # gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_rect)
            next_prev_rect = tf.cond(self._is_training, lambda: gt_rect, lambda: pred)

            self._num_frames += 1
            outputs = {'y': pred, 'vis': vis}
            state = {
                'y': next_prev_rect,
                'template_feat': prev_state['template_feat'],
                'mean_color':    prev_state['mean_color'],
            }
            return outputs, state, loss

    def end(self):
        extra_loss = 0.
        with tf.name_scope('loss_summary'):
            for loss_name, values in self._losses.items():
                tf.summary.scalar(loss_name, tf.reduce_mean(values))
        with tf.name_scope('summary'):
            sequence_keys = ['search', 'response', 'response_in_search',
                             'response_final', 'response_final_in_image']
            for key in sequence_keys:
                image_sequence_summary(key, tf.stack(self._info[key], axis=1),
                                       collections=self._image_summaries_collections)
            if self._enable_loss:
                image_sequence_summary('labels', tf.stack(self._info['labels'], axis=1),
                                       collections=self._image_summaries_collections)
        return extra_loss

    def _preproc(self, im, name='preproc'):
        with tf.name_scope(name) as scope:
            if self._center_input_range:
                im -= 0.5
            if self._keep_uint8_range:
                im *= 255.
            return tf.identity(im, scope)


def image_sequence_summary(name, tensor, **kwargs):
    '''
    Args:
        tensor: [b, t, h, w, c]
    '''
    ntimesteps = tensor.shape.as_list()[-4]
    assert ntimesteps is not None
    tf.summary.image(name, tensor[0], max_outputs=ntimesteps, **kwargs)


def _rect_translate_scale(rect, translate, scale, name='rect_translate_scale'):
    '''
    Args:
        rect: [..., 4]
        translate: [..., 2]
        scale: [..., 1]
    '''
    with tf.name_scope(name) as scope:
        min_pt, max_pt = geom.rect_min_max(rect)
        center, size = 0.5 * (min_pt + max_pt), max_pt - min_pt
        center += translate
        size *= scale
        return geom.make_rect(center - 0.5 * size, center + 0.5 * size)


def _feature_net(x, rfs=None, padding=None,
                 output_act='linear', enable_output_bnorm=False,
                 is_training=None, name='feature_net'):
    '''
    Returns:
        Tuple of (feature map, receptive fields).

    To use bnorm, use an external slim.arg_scope.
    '''
    assert padding is not None
    assert is_training is not None
    if rfs is None:
        rfs = {}
    # For feature pyramid, support rank > 4.
    if len(x.shape) > 4:
        # Merge dims (0, ..., n-4), n-3, n-2, n-1
        x, restore = merge_dims(x, 0, len(x.shape)-3)
        x, rfs = _feature_net(x, rfs, padding, output_act, enable_output_bnorm, is_training, name)
        x = restore(x, 0)
        return x, rfs

    with tf.name_scope(name) as scope:
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=dict(is_training=is_training)):
            # https://github.com/bertinetto/siamese-fc/blob/master/training/vid_create_net.m
            # https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
            # TODO: Support arg_scope for padding?
            x, rfs = util.conv2d_rf(x, rfs, 96, [11, 11], 2, padding=padding, scope='conv1')
            x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool1')
            x, rfs = util.conv2d_rf(x, rfs, 256, [5, 5], padding=padding, scope='conv2')
            x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
            x, rfs = util.conv2d_rf(x, rfs, 384, [3, 3], padding=padding, scope='conv3')
            x, rfs = util.conv2d_rf(x, rfs, 384, [3, 3], padding=padding, scope='conv4')
            x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], padding=padding, scope='conv5',
                                    activation_fn=get_act(output_act), normalizer_fn=None)
            return x, rfs


def _finalize_scores(response, stride, enable_hann, name='finalize_scores'):
    '''
    Args:
        response: [b, s, h, w, c] where c is 1 or 2.

    stride is (y, x) integer
    '''
    with tf.name_scope(name) as scope:
        response_size = response.shape.as_list()[-3:-1]
        assert all(response_size)
        response_size = np.array(response_size)
        assert all(response_size % 2 == 1)
        stride = np.array(n_positive_integers(2, stride))
        upsample_size = (response_size - 1) * stride + 1
        # Upsample.
        response, restore_fn = merge_dims(response, 0, 2)
        response = tf.image.resize_images(
            response, upsample_size, method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        response = restore_fn(response, 0)
        # Map to range [0, 1]. Assume c == 1.
        response = tf.sigmoid(tf.squeeze(response, -1))
        # Apply motion penalty at all scales.
        if enable_hann:
            response = response * hann_2d(upsample_size)
        return response


def _find_peak(response, search_size, scales, name='find_peak'):
    '''
    Args:
        response: [b, s, h, w]

    Assumes that response is centered and at same stride as search image.
    '''
    with tf.name_scope(name) as scope:
        upsample_size = response.shape.as_list()[-2:]
        assert all(upsample_size)
        upsample_size = np.array(upsample_size)
        # Find arg max over all scales.
        max_val = tf.reduce_max(response, axis=(-3, -2, -1), keep_dims=True)
        is_max = tf.to_float(response >= max_val)

        grid = tf.to_float(util.displacement_from_center(upsample_size))
        assert np.all(upsample_size <= search_size)
        grid = grid / search_size

        # Grid now has translation from center in search image co-ords.
        # Transform into co-ordinate frame of each scale.
        grid = tf.multiply(grid,                           # [h, w, 2]
                           expand_dims_n(scales, -1, n=3)) # [s, 1, 1, 1]

        translation = tf.reduce_sum(
            tf.multiply(tf.expand_dims(is_max, -1), # [b, s, h, w] -> [b, s, h, w, 1]
                        grid),                      # [s, h, w, 2]
            axis=(-4, -3, -2))                      # [b, s, h, w, 2] -> [b, 2]
        scale = tf.reduce_sum(
            tf.multiply(is_max,                          # [b, s, h, w]
                        expand_dims_n(scales, -1, n=2)), # [b, s] -> [b, s, 1, 1]
            axis=(-3, -2, -1))                           # [b, s, h, w] -> [b]
        return translation, scale


def hann(n, name='hann'):
    with tf.name_scope(name) as scope:
        n = tf.convert_to_tensor(n)
        x = tf.to_float(tf.range(n)) / tf.to_float(n - 1)
        return 0.5 * (1. - tf.cos(2.*math.pi*x))

def hann_2d(im_size, name='hann_2d'):
    with tf.name_scope(name) as scope:
        size_0, size_1 = im_size
        window_0 = hann(size_0)
        window_1 = hann(size_1)
        return tf.expand_dims(window_0, 1) * tf.expand_dims(window_1, 0)


def assert_center_alignment(input_size, output_size, rf):
    '''
    Args:
        input_size: (height, width)
        output_size: (height, width)
        rf: cnnutil.ReceptiveField, which uses (height, width)
    '''
    input_size = np.array(n_positive_integers(2, input_size))
    output_size = np.array(n_positive_integers(2, output_size))

    min_pt = rf.rect.int_center()
    max_pt = min_pt + rf.stride * (output_size - 1) + 1
    gap_before = min_pt
    gap_after = input_size - max_pt
    # If gap_before is equal to gap_after, then center of response map
    # corresponds to center of search image.
    np.testing.assert_array_equal(gap_before, gap_after)


def _find_rf_centers(input_size_hw, output_size_hw, rf, name='rf_centers'):
    input_size_hw = n_positive_integers(2, input_size_hw)
    output_size_hw = n_positive_integers(2, output_size_hw)
    min_pixel, max_pixel = cnnutil.rf_centers_in_input(output_size_hw, rf)
    with tf.name_scope(name) as scope:
        # Switch to (x, y) for rect.
        input_size = tf.to_float(input_size_hw[::-1])
        output_size = tf.to_float(output_size_hw[::-1])
        min_pt = tf.to_float(min_pixel[::-1]) + 0.5 # Take center point of pixel.
        max_pt = tf.to_float(max_pixel[::-1]) + 0.5
        scale = 1. / input_size
        return geom.make_rect(scale * min_pt, scale * max_pt)

def _align_corner_centers(rect, im_size_hw, name='align_corner_centers'):
    with tf.name_scope(name) as scope:
        min_pt, max_pt = geom.rect_min_max(rect)
        # Switch to (x, y) for rect.
        im_size = tf.to_float(im_size_hw[::-1])
        return geom.grow_rect(im_size / (im_size - 1), rect)

def paste_image_at_rect(target, overlay, rect, alpha=1, name='paste_image_at_rect'):
    with tf.name_scope(name) as scope:
        overlay = _ensure_rgba(overlay)
        target_size = target.shape.as_list()[-3:-1]
        overlay = util.crop(overlay, geom.crop_inverse(rect), target_size)
        return paste_image(target, overlay, alpha=alpha)

def paste_image(target, overlay, alpha=1, name='paste_image'):
    with tf.name_scope(name) as scope:
        overlay_rgb, overlay_a = tf.split(overlay, [3, 1], axis=-1)
        overlay_a *= alpha
        return overlay_a * overlay_rgb + (1-overlay_a) * target

def _ensure_rgba(im, name='ensure_rgba'):
    with tf.name_scope(name) as scope:
        num_channels = im.shape.as_list()[-1]
        assert num_channels is not None
        if num_channels == 1:
            return _ensure_rgba(tf.image.rgb_to_grayscale(im), name=name)
        if num_channels == 4:
            return tf.identity(im)
        assert num_channels == 3
        shape = most_static_shape(im)
        return tf.concat([im, tf.ones(shape[:-1]+[1], im.dtype)], axis=-1)

def _to_uint8(x):
    return tf.image.convert_image_dtype(x, tf.uint8, saturate=True)
