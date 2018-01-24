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

from seqtrack.helpers import merge_dims
from seqtrack.helpers import get_act
from seqtrack.helpers import leaky_relu
from seqtrack.helpers import expand_dims_n
from seqtrack.helpers import weighted_mean
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

_COLORMAP = 'viridis'


class SiamFC(models_interface.IterModel):

    def __init__(
            self,
            template_size=127,
            search_size=255,
            template_scale=2,
            aspect_method='perimeter', # TODO: Equivalent to SiamFC?
            use_gt=True,
            curr_as_prev=True,
            pad_with_mean=False, # Use mean of first image for padding.
            feather=True,
            feather_margin=0.1,
            center_input_range=True, # Make range [-0.5, 0.5] instead of [0, 1].
            keep_uint8_range=False, # Use input range of 255 instead of 1.
            feature_padding='VALID',
            feature_arch='alexnet',
            feature_act='linear',
            enable_feature_bnorm=True,
            enable_template_mask=False,
            xcorr_padding='VALID',
            bnorm_after_xcorr=True,
            freeze_siamese=False,
            learnable_prior=False,
            # Tracking parameters:
            num_scales=5,
            scale_step=1.03,
            scale_update_rate=0.6,
            report_square=False,
            hann_method='none', # none, mul_prob, add_logit
            hann_coeff=1.0,
            arg_max_eps_rel=0.05,
            # Loss parameters:
            wd=0.0,
            enable_ce_loss=True,
            ce_label='gaussian_distance',
            sigma=0.2,
            balance_classes=True,
            enable_margin_loss=False,
            margin_cost='iou',
            margin_structured=True):
        self._template_size = template_size
        self._search_size = search_size
        self._aspect_method = aspect_method
        self._use_gt = use_gt
        self._curr_as_prev = curr_as_prev
        self._pad_with_mean = pad_with_mean
        self._feather = feather
        self._feather_margin = feather_margin
        self._keep_uint8_range = keep_uint8_range
        self._center_input_range = center_input_range
        self._template_scale = template_scale
        # Size of search area relative to object.
        self._search_scale = float(search_size) / template_size * template_scale
        self._feature_padding = feature_padding
        self._feature_arch = feature_arch
        self._feature_act = feature_act
        self._enable_feature_bnorm = enable_feature_bnorm
        self._enable_template_mask = enable_template_mask
        self._xcorr_padding = xcorr_padding
        self._bnorm_after_xcorr = bnorm_after_xcorr
        self._freeze_siamese = freeze_siamese
        self._learnable_prior = learnable_prior
        self._num_scales = num_scales
        self._scale_step = scale_step
        self._scale_update_rate = scale_update_rate
        self._hann_method = hann_method
        self._hann_coeff = hann_coeff
        self._arg_max_eps_rel = arg_max_eps_rel
        self._wd = wd
        self._enable_ce_loss = enable_ce_loss
        self._ce_label = ce_label
        self._sigma = sigma
        self._balance_classes = balance_classes
        self._enable_margin_loss = enable_margin_loss
        self._margin_cost = margin_cost
        self._margin_structured = margin_structured

        self._num_frames = 0
        # For summaries in end():
        self._info = {}

    def start(self, frame, aspect, run_opts, enable_loss,
              image_summaries_collections=None, name='start'):
        with tf.name_scope(name) as scope:
            self._aspect = aspect
            self._is_training = run_opts['is_training']
            self._is_tracking = run_opts['is_tracking']
            self._enable_loss = enable_loss
            self._image_summaries_collections = image_summaries_collections

            mean_color = tf.reduce_mean(frame['x'], axis=(-3, -2), keep_dims=True)
            # TODO: frame['image'] and template_im have a viewport
            template_rect, _ = _get_context_rect(frame['y'], context_amount=self._template_scale,
                                                 aspect=self._aspect, aspect_method=self._aspect_method)
            template_im = util.crop(frame['x'], template_rect, self._template_size,
                                    pad_value=mean_color if self._pad_with_mean else 0.5,
                                    feather=self._feather, feather_margin=self._feather_margin)

            rfs = {'template': cnnutil.identity_rf()}
            template_input = _preproc(template_im, center_input_range=self._center_input_range,
                                      keep_uint8_range=self._keep_uint8_range)
            with tf.variable_scope('feature_net', reuse=False):
                template_feat, rfs = _feature_net(
                    template_input, rfs, padding=self._feature_padding, arch=self._feature_arch,
                    output_act=self._feature_act, enable_bnorm=self._enable_feature_bnorm,
                    wd=self._wd, is_training=self._is_training,
                    variables_collections=['siamese'], trainable=(not self._freeze_siamese))
            feat_size = template_feat.shape.as_list()[-3:-1]
            cnnutil.assert_center_alignment(self._template_size, feat_size, rfs['template'])
            if self._enable_template_mask:
                template_mask = tf.get_variable(
                    'template_mask', template_feat.shape.as_list()[-3:],
                    initializer=tf.ones_initializer(), collections=['siamese'])
                template_feat *= template_mask

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
            if self._curr_as_prev:
                prev_rect = tf.cond(self._is_tracking, lambda: prev_state['y'], lambda: gt_rect)
            else:
                prev_rect = prev_state['y']
            # Coerce the aspect ratio of the rectangle to construct the search area.
            search_rect, _ = _get_context_rect(prev_rect, self._search_scale,
                                               aspect=self._aspect, aspect_method=self._aspect_method)

            # Extract an image pyramid (use 1 scale when not in tracking mode).
            num_scales = tf.cond(self._is_tracking, lambda: self._num_scales, lambda: 1)
            mid_scale = (num_scales - 1) / 2
            scales = util.scale_range(num_scales, self._scale_step)
            search_ims, search_rects = util.crop_pyr(
                frame['x'], search_rect, self._search_size, scales,
                pad_value=prev_state['mean_color'] if self._pad_with_mean else 0.5,
                feather=self._feather, feather_margin=self._feather_margin)

            self._info.setdefault('search', []).append(_to_uint8(search_ims[:, mid_scale]))

            # Extract features, perform search, get receptive field of response wrt image.
            rfs = {'search': cnnutil.identity_rf()}
            search_input = _preproc(search_ims, center_input_range=self._center_input_range,
                                    keep_uint8_range=self._keep_uint8_range)
            with tf.variable_scope('feature_net', reuse=True):
                search_feat, rfs = _feature_net(
                    search_input, rfs, padding=self._feature_padding, arch=self._feature_arch,
                    output_act=self._feature_act, enable_bnorm=self._enable_feature_bnorm,
                    wd=self._wd, is_training=self._is_training,
                    variables_collections=['siamese'], trainable=(not self._freeze_siamese))

            response, rfs = util.diag_xcorr_rf(
                input=search_feat, filter=prev_state['template_feat'], input_rfs=rfs,
                padding=self._xcorr_padding)
            response = tf.reduce_sum(response, axis=-1, keep_dims=True)
            response_size = response.shape.as_list()[-3:-1]
            cnnutil.assert_center_alignment(self._search_size, response_size, rfs['search'])

            with tf.variable_scope('output', reuse=(self._num_frames > 0)):
                if self._bnorm_after_xcorr:
                    response = slim.batch_norm(response, scale=True, is_training=self._is_training,
                                               variables_collections=['siamese'])
                else:
                    response = _affine_scalar(response, variables_collections=['siamese'])
                if self._freeze_siamese:
                    # TODO: Prevent batch-norm updates as well.
                    # TODO: Set trainable=False for all variables above.
                    response = tf.stop_gradient(response)
                if self._learnable_prior:
                    response = _add_motion_prior(response)

            self._info.setdefault('response', []).append(
                _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))

            losses = {}
            if self._enable_loss:
                if self._enable_ce_loss:
                    losses['ce'], labels = _cross_entropy_loss(
                        response, rfs['search'], prev_rect, gt_rect, frame['y_is_valid'], search_rect,
                        search_size=self._search_size, search_scale=self._search_scale,
                        label_method=self._ce_label, sigma=self._sigma,
                        balance_classes=self._balance_classes)
                    self._info.setdefault('ce_labels', []).append(
                        _to_uint8(util.colormap(tf.expand_dims(labels, -1), _COLORMAP)))
                if self._enable_margin_loss:
                    losses['margin'], cost = _max_margin_loss(
                        response, rfs['search'], prev_rect, gt_rect, frame['y_is_valid'], search_rect,
                        search_size=self._search_size, search_scale=self._search_scale,
                        cost_method=self._margin_cost, structured=self._margin_structured)
                    self._info.setdefault('margin_cost', []).append(
                        _to_uint8(util.colormap(tf.expand_dims(cost, -1), _COLORMAP)))

            # Get relative translation and scale from response.
            response_final = _finalize_scores(response, rfs['search'].stride,
                                              self._hann_method, self._hann_coeff)
            # upsample_response_size = response_final.shape.as_list()[-3:-1]
            # assert np.all(upsample_response_size <= self._search_size)
            translation, scale = util.find_peak_pyr(response_final, scales,
                                                    eps_rel=self._arg_max_eps_rel)
            translation = translation / self._search_size

            vis = _visualize_response(
                response[:, mid_scale], response_final[:, mid_scale],
                search_ims[:, mid_scale], rfs['search'], frame['x'], search_rect)
            self._info.setdefault('vis', []).append(_to_uint8(vis))

            # Damp the scale update towards 1.
            scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.
            # Get rectangle in search image.
            prev_target_in_search = geom.crop_rect(prev_rect, search_rect)
            pred_in_search = _rect_translate_scale(prev_target_in_search, translation,
                                                   tf.expand_dims(scale, -1))
            # Move from search back to original image.
            # TODO: Test that this is equivalent to scaling translation?
            pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            # gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_rect)
            if self._use_gt:
                next_prev_rect = tf.cond(self._is_tracking, lambda: pred, lambda: gt_rect)
            else:
                next_prev_rect = pred

            self._num_frames += 1
            outputs = {'y': pred, 'vis': vis}
            state = {
                'y': next_prev_rect,
                'template_feat': prev_state['template_feat'],
                'mean_color':    prev_state['mean_color'],
            }
            return outputs, state, losses

    def end(self):
        losses = {}
        with tf.name_scope('summary'):
            for key in self._info:
                _image_sequence_summary(key, tf.stack(self._info[key], axis=1),
                                        collections=self._image_summaries_collections)
        return losses


def _feature_net(x, rfs=None, padding=None, arch='alexnet', output_act='linear', enable_bnorm=True,
                 wd=0.0, is_training=None, variables_collections=None, trainable=False,
                 name='feature_net'):
    '''
    Returns:
        Tuple of (feature map, receptive fields).
    '''
    assert padding is not None
    assert is_training is not None
    if rfs is None:
        rfs = {}
    # For feature pyramid, support rank > 4.
    if len(x.shape) > 4:
        # Merge dims (0, ..., n-4), n-3, n-2, n-1
        x, restore = merge_dims(x, 0, len(x.shape)-3)
        x, rfs = _feature_net(
            x, rfs=rfs, padding=padding, arch=arch, output_act=output_act,
            enable_bnorm=enable_bnorm, wd=wd, is_training=is_training,
            variables_collections=variables_collections, trainable=trainable, name=name)
        x = restore(x, 0)
        return x, rfs

    with tf.name_scope(name) as scope:
        conv_args = dict(weights_regularizer=slim.l2_regularizer(wd) if wd > 0 else None,
                         variables_collections=variables_collections,
                         trainable=trainable)
        if enable_bnorm:
            conv_args.update(dict(
                normalizer_fn=slim.batch_norm,
                normalizer_params=dict(
                    is_training=is_training if trainable else False, # Fix bnorm if not trainable.
                    variables_collections=variables_collections)))
        with slim.arg_scope([slim.conv2d], **conv_args):
            if arch == 'alexnet':
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
            elif arch == 'darknet':
                # https://github.com/pjreddie/darknet/blob/master/cfg/darknet.cfg
                with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu):
                    x, rfs = util.conv2d_rf(x, rfs, 16, [3, 3], 1, padding=padding, scope='conv1')
                    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool1')
                    x, rfs = util.conv2d_rf(x, rfs, 32, [3, 3], 1, padding=padding, scope='conv2')
                    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
                    x, rfs = util.conv2d_rf(x, rfs, 64, [3, 3], 1, padding=padding, scope='conv3')
                    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool3')
                    x, rfs = util.conv2d_rf(x, rfs, 128, [3, 3], 1, padding=padding, scope='conv4')
                    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool4')
                    x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], 1, padding=padding, scope='conv5',
                                            activation_fn=get_act(output_act), normalizer_fn=None)
            else:
                raise ValueError('unknown architecture: {}'.format(arch))
            return x, rfs


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

def _paste_image_at_rect(target, overlay, rect, alpha=1, name='paste_image_at_rect'):
    with tf.name_scope(name) as scope:
        overlay = _ensure_rgba(overlay)
        target_size = target.shape.as_list()[-3:-1]
        overlay = util.crop(overlay, geom.crop_inverse(rect), target_size)
        return _paste_image(target, overlay, alpha=alpha)

def _paste_image(target, overlay, alpha=1, name='paste_image'):
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

def _affine_scalar(x, name='affine', variables_collections=None):
    with tf.name_scope(name) as scope:
        gain = tf.get_variable('gain', shape=[], dtype=tf.float32,
                               variables_collections=variables_collections)
        bias = tf.get_variable('bias', shape=[], dtype=tf.float32,
                               variables_collections=variables_collections)
        return gain * x + bias


def _preproc(im, center_input_range, keep_uint8_range, name='preproc'):
    with tf.name_scope(name) as scope:
        if center_input_range:
            im -= 0.5
        if keep_uint8_range:
            im *= 255.
        return tf.identity(im, scope)


def _get_context_rect(rect, context_amount, aspect, aspect_method):
    square = util.coerce_aspect(rect, im_aspect=aspect, aspect_method=aspect_method)
    context = geom.grow_rect(context_amount, square)
    return context, square


def _add_motion_prior(response, name='motion_prior'):
    with tf.name_scope(name) as scope:
        response_shape = response.shape.as_list()[-3:]
        assert response_shape[-1] == 1
        prior = tf.get_variable('prior', shape=response_shape, dtype=tf.float32,
                                initializer=tf.zeros_initializer(dtype=tf.float32))
        # self._info.setdefault('response_appearance', []).append(
        #     _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))
        # if self._num_frames == 0:
        #     tf.summary.image(
        #         'motion_prior', max_outputs=1,
        #         tensor=_to_uint8(util.colormap(tf.sigmoid([motion_prior]), _COLORMAP)),
        #         collections=self._image_summaries_collections)
        return response + prior


def _cross_entropy_loss(
        response, response_rf, prev_rect, gt_rect, gt_is_valid, search_rect,
        search_size, sigma, search_scale, balance_classes,
        label_method='gaussian_distance', name='cross_entropy_translation'):
    '''Computes the loss for a 2D map of logits.

    Args:
        response -- 2D map of logits.
        response_rf -- Receptive field of response BEFORE upsampling.
        gt_rect -- Ground-truth rectangle in original image frame.
        gt_is_valid -- Whether ground-truth label is present (valid).
        search_rect -- Rectangle that describes search area in original image.
        search_size -- Size of search image cropped from original image.
        sigma -- Size of ground-truth labels relative to object size.
        search_scale -- Size of search region relative to object size.
        balance_classes -- Should classes be balanced?
    '''
    with tf.name_scope(name) as scope:
        # [b, s, h, w, 1] -> [b, h, w]
        response = tf.squeeze(response, 1) # Remove the scales dimension.
        response = tf.squeeze(response, -1) # Remove the trailing dimension.
        response_size = response.shape.as_list()[-2:]

        # Obtain displacement from center of search image.
        disp = util.displacement_from_center(response_size)
        disp = tf.to_float(disp) * response_rf.stride / search_size
        # Get centers of receptive field of each pixel.
        centers = 0.5 + disp

        gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
        prev_rect_in_search = geom.crop_rect(prev_rect, search_rect)
        rect_grid = util.rect_grid(response_size, response_rf, search_size,
                                   geom.rect_size(prev_rect_in_search))

        if label_method == 'gaussian_distance':
            labels, has_label = lossfunc.translation_labels(
                centers, gt_rect_in_search, shape='gaussian', sigma=sigma/search_scale)
        elif label_method == 'best_iou':
            gt_rect_in_search = expand_dims_n(gt_rect_in_search, -2, 2)
            iou = geom.rect_iou(rect_grid, gt_rect_in_search)
            is_pos = (iou >= tf.reduce_max(iou, axis=[-2, -1], keep_dims=True))
            labels = tf.to_float(is_pos)
            has_label = tf.ones_like(labels, dtype=tf.bool)
        else:
            raise ValueError('unknown label method: {}'.format(label_method))

        # Note: This squeeze will fail if there is an image pyramid (is_tracking=True).
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=response)

        if balance_classes:
            weights = lossfunc.make_balanced_weights(labels, has_label, axis=(-2, -1))
        else:
            weights = lossfunc.make_uniform_weights(has_label, axis=(-2, -1))
        loss = tf.reduce_sum(weights * loss, axis=(1, 2))

        # TODO: Is this the best way to handle gt_is_valid?
        # (set loss to zero and include in mean)
        loss = tf.where(gt_is_valid, loss, tf.zeros_like(loss))
        loss = tf.reduce_mean(loss)
        return loss, labels


def _max_margin_loss(
        score, score_rf, prev_rect, gt_rect, gt_is_valid, search_rect,
        search_size, search_scale, cost_method='iou', structured=True,
        name='max_margin_loss'):
    '''Computes the loss for a 2D map of logits.

    Args:
        score -- 2D map of logits.
        score_rf -- Receptive field of score BEFORE upsampling.
        prev_rect -- Rectangle in previous frame (used to set size of predicted rectangle).
        gt_rect -- Ground-truth rectangle in original image frame.
        gt_is_valid -- Whether ground-truth label is present (valid).
        search_rect -- Rectangle that describes search area in original image.
        search_size -- Size of search image cropped from original image.
    '''
    with tf.name_scope(name) as scope:
        # [b, s, h, w, 1] -> [b, h, w]
        score = tf.squeeze(score, 1) # Remove the scales dimension.
        score = tf.squeeze(score, -1) # Remove the trailing dimension.
        score_size = score.shape.as_list()[-2:]
        gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
        prev_rect_in_search = geom.crop_rect(prev_rect, search_rect)
        rect_grid = util.rect_grid(score_size, score_rf, search_size,
                                   geom.rect_size(prev_rect_in_search))
        # rect_grid -- [b, h, w, 4]
        # gt_rect_in_search -- [b, 4]
        gt_rect_in_search = expand_dims_n(gt_rect_in_search, -2, 2)
        if cost_method == 'iou':
            iou = geom.rect_iou(rect_grid, gt_rect_in_search)
            cost = 1. - iou
        elif cost_method == 'iou_correct':
            iou = geom.rect_iou(rect_grid, gt_rect_in_search)
            max_iou = tf.reduce_max(iou, axis=[-2, -1], keep_dims=True)
            cost = tf.to_float(tf.logical_not(iou >= max_iou))
        elif cost_method == 'distance':
            delta = geom.rect_center(rect_grid) - geom.rect_center(gt_rect_in_search)
            # Distance in "object" units.
            cost = tf.norm(delta, axis=-1) * search_scale
        else:
            raise ValueError('unknown cost method: {}'.format(cost_method))
        loss = _max_margin(score, cost, axis=[-2, -1], structured=structured)
        # TODO: Is this the best way to handle gt_is_valid?
        # (set loss to zero and include in mean)
        loss = tf.where(gt_is_valid, loss, tf.zeros_like(loss))
        loss = tf.reduce_mean(loss)
        return loss, cost


def _max_margin(score, cost, axis=None, structured=True, name='max_margin'):
    '''Computes the max-margin loss.'''
    with tf.name_scope(name) as scope:
        is_best = tf.to_float(cost <= tf.reduce_min(cost, axis=axis, keep_dims=True))
        cost_best = weighted_mean(cost, is_best, axis=axis, keep_dims=True)
        score_best = weighted_mean(score, is_best, axis=axis, keep_dims=True)
        # We want the rectangle with the minimum cost to have the highest score.
        # => Ensure that the gap in score is at least the difference in cost.
        # i.e. score_best - score >= cost - cost_best
        # i.e. (cost - cost_best) - (score_best - score) <= 0
        # Therefore penalize max(0, above expr).
        violation = tf.maximum(0., (cost - cost_best) - (score_best - score))
        if structured:
            # Structured output loss.
            loss = tf.reduce_max(violation, axis=axis)
        else:
            # Mean over all hinges; like triplet loss.
            loss = tf.reduce_mean(violation, axis=axis)
        return loss


def _finalize_scores(response, stride, hann_method, hann_coeff, name='finalize_scores'):
    '''Modify scores before finding arg max.

    Includes upsampling, sigmoid and Hann window.

    Args:
        response: [b, s, h, w, 1]

    Returns:
        [b, s, h', w', 1]

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
        # Apply motion penalty at all scales.
        if hann_method == 'add_logit':
            response += hann_coeff * tf.expand_dims(hann_2d(upsample_size), -1)
            response = tf.sigmoid(response)
        elif hann_method == 'mul_prob':
            response = tf.sigmoid(response)
            response *= tf.expand_dims(hann_2d(upsample_size), -1)
        elif hann_method == 'none' or not hann_method:
            response = tf.sigmoid(response)
        else:
            raise ValueError('unknown hann method: {}'.format(hann_method))
        return response


def _visualize_response(
        response, response_final, search_im, response_rf, im, search_rect,
        name='visualize_response'):
    with tf.name_scope(name) as scope:
        response_size = response.shape.as_list()[-3:-1]
        search_size = search_im.shape.as_list()[-3:-1]
        rf_centers = _find_rf_centers(search_size, response_size, response_rf)

        # response is logits
        response_cmap = util.colormap(tf.sigmoid(response), _COLORMAP)
        # self._info.setdefault('response', []).append(_to_uint8(response_cmap))
        # Draw coarse response over search image.
        response_in_search = _align_corner_centers(rf_centers, response_size)
        # self._info.setdefault('response_in_search', []).append(_to_uint8(_paste_image_at_rect(
        #     search_im, response_cmap, response_in_search, alpha=0.5)))

        # response_final is probability
        response_final_cmap = util.colormap(response_final, _COLORMAP)
        # self._info.setdefault('response_final', []).append(_to_uint8(response_final_cmap))
        # Draw upsample, regularized response over original image.
        upsample_response_size = response_final.shape.as_list()[-3:-1]
        response_final_in_search = _align_corner_centers(rf_centers, upsample_response_size)
        response_final_in_image = geom.crop_rect(response_final_in_search, geom.crop_inverse(search_rect))
        # TODO: How to visualize multi-scale responses?
        response_final_in_image = _paste_image_at_rect(
            im, response_final_cmap, response_final_in_image, alpha=0.5)
        # self._info.setdefault('response_final_in_image', []).append(_to_uint8(response_final_in_image))

        return response_final_in_image


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


def _image_sequence_summary(name, tensor, **kwargs):
    '''
    Args:
        tensor: [b, t, h, w, c]
    '''
    ntimesteps = tensor.shape.as_list()[-4]
    assert ntimesteps is not None
    tf.summary.image(name, tensor[0], max_outputs=ntimesteps, **kwargs)
