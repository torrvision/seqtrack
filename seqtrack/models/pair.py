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

from seqtrack.helpers import merge_dims, get_act, leaky_relu
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

_COLORMAP = 'viridis'


class PairConcat(models_interface.IterModel):

    def __init__(
            self,
            template_size=135,
            search_size=263,
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
            xcorr_padding='VALID',
            bnorm_after_xcorr=True,
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
            sigma=0.2,
            balance_classes=True):
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
        self._search_scale = float(search_size) / template_size * template_scale
        self._feature_padding = feature_padding
        self._feature_arch = feature_arch
        self._feature_act = feature_act
        self._enable_feature_bnorm = enable_feature_bnorm
        self._xcorr_padding = xcorr_padding
        self._bnorm_after_xcorr = bnorm_after_xcorr
        self._learnable_prior = learnable_prior
        self._num_scales = num_scales
        self._scale_step = scale_step
        self._scale_update_rate = scale_update_rate
        self._report_square = report_square
        self._hann_method = hann_method
        self._hann_coeff = hann_coeff
        self._arg_max_eps_rel = arg_max_eps_rel
        self._sigma = sigma
        self._balance_classes = balance_classes

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
            state = {
                'x':          tf.identity(frame['x']),
                'y':          tf.identity(frame['y']),
                'mean_color': tf.identity(mean_color),
            }
            return state

    def next(self, frame, prev_state, name='timestep'):
        with tf.name_scope(name) as scope:
            gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_state['y'])
            prev_rect = prev_state['y']
            # Coerce the aspect ratio of the rectangle to construct the search area.
            search_rect, _ = self._get_search_rect(prev_rect)

            num_scales = tf.cond(self._is_tracking, lambda: self._num_scales, lambda: 1)
            mid_scale = (num_scales - 1) / 2
            # Extract an image pyramid (use 1 scale when not in tracking mode).
            scales = util.scale_range(num_scales, self._scale_step)
            search_ims, search_rects = self._crop_pyr(
                frame['x'], search_rect, self._search_size, scales, prev_state['mean_color'])
            self._info.setdefault('search', []).append(_to_uint8(search_ims[:, mid_scale]))
            # Extract same rectangle from previous image.
            prev_im = self._crop(prev_state['x'], search_rect, self._search_size, prev_state['mean_color'])
            prev_ims = tf.tile(tf.expand_dims(prev_im, axis=1), [1, num_scales, 1, 1, 1])

            # Extract features, perform search, get receptive field of response wrt image.
            search_input = tf.concat((self._preproc(search_ims), self._preproc(prev_ims)), axis=-1)

            rfs = {'search': cnnutil.identity_rf()}
            with tf.variable_scope('feature_net', reuse=(self._num_frames > 0)):
                search_feat, rfs = _feature_net(
                    search_input, rfs, padding=self._feature_padding, arch=self._feature_arch,
                    enable_bnorm=self._enable_feature_bnorm, is_training=self._is_training)
            feat_size = search_feat.shape.as_list()[-3:-1]
            cnnutil.assert_center_alignment(self._search_size, feat_size, rfs['search'])

            with tf.variable_scope('output', reuse=(self._num_frames > 0)):
                response = _output_net(search_feat)

            losses = {}
            if self._enable_loss:
                losses['ce'], labels = self._translation_loss(
                    response, rfs['search'], gt_rect, frame['y_is_valid'], search_rect)
                labels_cmap = util.colormap(tf.expand_dims(labels, -1), _COLORMAP)
                self._info.setdefault('labels', []).append(_to_uint8(labels_cmap))

            # Get relative translation and scale from response.
            response_final = self._finalize_scores(response, rfs['search'].stride)
            # upsample_response_size = response_final.shape.as_list()[-3:-1]
            # assert np.all(upsample_response_size <= self._search_size)
            translation, scale = util.find_peak_pyr(response_final, scales,
                                                    eps_rel=self._arg_max_eps_rel)
            translation = translation / self._search_size

            vis = self._visualize_response(
                response[:, mid_scale], response_final[:, mid_scale],
                search_ims[:, mid_scale], rfs['search'], frame['x'], search_rect)

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
                'x':          frame['x'],
                'y':          next_prev_rect,
                'mean_color': prev_state['mean_color'],
            }
            return outputs, state, losses

    def end(self):
        losses = {}
        with tf.name_scope('summary'):
            for key in self._info:
                _image_sequence_summary(key, tf.stack(self._info[key], axis=1),
                                        collections=self._image_summaries_collections)
        return losses

    def _preproc(self, im, name='preproc'):
        with tf.name_scope(name) as scope:
            if self._center_input_range:
                im -= 0.5
            if self._keep_uint8_range:
                im *= 255.
            return tf.identity(im, scope)

    def _get_template_rect(self, target):
        target_square = util.coerce_aspect(
            target, im_aspect=self._aspect, aspect_method=self._aspect_method)
        search = geom.grow_rect(self._template_scale, target_square)
        return search, target_square

    def _get_search_rect(self, target):
        target_square = util.coerce_aspect(
            target, im_aspect=self._aspect, aspect_method=self._aspect_method)
        search = geom.grow_rect(self._search_scale, target_square)
        return search, target_square

    def _crop(self, im, rect, im_size, mean_color):
        return util.crop(
            im, rect, im_size,
            pad_value=mean_color if self._pad_with_mean else 0.5,
            feather=self._feather, feather_margin=self._feather_margin)

    def _crop_pyr(self, im, rect, im_size, scales, mean_color):
        return util.crop_pyr(
            im, rect, im_size, scales,
            pad_value=mean_color if self._pad_with_mean else 0.5,
            feather=self._feather, feather_margin=self._feather_margin)

    def _add_motion_prior(self, response, name='motion_prior'):
        with tf.name_scope(name) as scope:
            response_shape = response.shape.as_list()[-3:]
            assert response_shape[-1] == 1
            prior = tf.get_variable('prior', shape=response_shape, dtype=tf.float32,
                                    initializer=tf.zeros_initializer(dtype=tf.float32))
            self._info.setdefault('response_appearance', []).append(
                _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))
            if self._num_frames == 0:
                tf.summary.image(
                    'motion_prior', max_outputs=1,
                    tensor=_to_uint8(util.colormap(tf.sigmoid([motion_prior]), _COLORMAP)),
                    collections=self._image_summaries_collections)
            return response + prior

    def _translation_loss(self, response, response_rf, gt_rect, gt_is_valid, search_rect,
                          name='translation_loss'):
        with tf.name_scope(name) as scope:
            response_size = response.shape.as_list()[-3:-1]
            # Obtain displacement from center of search image.
            disp = util.displacement_from_center(response_size)
            disp = tf.to_float(disp) * response_rf.stride / self._search_size
            # Get centers of receptive field of each pixel.
            centers = 0.5 + disp

            gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
            labels, has_label = lossfunc.translation_labels(
                centers, gt_rect_in_search, shape='gaussian', sigma=self._sigma/self._search_scale)
            # Note: This squeeze will fail if there is an image pyramid (is_tracking=True).
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels,                            # [b, h, w]
                logits=tf.squeeze(response, axis=(1, 4))) # [b, s, h, w, 1] -> [b, h, w]

            if self._balance_classes:
                weights = lossfunc.make_balanced_weights(labels, has_label, axis=(-2, -1))
            else:
                weights = lossfunc.make_uniform_weights(has_label, axis=(-2, -1))
            loss = tf.reduce_sum(weights * loss, axis=(1, 2))

            # TODO: Is this the best way to handle gt_is_valid?
            # (set loss to zero and include in mean)
            loss = tf.where(gt_is_valid, loss, tf.zeros_like(loss))
            loss = tf.reduce_mean(loss)
            return loss, labels

    def _finalize_scores(self, response, stride, name='finalize_scores'):
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
            if self._hann_method == 'add_logit':
                response += self._hann_coeff * tf.expand_dims(hann_2d(upsample_size), -1)
                response = tf.sigmoid(response)
            elif self._hann_method == 'mul_prob':
                response = tf.sigmoid(response)
                response *= tf.expand_dims(hann_2d(upsample_size), -1)
            elif self._hann_method == 'none' or not self._hann_method:
                response = tf.sigmoid(response)
            else:
                raise ValueError('unknown hann method: {}'.format(self._hann_method))
            return response

    def _visualize_response(self, response, response_final, search_im, response_rf,
                            im, search_rect, name='visualize_response'):
        with tf.name_scope(name) as scope:
            response_size = response.shape.as_list()[-3:-1]
            rf_centers = _find_rf_centers(self._search_size, response_size, response_rf)

            # response is logits
            response_cmap = util.colormap(tf.sigmoid(response), _COLORMAP)
            self._info.setdefault('response', []).append(_to_uint8(response_cmap))
            # Draw coarse response over search image.
            response_in_search = _align_corner_centers(rf_centers, response_size)
            self._info.setdefault('response_in_search', []).append(_to_uint8(_paste_image_at_rect(
                search_im, response_cmap, response_in_search, alpha=0.5)))

            # response_final is probability
            response_final_cmap = util.colormap(response_final, _COLORMAP)
            self._info.setdefault('response_final', []).append(_to_uint8(response_final_cmap))
            # Draw upsample, regularized response over original image.
            upsample_response_size = response_final.shape.as_list()[-3:-1]
            response_final_in_search = _align_corner_centers(rf_centers, upsample_response_size)
            response_final_in_image = geom.crop_rect(response_final_in_search, geom.crop_inverse(search_rect))
            # TODO: How to visualize multi-scale responses?
            response_final_in_image = _paste_image_at_rect(
                im, response_final_cmap, response_final_in_image, alpha=0.5)
            self._info.setdefault('response_final_in_image', []).append(_to_uint8(response_final_in_image))

            return response_final_in_image


def _image_sequence_summary(name, tensor, **kwargs):
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
                 arch='alexnet', enable_bnorm=True,
                 is_training=None, name='feature_net'):
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
        x, rfs = _feature_net(x, rfs=rfs, padding=padding, arch=arch,
                              enable_bnorm=enable_bnorm, is_training=is_training, name=name)
        x = restore(x, 0)
        return x, rfs

    with tf.name_scope(name) as scope:
        args = {}
        if enable_bnorm:
            args.update(dict(normalizer_fn=slim.batch_norm,
                             normalizer_params=dict(is_training=is_training)))
        with slim.arg_scope([slim.conv2d], **args):
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
                x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], padding=padding, scope='conv5')
                x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
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
                    x, rfs = util.conv2d_rf(x, rfs, 256, [3, 3], 1, padding=padding, scope='conv5')
                    x, rfs = util.max_pool2d_rf(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
            else:
                raise ValueError('unknown architecture: {}'.format(arch))
            return x, rfs


def _output_net(x, name='output'):
    with tf.name_scope(name) as scope:
        x, restore = merge_dims(x, 0, 2)
        x = slim.conv2d(x, 1, [1, 1], 1, padding='VALID', activation_fn=None)
        x = restore(x, axis=0)
        return x


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

def _affine_scalar(x, name='affine'):
    with tf.name_scope(name) as scope:
        gain = tf.get_variable('gain', shape=[], dtype=tf.float32)
        bias = tf.get_variable('bias', shape=[], dtype=tf.float32)
        return gain * x + bias
