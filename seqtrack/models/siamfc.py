from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import functools
import math
import pprint

import logging
logger = logging.getLogger(__name__)

from seqtrack import cnn
from seqtrack import geom
from seqtrack import helpers
from seqtrack import lossfunc
from seqtrack import models
from seqtrack import receptive_field

from . import interface as models_interface
from . import util
from . import feature_nets
from . import join_nets

from seqtrack.helpers import merge_dims
from seqtrack.helpers import expand_dims_n
from seqtrack.helpers import weighted_mean
from seqtrack.helpers import normalize_prob
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

_COLORMAP = 'viridis'


class SiamFC(models_interface.IterModel):

    def __init__(
            self,
            use_desired_size=False,
            # If use_desired_size is False:
            template_size=127,
            search_size=255,
            template_scale=2,
            # If use_desired_size is True:
            target_size=64,
            desired_template_scale=2.0,
            desired_search_radius=1.0,
            # End of size args.
            aspect_method='perimeter',  # TODO: Equivalent to SiamFC?
            use_gt=True,
            curr_as_prev=True,  # Extract centered examples?
            pad_with_mean=True,  # Use mean of first image for padding.
            feather=False,
            feather_margin=0.1,
            center_input_range=True,  # Make range [-0.5, 0.5] instead of [0, 1].
            keep_uint8_range=False,  # Use input range of 255 instead of 1.
            feature_arch='alexnet',
            feature_arch_params=None,
            join_type='single',  # Either 'single' or 'multi'
            join_arch='xcorr',
            join_params=None,
            multi_join_layers=None,
            feature_model_file='',
            # feature_act='linear',
            # enable_feature_bnorm=True,
            # template_mask_kind='none',  # none, static, dynamic
            # xcorr_padding='VALID',
            # bnorm_after_xcorr=True,
            freeze_siamese=False,
            learnable_prior=False,
            train_multiscale=False,
            # Tracking parameters:
            search_method='local',
            global_search_min_resolution=64,
            global_search_max_resolution=512,
            global_search_num_scales=4,  # 64, 128, 256, 512
            num_scales=5,
            scale_step=1.03,
            scale_update_rate=1,
            report_square=False,
            window_params=None,
            window_radius=1.0,
            arg_max_eps=0.0,
            # Loss parameters:
            wd=0.0,
            loss_params=None,
            ):

        if use_desired_size:
            template_size, search_size, template_scale = dimensions(
                target_size=target_size,
                desired_template_scale=desired_template_scale,
                desired_search_radius=desired_search_radius,
                arch=feature_arch, arch_params=feature_arch_params)
        else:
            # template_size = template_scale * target_size
            target_size = template_size / template_scale

        self._target_size = target_size
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
        self._feature_arch = feature_arch
        self._feature_arch_params = feature_arch_params
        self._feature_model_file = feature_model_file
        self._join_type = join_type
        self._join_arch = join_arch
        self._join_params = join_params or {}
        self._multi_join_layers = multi_join_layers or []
        self._freeze_siamese = freeze_siamese
        self._learnable_prior = learnable_prior
        self._train_multiscale = train_multiscale
        self._search_method = search_method
        self._global_search_min_resolution = global_search_min_resolution
        self._global_search_max_resolution = global_search_max_resolution
        self._global_search_num_scales = global_search_num_scales
        self._num_scales = num_scales
        self._scale_step = scale_step
        self._scale_update_rate = float(scale_update_rate)
        self._report_square = report_square
        self._window_params = window_params or {}
        self._window_radius = window_radius
        self._arg_max_eps = arg_max_eps
        self._wd = wd
        self._loss_params = loss_params or {}

        self._num_frames = 0
        # For summaries in end():
        self._info = {}

        self._feature_saver = None

    def derived_properties(self):
        return dict(
            target_size=self._target_size,
            template_size=self._template_size,
            search_size=self._search_size,
            template_scale=self._template_scale,
        )

    def start(self, frame, aspect, run_opts, enable_loss,
              image_summaries_collections=None, name='start'):
        with tf.name_scope(name) as scope:
            self._aspect = aspect
            self._is_training = run_opts['is_training']
            self._is_tracking = run_opts['is_tracking']
            self._enable_loss = enable_loss
            self._image_summaries_collections = image_summaries_collections

            y = frame['y']
            mean_color = tf.reduce_mean(frame['x'], axis=(-3, -2), keepdims=True)
            # TODO: frame['image'] and template_im have a viewport
            template_rect, y_square = _get_context_rect(
                y, context_amount=self._template_scale, aspect=self._aspect,
                aspect_method=self._aspect_method)
            if self._report_square:
                y = y_square
            template_im = util.crop(frame['x'], template_rect, self._template_size,
                                    pad_value=mean_color if self._pad_with_mean else 0.5,
                                    feather=self._feather, feather_margin=self._feather_margin)

            self._target_in_template = geom.crop_rect(y, template_rect)

            template_input = _preproc(template_im, center_input_range=self._center_input_range,
                                      keep_uint8_range=self._keep_uint8_range)
            template_input = cnn.as_tensor(template_input, add_to_set=True)
            with tf.variable_scope('features', reuse=False):
                template_feat, template_layers, feature_scope = _branch_net(
                    template_input, self._is_training,
                    trainable=(not self._freeze_siamese),
                    variables_collections=['siamese'],
                    weight_decay=self._wd,
                    arch=self._feature_arch,
                    arch_params=self._feature_arch_params)
                # Get names relative to this scope for loading pre-trained.
                self._feature_vars = _global_variables_relative_to_scope(feature_scope)
            self._template_feat = template_feat
            self._template_layers = template_layers
            rf_template = template_feat.fields[template_input.value]
            template_feat = cnn.get_value(template_feat)
            feat_size = template_feat.shape.as_list()[-3:-1]
            try:
                receptive_field.assert_center_alignment(self._template_size, feat_size, rf_template)
            except AssertionError as ex:
                raise ValueError('template features not centered: {:s}'.format(ex))

            self._feature_saver = tf.train.Saver(self._feature_vars)

            with tf.name_scope('summary'):
                tf.summary.image('template', _to_uint8(template_im[0:1]),
                                 max_outputs=1, collections=self._image_summaries_collections)

            # TODO: Avoid passing template_feat to and from GPU (or re-computing).
            state = {
                'y': tf.identity(y),
                # 'template_feat': tf.identity(template_feat),
                'mean_color': tf.identity(mean_color),
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
            num_scales = tf.cond(tf.logical_or(self._is_tracking, self._train_multiscale),
                                 lambda: self._num_scales, lambda: 1)
            mid_scale = (num_scales - 1) // 2
            scales = util.scale_range(num_scales, self._scale_step)
            search_ims, search_rects = util.crop_pyr(
                frame['x'], search_rect, self._search_size, scales,
                pad_value=prev_state['mean_color'] if self._pad_with_mean else 0.5,
                feather=self._feather, feather_margin=self._feather_margin)

            self._info.setdefault('search', []).append(_to_uint8(search_ims[:, mid_scale]))

            # Extract features, perform search, get receptive field of response wrt image.
            search_input = _preproc(search_ims, center_input_range=self._center_input_range,
                                    keep_uint8_range=self._keep_uint8_range)
            # Convert to cnn.Tensor to track receptive field.
            search_input = cnn.as_tensor(search_input, add_to_set=True)

            with tf.variable_scope('features', reuse=True):
                search_feat, search_layers, _ = _branch_net(
                    search_input, self._is_training,
                    trainable=(not self._freeze_siamese),
                    variables_collections=['siamese'],
                    weight_decay=self._wd,
                    arch=self._feature_arch,
                    arch_params=self._feature_arch_params)
            rf_search = search_feat.fields[search_input.value]
            search_feat_size = search_feat.value.shape.as_list()[-3:-1]
            try:
                receptive_field.assert_center_alignment(self._search_size, search_feat_size, rf_search)
            except AssertionError as ex:
                raise ValueError('search features not centered: {:s}'.format(ex))

            # template_feat = prev_state['template_feat']
            with tf.variable_scope('join', reuse=(self._num_frames >= 1)):
                join_fn = join_nets.BY_NAME[self._join_arch]
                if self._join_type == 'single':
                    response = join_fn(self._template_feat, search_feat, self._is_training,
                                       **self._join_params)
                elif self._join_type == 'multi':
                    response = join_fn(self._template_feat, search_feat, self._is_training,
                                       self._multi_join_layers,
                                       self._template_layers, search_layers, search_input,
                                       **self._join_params)
                else:
                    raise ValueError('unknown join type: "{}"'.format(self._join_type))
            rf_response = response.fields[search_input.value]
            response = cnn.get_value(response)
            response_size = response.shape[-3:-1].as_list()
            try:
                receptive_field.assert_center_alignment(self._search_size, response_size, rf_response)
            except AssertionError as ex:
                raise ValueError('response map not centered: {:s}'.format(ex))

            response = tf.verify_tensor_all_finite(response, 'output of xcorr is not finite')

            # with tf.variable_scope('output', reuse=(self._num_frames > 0)):
            #     if self._bnorm_after_xcorr:
            #         response = slim.batch_norm(response, scale=True, is_training=self._is_training,
            #                                    trainable=(not self._freeze_siamese),
            #                                    variables_collections=['siamese'])
            #     else:
            #         response = _affine_scalar(response, variables_collections=['siamese'])
            #     if self._freeze_siamese:
            #         # TODO: Prevent batch-norm updates as well.
            #         # TODO: Set trainable=False for all variables above.
            #         response = tf.stop_gradient(response)
            #     if self._learnable_prior:
            #         response = _add_learnable_motion_prior(response)

            self._info.setdefault('response', []).append(
                _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))

            losses = {}
            if self._enable_loss:
                loss_name, loss = compute_loss(
                    response, self._target_size / rf_response.stride, **self._loss_params)
                losses[loss_name] = loss

            if self._search_method == 'local':
                # Use pyramid from loss function to obtain position.
                # Get relative translation and scale from response.
                # TODO: Upsample to higher resolution than original image?
                response_resize = cnn.get_value(cnn.upsample(
                    response, rf_response.stride, method=tf.image.ResizeMethod.BICUBIC))
                # TODO: Could have target size be dynamic based on object? Probably overkill.
                response_final = apply_motion_penalty(
                    response_resize, radius=self._window_radius * self._target_size,
                    **self._window_params)
                translation, scale, in_arg_max = util.find_peak_pyr(
                    response_final, scales, eps_abs=self._arg_max_eps)
                # Obtain translation in relative co-ordinates within search image.
                translation = 1 / tf.to_float(self._search_size) * translation
                # Get scalar representing confidence in prediction.
                # Use raw appearance score (before motion penalty).
                confidence = weighted_mean(response_resize, in_arg_max, axis=(-4, -3, -2))
                # Damp the scale update towards 1 (no change).
                # TODO: Should this be in log space?
                scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.
                # Get rectangle in search image.
                prev_target_in_search = geom.crop_rect(prev_rect, search_rect)
                pred_in_search = _rect_translate_scale(prev_target_in_search, translation,
                                                       tf.expand_dims(scale, -1))
                # Move from search back to original image.
                pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))

            elif self._search_method == 'global':
                pred = _global_search()
            else:
                raise ValueError('unknown search method "{}"'.format(self._search_method))

            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            # gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_rect)
            if self._use_gt:
                next_prev_rect = tf.cond(self._is_tracking, lambda: pred, lambda: gt_rect)
            else:
                next_prev_rect = pred

            self._num_frames += 1
            # outputs = {'y': pred, 'vis': vis}
            outputs = {'y': pred, 'score': confidence}
            state = {
                'y': next_prev_rect,
                # 'template_feat': prev_state['template_feat'],
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

    def init(self, sess):
        if self._feature_model_file:
            # TODO: Confirm that all variables were loaded?
            try:
                self._feature_saver.restore(sess, self._feature_model_file)
            except tf.errors.NotFoundError as ex:
                pprint.pprint(tf.contrib.framework.list_variables(self._feature_model_file))
                raise
            # # initialize uninitialized variables
            # vars_uninit = sess.run(tf.report_uninitialized_variables())
            # sess.run(tf.variables_initializer([v for v in tf.global_variables()
            #                                    if v.name.split(':')[0] in vars_uninit]))
            # assert len(sess.run(tf.report_uninitialized_variables())) == 0


def dimensions(target_size=64,
               desired_template_scale=2.0,
               desired_search_radius=1.0,
               # Must be same as constructor:
               arch='alexnet',
               arch_params=None):
    '''
    Returns:
        template_size, search_size, template_scale
    '''
    arch_params = arch_params or {}

    feature_fn = feature_nets.BY_NAME[arch]
    feature_fn = functools.partial(feature_fn, **arch_params)
    field = feature_nets.get_receptive_field(feature_fn)

    def snap(x):
        return helpers.round_lattice(_unique(field.size), _unique(field.stride), x)

    template_size = snap(target_size * desired_template_scale)
    search_size = snap(template_size + 2 * desired_search_radius * target_size)
    # Actual context amount will not be exactly equal to desired after snap.
    template_scale = _unique(template_size) / _unique(target_size)

    logger.debug('template_size %d, search_size %d, template_scale %.3g (desired %.3g)',
                 template_size, search_size, template_scale, desired_template_scale)
    return template_size, search_size, template_scale


# When we implement a multi-layer join,
# we should add a convolution to each intermediate activation.
# Should this happen in the join or the feature function?
# Perhaps it makes more sense to put it in the join function,
# in case different join functions want to do it differently.
# (For example, if we combine multiple join functions, we need multiple convolutions.)

# Should the extra output convolution be added in the join function or feature function?
# If we put it in the join function, then we can adapt it to the size of the template.


def _branch_net(x, is_training, trainable, variables_collections,
                weight_decay=0,
                name='features',
                # Additional arguments:
                arch='alexnet',
                arch_params=None,
                # extra_conv_enable=False,
                # extra_conv_params=None,
                ):
    '''
    Args:
        x: Image of which to compute features. Shape [..., h, w, c]

    Returns:
        Output of network, intermediate layers, variable scope of feature net.
        The variables in the feature scope can be loaded from a pre-trained model.
    '''
    with tf.name_scope(name) as scope:
        arch_params = arch_params or {}
        # extra_conv_params = extra_conv_params or {}
        try:
            func = feature_nets.BY_NAME[arch]
        except KeyError:
            raise ValueError('unknown architecture: {}'.format(arch))

        x = cnn.as_tensor(x)
        num_dims = len(x.value.shape)
        if num_dims > 4:
            merged, unmerge = helpers.merge_dims(x.value, 0, num_dims - 3)
            x = cnn.Tensor(merged, x.fields)

        with tf.variable_scope('feature') as feature_vs:
            x, end_points = func(x, is_training, trainable, variables_collections,
                                 weight_decay=weight_decay,
                                 **arch_params)
        # if extra_conv_enable:
        #     with tf.variable_scope('extra'):
        #         x = _extra_conv(x, is_training, trainable, variables_collections,
        #                         **extra_conv_params)

        if num_dims > 4:
            x = cnn.Tensor(unmerge(x.value, 0), x.fields)
        return x, end_points, feature_vs


def _extra_conv(x, is_training, trainable, variables_collections,
                num_outputs=32,
                kernel_size=3,
                stride=1,
                padding='VALID',
                activation='linear'):
    if not trainable:
        raise NotImplementedError('trainable not supported')

    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope([cnn.slim_conv2d],
                            variables_collections=variables_collections):
            return cnn.slim_conv2d(x, num_outputs, kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   activation_fn=helpers.get_act(activation))


def apply_motion_penalty(scores, radius,
                         normalize_method='mean',
                         window_profile='hann',
                         window_mode='radial',
                         combine_method='mul',
                         combine_lambda=1.0):
    '''
    Args:
        scores: [n, s, h, w, c]
    '''
    assert len(scores.shape) == 5
    scores = _normalize_scores(scores, normalize_method)

    score_size = scores.shape[-3:-1].as_list()
    window = _make_window(score_size, window_profile, radius=radius, mode=window_mode)
    window = tf.expand_dims(window, -1)

    if combine_method == 'mul':
        scores = scores * window
    elif combine_method == 'add':
        scores = scores + combine_lambda * window
    else:
        raise ValueError('unknown combine method "{}"'.format(combine_method))
    return scores


def _normalize_scores(scores, method):
    '''
    Args:
        method: 'none', 'mean', 'sigmoid'
    '''
    assert len(scores.shape) == 5
    if method == 'none':
        pass
    elif method == 'mean':
        min_val = tf.reduce_min(scores, axis=(-4, -3, -2, -1), keepdims=True)
        mean_val = tf.reduce_mean(scores, axis=(-4, -3, -2, -1), keepdims=True)
        scores = (1 / (mean_val - min_val)) * (scores - min_val)
    elif method == 'sigmoid':
        scores = tf.sigmoid(scores)
    else:
        raise ValueError('unknown normalize method "{}"'.format(method))
    return scores


def _make_window(size, profile, radius, mode='radial', name='make_window'):
    '''
    Args:
        size: Spatial dimension of response map.
        radius: Float. Scalar or two-element array.

    Returns:
        Tensor with specified size.
    '''
    with tf.name_scope(name) as scope:
        window_fn = _window_func(profile)
        size = np.array(n_positive_integers(2, size))
        u0, u1 = tf.meshgrid(tf.range(size[0]), tf.range(size[1]), indexing='ij')
        u = tf.stack((u0, u1), axis=-1)
        # Center of pixels 0, ..., size - 1 is (size - 1) / 2.
        center = tf.to_float(tf.constant(size) - 1) / 2
        u = (tf.to_float(u) - center) / tf.to_float(radius)

        if mode == 'radial':
            w = window_fn(tf.sqrt(tf.reduce_sum(tf.square(u), axis=-1)))
        elif mode == 'cartesian':
            w = tf.reduce_prod(window_fn(u), axis=-1)
        else:
            raise ValueError('unknown mode: "{}"'.format(mode))
        return w


def _window_func(profile):
    try:
        fn = {
            'rect': lambda x: _mask(x, tf.ones_like(x)),
            'linear': lambda x: _mask(x, 1 - tf.abs(x)),
            'quadratic': lambda x: _mask(x, 1 - tf.square(x)),
            'cosine': lambda x: _mask(x, tf.cos(math.pi / 2 * x)),
            # 'hann': lambda x: _mask(x, 0.5 * (1 + tf.cos(math.pi * x))),
            'hann': lambda x: _mask(x, 1 - tf.square(tf.sin(math.pi / 2 * x))),
            # The gaussian window is approximately equal to hann near zero:
            # The linear approximation of sin(x) at 0 is x.
            # The linear approximation of exp(x) at 0 is 1 + x.
            # Hence exp(-x^2) should be similar to 1 - sin(x)^2 at 0.
            'gaussian': lambda x: tf.exp(-tf.square(math.pi / 2 * x)),
        }[profile]
    except KeyError:
        raise ValueError('unknown window profile: "{}"'.format(profile))
    return fn


def _mask(x, y):
    '''Sets window to 0 outside [-1, 1].'''
    # Ensure that y is >= 0 (protect against numerical error).
    y = tf.maximum(float(0), y)
    return tf.where(tf.abs(x) <= 1, y, tf.zeros_like(y))


def _find_rf_centers(input_size_hw, output_size_hw, rf, name='rf_centers'):
    input_size_hw = n_positive_integers(2, input_size_hw)
    output_size_hw = n_positive_integers(2, output_size_hw)
    min_pixel, max_pixel = cnnutil.rf_centers_in_input(output_size_hw, rf)
    with tf.name_scope(name) as scope:
        # Switch to (x, y) for rect.
        input_size = tf.to_float(input_size_hw[::-1])
        output_size = tf.to_float(output_size_hw[::-1])
        min_pt = tf.to_float(min_pixel[::-1]) + 0.5  # Take center point of pixel.
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
        return overlay_a * overlay_rgb + (1 - overlay_a) * target


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
        return tf.concat([im, tf.ones(shape[:-1] + [1], im.dtype)], axis=-1)


def _to_uint8(x):
    return tf.image.convert_image_dtype(x, tf.uint8, saturate=True)


def _affine_scalar(x, name='affine', variables_collections=None):
    with tf.name_scope(name) as scope:
        gain = slim.model_variable('gain', shape=[], dtype=tf.float32,
                                   collections=variables_collections)
        bias = slim.model_variable('bias', shape=[], dtype=tf.float32,
                                   collections=variables_collections)
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


def _add_learnable_motion_prior(response, name='motion_prior'):
    with tf.name_scope(name) as scope:
        response_shape = response.shape.as_list()[-3:]
        assert response_shape[-1] == 1
        prior = slim.model_variable('prior', shape=response_shape, dtype=tf.float32,
                                    initializer=tf.zeros_initializer(dtype=tf.float32))
        # self._info.setdefault('response_appearance', []).append(
        #     _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))
        # if self._num_frames == 0:
        #     tf.summary.image(
        #         'motion_prior', max_outputs=1,
        #         tensor=_to_uint8(util.colormap(tf.sigmoid([motion_prior]), _COLORMAP)),
        #         collections=self._image_summaries_collections)
        return response + prior


def compute_loss(scores, target_size, method='sigmoid', params=None):
    params = params or {}
    try:
        loss_fn = {
            'sigmoid': _sigmoid_cross_entropy_loss,
            'max_margin': _max_margin_loss,
            'softmax': _softmax_cross_entropy_loss,
        }[method]
    except KeyError as ex:
        raise ValueError('unknown loss method: "{}"'.format(method))
    name, loss = loss_fn(scores, target_size, **params)
    return name, loss


def _sigmoid_cross_entropy_loss(scores, target_size,
                                balanced=False,
                                pos_weight=1,
                                label_method='hard',
                                label_params=None):
    '''
    Args:
        scores: Tensor with shape [b, 1, h, w, 1]
        target_size: Size of target in `scores` (float).

    Returns:
        Loss name, loss tensor with shape [b].
    '''
    label_params = label_params or {}
    try:
        label_fn = {
            'hard': _hard_labels,
            'gaussian': _wrap_with_spatial_weight(_gaussian_labels),
            'hard_binary': _wrap_with_spatial_weight(_hard_binary_labels),
        }[label_method]
    except KeyError as ex:
        raise ValueError('unknown label shape: "{}"'.format(label_method))

    # Remove the scales dimension.
    # Note: This squeeze will fail if there is an image pyramid (is_tracking is true).
    scores = tf.squeeze(scores, 1)
    score_size = helpers.known_spatial_dim(scores)
    # Obtain displacement from center of search image.
    # Center pixel has zero displacement.
    disp = util.displacement_from_center(score_size)
    disp = (1 / tf.to_float(target_size)) * tf.to_float(disp)
    distance = tf.norm(disp, axis=-1, keepdims=True)

    label_name, labels, weights = label_fn(distance, **label_params)
    loss = lossfunc.normalized_sigmoid_cross_entropy_with_logits(
        logits=scores, targets=tf.broadcast_to(labels, tf.shape(scores)), weights=weights,
        pos_weight=pos_weight, balanced=balanced,
        axis=(-3, -2))
    # Remove singleton channels dimension.
    loss = tf.squeeze(loss, -1)

    # Reflect all parameters that affect magnitude of loss.
    # (Losses with same name are comparable.)
    loss_name = 'sigmoid_labels_{}_balanced_{}_pos_weight_{}'.format(
        label_name, balanced, pos_weight)
    return loss_name, loss


def _hard_labels(r, positive_radius, negative_radius):
    is_pos = (r <= positive_radius)
    # TODO: Could force the minimum distance to be a positive?
    is_neg = (r >= negative_radius)
    # Ensure exclusivity and give priority to positive labels.
    # (Not strictly necessary but just to be explicit.)
    is_neg = tf.logical_and(tf.logical_not(is_pos), is_neg)
    label = tf.to_float(is_pos)
    spatial_weight = tf.to_float(tf.logical_or(is_pos, is_neg))
    name = 'hard_pos_radius_{}_neg_radius_{}'.format(positive_radius, negative_radius)
    return name, label, spatial_weight


def _wrap_with_spatial_weight(fn):
    return functools.partial(_add_spatial_weight, fn)


def _add_spatial_weight(fn, *args, **kwargs):
    '''Calls fn() and additionally returns uniform spatial weights.'''
    name, label = fn(*args, **kwargs)
    return name, label, tf.ones_like(label)


def _hard_binary_labels(r, radius):
    '''
    Does not support un-labelled examples.
    More suitable for use with softmax.
    '''
    is_pos = (r <= radius)
    label = tf.to_float(is_pos)
    name = 'hard_binary_radius{}'.format(radius)
    return name, label


def _gaussian_labels(r, sigma):
    label = tf.exp(-tf.square(r) / (2 * tf.square(tf.to_float(sigma))))
    name = 'gaussian_sigma_{}'.format(str(sigma))
    return name, label


def _max_margin_loss(scores, target_size, cost_method='distance_greater', cost_params=None):
    '''
    Args:
        scores: Tensor with shape [b, 1, h, w, 1]
        target_size: Size of target in `scores` (float).

    Returns:
        Loss name, loss tensor with shape [b].
    '''
    cost_params = cost_params or {}
    try:
        cost_fn = {
            'distance_greater': _cost_distance_threshold,
        }[cost_method]
    except KeyError as ex:
        raise ValueError('unknown cost method: "{}"'.format(cost_method))

    # TODO: Avoid copying this block?
    # Remove the scales dimension.
    # Note: This squeeze will fail if there is an image pyramid (is_tracking is true).
    scores = tf.squeeze(scores, 1)
    score_size = helpers.known_spatial_dim(scores)
    # Obtain displacement from center of search image.
    # Center pixel has zero displacement.
    pixel_disp = util.displacement_from_center(score_size)
    disp = (1 / tf.to_float(target_size)) * tf.to_float(pixel_disp)
    distance = tf.norm(disp, axis=-1, keepdims=True)

    cost_name, costs = cost_fn(distance, **cost_params)
    is_center = tf.reduce_all(tf.abs(pixel_disp) <= 0, axis=-1, keepdims=True)
    correct_score = weighted_mean(scores, is_center, axis=(-3, -2), keepdims=False)
    # We want to achieve a margin:
    #   correct_score >= scores + costs
    #   costs + scores - correct_score <= 0
    #   costs - (correct_score - scores) <= 0
    # Therefore penalize max(0, this expr).
    # max_u (costs[u] + scores[u]) - correct_score
    loss = tf.reduce_max(costs + scores, axis=(-3, -2), keepdims=False) - correct_score
    loss = tf.maximum(float(0), loss)
    # If we instead use cost to get `correct_score`:
    # is_best = tf.to_float(cost <= tf.reduce_min(cost, axis=axis, keepdims=True))
    # cost_best = weighted_mean(cost, is_best, axis=axis, keepdims=True)
    # score_best = weighted_mean(score, is_best, axis=axis, keepdims=True)
    # violation = tf.maximum(0., (cost - cost_best) - (score_best - score))
    loss = tf.squeeze(loss, -1)

    loss_name = 'max_margin_cost_{}'.format(cost_name)
    return loss_name, loss


def _cost_distance_threshold(r, threshold):
    is_neg = (r >= threshold)
    cost_name = 'distance_greater_threshold_{}'.format(str(threshold))
    return cost_name, tf.to_float(is_neg)


def _softmax_cross_entropy_loss(scores, target_size,
                                label_method='gaussian',
                                label_params=None):
    '''
    Args:
        scores: Tensor with shape [b, 1, h, w, 1]
        target_size: Size of target in `scores` (float).

    Returns:
        Loss name, loss tensor with shape [b].
    '''
    label_params = label_params or {}
    try:
        label_fn = {
            'hard_binary': _hard_binary_labels,
            'gaussian': _gaussian_labels,
        }[label_method]
    except KeyError as ex:
        raise ValueError('unknown label shape: "{}"'.format(label_method))

    # Remove the scales dimension.
    # Note: This squeeze will fail if there is an image pyramid (is_tracking is true).
    scores = tf.squeeze(scores, 1)
    score_size = helpers.known_spatial_dim(scores)
    # Obtain displacement from center of search image.
    # Center pixel has zero displacement.
    disp = util.displacement_from_center(score_size)
    disp = (1 / tf.to_float(target_size)) * tf.to_float(disp)
    distance = tf.norm(disp, axis=-1, keepdims=True)

    label_name, labels = label_fn(distance, **label_params)
    labels = helpers.normalize_prob(labels, axis=(-3, -2))

    # Flatten spatial dimensions.
    scores, _ = helpers.merge_dims(scores, -3, -1)
    labels, _ = helpers.merge_dims(labels, -3, -1)
    # Remove singleton channel dimension.
    scores = tf.squeeze(scores, -1)
    labels = tf.squeeze(labels, -1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=scores, labels=tf.broadcast_to(labels, tf.shape(scores)), dim=-1)

    # Reflect all parameters that affect magnitude of loss.
    # (Losses with same name are comparable.)
    loss_name = 'softmax_labels_{}'.format(label_name)
    return loss_name, loss


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


def _clip_rect_size(rect, min_size=None, max_size=None, name='clip_rect_size'):
    with tf.name_scope(name) as scope:
        center, size = geom.rect_center_size(rect)
        if max_size is not None:
            size = tf.minimum(size, max_size)
        if min_size is not None:
            size = tf.maximum(size, min_size)
        return geom.make_rect_center_size(center, size)


def _center_rect_with_aspect(aspect, name='center_rect_with_aspect'):
    '''
    Args:
        aspect -- [b]

    Returns:
        [b, 4]
    '''
    with tf.name_scope(name) as scope:
        # From definition, aspect = width / height; width = aspect * height
        # Want max(width, height) = 1.
        width, height = aspect, tf.ones_like(aspect)
        max_dim = tf.maximum(width, height)
        width, height = width / max_dim, height / max_dim
        size = tf.stack((width, height), axis=-1)
        return geom.make_rect_center_size(0.5 * tf.ones_like(size), size)


def _position_grid(size):
    size_y, size_x = size
    # Index from 0 to 1 since corners are aligned.
    range_y = tf.to_float(tf.range(size_y)) / float(size_y - 1)
    range_x = tf.to_float(tf.range(size_x)) / float(size_x - 1)
    grid_y, grid_x = tf.meshgrid(range_y, range_x, indexing='ij')
    return tf.stack((grid_x, grid_y), axis=-1)


def _merge_dims_and_concat(xs, a, b):
    pairs = list(map(lambda x: merge_dims(x, a, b), xs))
    xs, restores = zip(*pairs)  # Inverse of zip is zip-star.
    ns = [x.shape.as_list()[a] for x in xs]
    xs = tf.concat(xs, axis=a)

    def _restore(ys):
        ys = tf.split(ys, ns, axis=a)
        ys = [r(y, axis=a) for y, r in zip(ys, restores)]
        return ys

    return xs, _restore


def _global_variables_relative_to_scope(scope):
    '''
    Args:
        scope: VariableScope
    '''
    prefix = scope.name + '/'
    # tf.Saver uses var.op.name to get variable name.
    # https://stackoverflow.com/a/36156697/1136018
    return {_remove_prefix(prefix, v.op.name): v for v in scope.global_variables()}


def _remove_prefix(prefix, x):
    if not x.startswith(prefix):
        raise ValueError('does not have prefix "{}": "{}"'.format(prefix, x))
    return x[len(prefix):]


def _unique(elems):
    if len(np.shape(elems)) == 0:
        return elems
    assert len(elems) > 0
    value = elems[0]
    assert all(x == value for x in elems)
    return value


def _global_search():
    # TODO: Restore this functionality.
    raise NotImplementedError('legacy code')

    # Construct new search pyramid for entire image.
    # Careful! Any summaries or update opts (e.g. bnorm) can cause
    # this to be evaluated unnecessarily during training.

    # TODO: Determine automatically!!
    # Receptive field of feature transform is 87 at stride 8.
    # Feature template is size 5.
    rf_edge = 87 + 8 * (5 - 1)
    rf_half = (rf_edge - 1) / 2
    rf_stride = 8
    # TODO: Assert that these are correct.

    # Resize the image to multiple sizes.
    # The image itself takes up `valid_edge` pixels within an image of size `edge`.
    ideal_edges = np.geomspace(self._global_search_min_resolution,
                               self._global_search_max_resolution,
                               self._global_search_num_scales)
    valid_edges = [
        int(round(float(ideal_edge - 1) / rf_stride)) * rf_stride + 1
        for ideal_edge in ideal_edges]
    assert len(set(valid_edges)) == len(valid_edges)
    logger.debug('ideal edges: %s', ['{:.1f}'.format(edge) for edge in ideal_edges])
    logger.debug('valid edges: %s', valid_edges)
    edges = [2 * rf_half + valid_edge for valid_edge in valid_edges]
    # The image is a rectangle with the correct aspect ratio within a square.
    # The square has size `valid_edge` within a larger padded image of size `edge`.
    # TODO: Use a random region within the square.
    im_in_square = _center_rect_with_aspect(self._aspect)
    square_in_padded = [
        geom.make_rect_center_size(
            0.5 * tf.ones(2, tf.float32),
            (float(valid_edges[i]) / edges[i]) * tf.ones(2, tf.float32))
        for i in range(self._global_search_num_scales)]
    # Get aspect within inset rect.
    im_in_padded = [
        geom.crop_rect(im_in_square, geom.crop_inverse(square_in_padded[i]))
        for i in range(self._global_search_num_scales)]
    search_im = [
        util.crop(frame['x'], geom.crop_inverse(im_in_padded[i]), (edges[i], edges[i]),
                  pad_value=prev_state['mean_color'] if self._pad_with_mean else 0.5,
                  feather=self._feather, feather_margin=self._feather_margin)
        for i in range(self._global_search_num_scales)]

    # Extract features, perform search, get receptive field of response wrt image.
    search_input = [
        _preproc(search_im[i], center_input_range=self._center_input_range,
                 keep_uint8_range=self._keep_uint8_range)
        for i in range(self._global_search_num_scales)]

    rfs = [{'search': cnnutil.identity_rf()} for _ in edges]
    search_feat = [None for _ in edges]
    for i in range(self._global_search_num_scales):
        with tf.variable_scope('features', reuse=True):
            search_feat[i], rfs[i] = _branch_net(
                search_input[i], rfs[i], padding=self._feature_padding, arch=self._feature_arch,
                output_act=self._feature_act, enable_bnorm=self._enable_feature_bnorm,
                wd=self._wd, is_training=self._is_training,
                # variables_collections=['siamese'], trainable=(not self._freeze_siamese))
                variables_collections=['siamese'], trainable=False)

    response = [None for _ in edges]
    for i in range(self._global_search_num_scales):
        response[i], rfs[i] = util.diag_xcorr_rf(
            input=search_feat[i], filter=prev_state['template_feat'], input_rfs=rfs[i],
            padding=self._xcorr_padding)
        response[i] = tf.reduce_sum(response[i], axis=-1, keepdims=True)
        cnnutil.assert_center_alignment(n_positive_integers(2, edges[i]),
                                        response[i].shape.as_list()[-3:-1],
                                        rfs[i]['search'])

    # Compute batch-norm of all responses together.
    # This requires concatenating into a large vector.
    response, unvec = _merge_dims_and_concat(response, 1, 3)

    # with tf.variable_scope('output', reuse=(self._num_frames > 0)):
    with tf.variable_scope('output', reuse=True):
        if self._bnorm_after_xcorr:
            response = slim.batch_norm(response, scale=True, is_training=self._is_training,
                                       # trainable=(not self._freeze_siamese),
                                       trainable=False,
                                       variables_collections=['siamese'])
        else:
            response = _affine_scalar(response, variables_collections=['siamese'])
        if self._freeze_siamese:
            # TODO: Prevent batch-norm updates as well.
            # TODO: Set trainable=False for all variables above.
            response = tf.stop_gradient(response)
        # if self._learnable_prior:
        #     response = _add_learnable_motion_prior(response)

    # Return from super-vector to separate images per scale.
    response = unvec(response)

    # Compute the rectangle for each position.
    # Size of object at each scale.
    # Given size of object in template and resolution of template.
    size_in_template = geom.rect_size(self._target_in_template)
    # TODO: Original image covers `valid_edge` pixels.
    # sizes = (size_in_template *
    #     tf.expand_dims(float(self._template_size) / tf.to_float(valid_edges), axis=-1))
    # scales = float(self._template_size) / np.asfarray(valid_edges)
    size = [
        (float(self._template_size) / valid_edges[i]) * size_in_template
        for i in range(self._global_search_num_scales)]
    # Center of receptive field at each location.
    # TODO: The corners at 0 and 1 are off by half a pixel?
    centers = [
        _position_grid(response[i].shape.as_list()[-3:-1])
        for i in range(self._global_search_num_scales)]
    rects = [
        geom.make_rect_center_size(centers[i], expand_dims_n(size[i], -2, 2))
        for i in range(self._global_search_num_scales)]

    # for i in range(self._global_search_num_scales):
    #     self._info.setdefault('search/scale_{}'.format(i), []).append(_to_uint8(
    #         search_im[i]))
    #     response_cmap = util.colormap(tf.sigmoid(response[i]), _COLORMAP)
    #     self._info.setdefault('response/scale_{}'.format(i), []).append(_to_uint8(
    #         response_cmap))
    #     self._info.setdefault('response_overlay/scale_{}'.format(i), []).append(_to_uint8(
    #         _visualize_response(search_im[i], response_cmap, rfs[i]['search'])))

    # TODO: Add multi-scale loss here?

    # Upsample all responses to same size.
    response_size = response[-1].shape.as_list()[-3:-1]
    response_resize = [
        tf.image.resize_images(
            response[i], response_size,
            method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        if i < self._global_search_num_scales - 1 else response[i]
        for i in range(self._global_search_num_scales)]
    response_resize = tf.stack(response_resize, axis=1)

    # TODO: Unify this code with above `rects`.
    # Find max over all responses.
    response_sigmoid = tf.sigmoid(response_resize)
    is_max = tf.to_float(util.is_peak(response_sigmoid, axis=(-4, -3, -2),
                                      eps_rel=self._arg_max_eps_rel))
    score = weighted_mean(response_resize, is_max, axis=(-4, -3, -2))
    sizes = tf.stack(size, axis=1)
    pred_size = weighted_mean(expand_dims_n(sizes, -2, 2), is_max, axis=(-4, -3, -2))
    center_grid = _position_grid(response_resize.shape.as_list()[-3:-1])
    pred_center = weighted_mean(tf.expand_dims(center_grid, -4), is_max, axis=(-4, -3, -2))
    pred = geom.make_rect_center_size(pred_center, pred_size)
    # Move back from square image to valid part.
    pred = geom.crop_rect(pred, im_in_square)
    return pred
