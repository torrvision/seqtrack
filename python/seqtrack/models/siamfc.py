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


IMAGE_SUMMARIES_COLLECTIONS = ['image_summaries']
MODE_KEYS_SUPERVISED = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]


def default_params():
    return dict(
        target_size=64,
        template_size=127,
        search_size=255,
        template_method='init',
        template_prev_weight=0.0,
        aspect_method='perimeter',  # TODO: Equivalent to SiamFC?
        pad_with_mean=True,  # Use mean of first image for padding.
        feather=False,
        feather_margin=0.1,
        center_input_range=True,  # Make range [-0.5, 0.5] instead of [0, 1].
        keep_uint8_range=False,  # Use input range of 255 instead of 1.
        feature_arch='alexnet',
        feature_arch_params=None,
        feature_extra_conv_enable=False,
        feature_extra_conv_params=None,
        join_type='single',  # Either 'single' or 'multi'
        join_arch='xcorr',
        join_params=None,
        multi_join_layers=None,
        feature_model_file='',
        # template_mask_kind='none',  # none, static, dynamic
        learn_appearance=True,
        learn_motion=False,
        use_predictions=True,  # Use predictions for previous positions?
        train_multiscale=False,
        # Tracking parameters:
        search_method='local',
        # global_search_min_resolution=64,
        # global_search_max_resolution=512,
        # global_search_num_scales=4,  # 64, 128, 256, 512
        num_scales=5,
        scale_step=1.03,
        scale_update_rate=1,
        report_square=False,
        window_params=None,
        window_radius=1.0,
        arg_max_eps=0.0,
        # Loss parameters:
        wd=0.0,
        loss_params=None,  # kwargs for compute_loss()
    )


class SiamFC(object):
    '''Instantiates the TF graph.

    Use either `train` or `start` and `next`.
    '''

    def __init__(self, mode, params):
        self.mode = mode

        defaults = default_params()
        helpers.update_existing_keys(defaults, params)
        for key, value in params.items():
            assert not hasattr(self, key)
            setattr(self, key, value)

        # Size of search area relative to object.
        self.template_scale = float(self.template_size) / self.target_size
        self.search_scale = float(self.search_size) / self.target_size
        # Defaults instead of None.
        self.join_params = self.join_params or {}
        self.multi_join_layers = self.multi_join_layers or []
        self.window_params = self.window_params or {}
        self.loss_params = self.loss_params or {}
        # Ensure types are correct.
        self.scale_update_rate = float(self.scale_update_rate)

        self._num_frames = 0
        # For summaries in end():
        self._info = {}

        # self._feature_saver = None

    def _context_rect(self, rect, aspect, amount):
        template_rect, _ = _get_context_rect(
            rect,
            context_amount=amount,
            aspect=aspect,
            aspect_method=self.aspect_method)
        return template_rect

    def _crop(image, rect, size, mean_color):
        return util.crop(
            image, rect, size,
            pad_value=mean_color if self.pad_with_mean else 0.5,
            feather=self.feather,
            feather_margin=self.feather_margin)

    def _crop_pyr(image, rect, size, mean_color):
        return util.crop_pyr(
            image, rect, size, scales,
            pad_value=mean_color if self.pad_with_mean else 0.5,
            feather=self.feather,
            feather_margin=self.feather_margin)

    def _preproc(self, im):
        return _preproc(
            im,
            center_input_range=self.center_input_range,
            keep_uint8_range=self.keep_uint8_range)

    def _embed_net(self, im, is_training):
        return _embed_net(
            im, is_training,
            trainable=self.learn_appearance,
            variables_collections=['siamese'],
            weight_decay=self.wd,
            arch=self.feature_arch,
            arch_params=self.feature_arch_params,
            extra_conv_enable=self.feature_extra_conv_enable,
            extra_conv_params=self.feature_extra_conv_params)

    def start(self, features_init, run_opts, name='start'):
        with tf.name_scope(name) as scope:
            im = features_init['image']['data']
            aspect = features_init['aspect']
            target_rect = features_init['rect']
            mean_color = tf.reduce_mean(im, axis=(-3, -2), keepdims=True)

            template_rect = self._context_rect(target_rect, aspect, self.template_scale)
            template_im = self._crop(im, template_rect, self.template_size, mean_color)
            template_input = self._preproc(template_im)
            template_input = cnn.as_tensor(template_input, add_to_set=True)
            with tf.variable_scope('embed', reuse=False):
                template_feat, template_layers, feature_scope = self._embed_net(
                    template_input, run_opts['is_training'])
                # Get names relative to this scope for loading pre-trained.
                self._feature_vars = _global_variables_relative_to_scope(feature_scope)
            rf_template = template_feat.fields[template_input.value]
            template_feat = cnn.get_value(template_feat)
            feat_size = template_feat.shape[-3:-1].as_list()
            receptive_field.assert_center_alignment(self.template_size, feat_size, rf_template)

            # self._feature_saver = tf.train.Saver(self._feature_vars)

            with tf.name_scope('summary'):
                tf.summary.image('template', _to_uint8(template_im[0:1]),
                                 max_outputs=1,
                                 collections=IMAGE_SUMMARIES_COLLECTIONS)

            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                'rect': tf.identity(target_rect),
                'template_init': tf.identity(template_feat),
                'mean_color': tf.identity(mean_color),
            }
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                state['template_prev'] = tf.identity(template_feat)
            return state

    def next(self, features, labels, state, name='timestep', reset_position=False):
        '''
        Args:
            reset_position: Keep the appearance model but reset the position.
                If this is true, then features['rect'] must be present.
        '''
        with tf.name_scope(name) as scope:
            im = features['image']['data']
            run_opts = state['run_opts']
            aspect = state['aspect']
            mean_color = state['mean_color']

            # If the label is not valid, there will be no loss for this frame.
            # However, the input image may still be processed.
            # Adopt the previous rectangle as the "ground-truth".
            if mode in MODE_KEYS_SUPERVISED:
                last_valid_gt_rect = tf.where(labels['valid'], labels['rect'], state['rect'])
            else:
                last_valid_gt_rect = None
            # When learning motion or tracking, use the previous rectangle.
            # Otherwise adopt the ground-truth as the previous rectangle
            # so that the object is in the center of the search area.
            if self.learn_motion or self.mode == tf.estimator.ModeKeys.PREDICT:
                # Use previous position from state.
                prev_target_rect = state['rect']
            else:
                # Use ground-truth rectangle as previous position
                prev_target_rect = last_valid_gt_rect

            # How to obtain template from previous state?
            if self.template_method == 'init':
                template_feat = state['template_init']
            elif self.template_method == 'convex_init_prev':
                # During training, still use the initial template.
                # During tracking, take a combination of the previous and initial templates.
                if self.mode in MODE_KEYS_SUPERVISED:
                    template_feat = state['template_init']
                else:
                    # TODO: This will not work with "multi" joins yet.
                    # TODO: Could instead combine after computing response.
                    # (May be more important with fully-connected output stage.)
                    template_feat = (self.template_prev_weight * state['template_prev'] +
                                     (1 - self.template_prev_weight) * state['template_init'])
            else:
                raise ValueError('unknown template method: "{}"'.format(self.template_method))

            # Coerce the aspect ratio of the rectangle to construct the search area.
            search_rect = self._context_rect(prev_target_rect, aspect, self.search_scale)
            # Extract an image pyramid (use 1 scale when not in tracking mode).
            if self.train_multiscale or self.mode == tf.estimator.ModeKeys.PREDICT:
                num_scales = self.num_scales
            else:
                num_scales = 1
            mid_scale = (num_scales - 1) // 2
            scales = util.scale_range(num_scales, tf.to_float(self.scale_step))
            search_ims, search_rects = self._crop_pyr(
                im, search_rect, self.search_size, self._scales, mean_color)

            self._info.setdefault('search', []).append(_to_uint8(search_ims[:, mid_scale]))

            # Extract features, perform search, get receptive field of response wrt image.
            search_input = self._preproc(search_ims)
            search_input = cnn.as_tensor(search_input, add_to_set=True)
            with tf.variable_scope('embed', reuse=True):
                search_feat, search_layers, _ = self._embed_net(
                    search_input, run_opts['is_training'])
            rf_search = search_feat.fields[search_input.value]
            search_feat_size = search_feat.value.shape[-3:-1].as_list()
            receptive_field.assert_center_alignment(self.search_size, search_feat_size, rf_search)

            with tf.variable_scope('join', reuse=(self._num_frames >= 1)):
                join_fn = join_nets.BY_NAME[self.join_arch]
                if self.join_type == 'single':
                    response = join_fn(template_feat, search_feat, run_opts['is_training'],
                                       **self.join_params)
                elif self.join_type == 'multi':
                    response = join_fn(template_feat, search_feat, run_opts['is_training'],
                                       self.multi_join_layers,
                                       template_layers, search_layers, search_input,
                                       **self.join_params)
                else:
                    raise ValueError('unknown join type: "{}"'.format(self.join_type))
            rf_response = response.fields[search_input.value]
            response = cnn.get_value(response)
            response_size = response.shape[-3:-1].as_list()
            receptive_field.assert_center_alignment(self.search_size, response_size, rf_response)

            response = tf.verify_tensor_all_finite(response, 'output of xcorr is not finite')

            # # Post-process scores.
            # with tf.variable_scope('output', reuse=(self._num_frames > 0)):
            #     if self.bnorm_after_xcorr:
            #         response = slim.batch_norm(response, scale=True, is_training=self.is_training,
            #                                    trainable=self.learn_appearance,
            #                                    variables_collections=['siamese'])
            #     else:
            #         response = _affine_scalar(response, variables_collections=['siamese'])
            #     if not self.learn_appearance:
            #         # TODO: Prevent batch-norm updates as well.
            #         # TODO: Set trainable=False for all variables above.
            #         response = tf.stop_gradient(response)
            #     if self.learn_motion:
            #         response = _add_learnable_motion_prior(response)

            self._info.setdefault('response', []).append(
                _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))

            losses = {}
            if self.mode in MODE_KEYS_SUPERVISED:
                loss_name, loss = compute_loss(
                    response, self.target_size / rf_response.stride, **self.loss_params)
                losses[loss_name] = loss

            if self.search_method == 'local':
                # Use pyramid from loss function to obtain position.
                # Get relative translation and scale from response.
                # TODO: Upsample to higher resolution than original image?
                response_resize = cnn.get_value(cnn.upsample(
                    response, rf_response.stride, method=tf.image.ResizeMethod.BICUBIC))
                response_final = apply_motion_penalty(
                    response_resize, radius=self.window_radius * self.target_size,
                    **self.window_params)
                translation, scale, in_arg_max = util.find_peak_pyr(
                    response_final, self._scales, eps_abs=self.arg_max_eps)
                # Obtain translation in relative co-ordinates within search image.
                translation = 1 / tf.to_float(self.search_size) * translation
                # Get scalar representing confidence in prediction.
                # Use raw appearance score (before motion penalty).
                confidence = weighted_mean(response_resize, in_arg_max, axis=(-4, -3, -2))
                # Damp the scale update towards 1 (no change).
                # TODO: Should this be in log space?
                scale = self.scale_update_rate * scale + (1. - self.scale_update_rate) * 1.
                # Get rectangle in search image.
                prev_target_in_search = geom.crop_rect(prev_target_rect, search_rect)
                pred_in_search = _rect_translate_scale(prev_target_in_search, translation,
                                                       tf.expand_dims(scale, -1))
                # Move from search back to original image.
                pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))
            else:
                raise ValueError('unknown search method "{}"'.format(self.search_method))

            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            if tf.estimator.ModeKeys.PREDICT or self.use_predictions:
                next_prev_rect = pred
            else:
                next_prev_rect = last_valid_gt_rect

            self._num_frames += 1
            outputs = {'rect': pred, 'score': confidence}
            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                'rect': next_prev_rect,
                'template_init': state['template_init'],
                'mean_color': state['mean_color'],
            }

            # If required, extract new template at current position and add to state.
            if (self.template_method == 'convex_init_prev' and
                    self.mode == tf.estimator.ModeKeys.PREDICT):
                template_rect = self._context_rect(target_rect, aspect, self.template_scale)
                template_im = self._crop(im, template_rect, self.template_size, mean_color)
                template_input = self._preproc(template_im)
                # template_input = cnn.as_tensor(template_input, add_to_set=True)
                with tf.variable_scope('embed', reuse=True):
                    template_feat, template_layers, _ = self._embed_net(
                        template_input, run_opts['is_training'])
                template_feat = cnn.get_value(template_feat)
                state['template_prev'] = template_feat

            return outputs, state, losses

    def end(self):
        losses = {}
        with tf.name_scope('summary'):
            for key in self._info:
                _image_sequence_summary(key, tf.stack(self._info[key], axis=1),
                                        collections=IMAGE_SUMMARIES_COLLECTIONS)
        return losses

    # def init(self, sess):
    #     if self.feature_model_file:
    #         # TODO: Confirm that all variables were loaded?
    #         try:
    #             self._feature_saver.restore(sess, self.feature_model_file)
    #         except tf.errors.NotFoundError as ex:
    #             pprint.pprint(tf.contrib.framework.list_variables(self.feature_model_file))
    #             raise
    #         # # initialize uninitialized variables
    #         # vars_uninit = sess.run(tf.report_uninitialized_variables())
    #         # sess.run(tf.variables_initializer([v for v in tf.global_variables()
    #         #                                    if v.name.split(':')[0] in vars_uninit]))
    #         # assert len(sess.run(tf.report_uninitialized_variables())) == 0


def dimensions(target_size,
               desired_template_scale,
               desired_search_radius,
               # Must be same as constructor:
               feature_arch='alexnet',
               feature_arch_params=None,
               feature_extra_conv_enable=False,
               feature_extra_conv_params=None):
    '''
    Returns: Dict with "template_size", "search_size".
    '''
    field = _branch_net_receptive_field(
        arch=feature_arch,
        arch_params=feature_arch_params,
        extra_conv_enable=feature_extra_conv_enable,
        extra_conv_params=feature_extra_conv_params)

    field_size = helpers.get_unique_value(field.size)
    field_stride = helpers.get_unique_value(field.stride)
    def snap(x):
        return helpers.round_lattice(field_size, field_stride, x)

    template_size = snap(target_size * desired_template_scale)
    search_size = snap(template_size + 2 * desired_search_radius * target_size)
    # Actual context amount will not be exactly equal to desired after snap.
    template_scale = (helpers.get_unique_value(template_size) /
                      helpers.get_unique_value(target_size))

    logger.debug('template_size %d, search_size %d, template_scale %.3g (desired %.3g)',
                 template_size, search_size, template_scale, desired_template_scale)
    # return template_size, search_size, template_scale
    return dict(template_size=template_size, search_size=search_size)


def scale_sequence(num_scales, max_scale):
    '''
    >>> round(scale_sequence(3, 1.02)['scale_step'], 6)
    1.02
    >>> round(scale_sequence(5, 1.02)['scale_step'], 4)
    1.01
    '''
    # TODO: Use log1p and exp1m? Probably not necessary.
    if num_scales == 1:
        # There is no scale step.
        return dict(num_scales=1)
    assert num_scales % 2 == 1
    h = (num_scales - 1) // 2
    # Scales will be:
    #   scales = step ** [-h, ..., h]
    #   log(scales) = log(step) * [-h, ..., h]
    scale_step = math.exp(abs(math.log(max_scale)) / h)
    return dict(num_scales=num_scales, scale_step=scale_step)


# When we implement a multi-layer join,
# we should add a convolution to each intermediate activation.
# Should this happen in the join or the feature function?
# Perhaps it makes more sense to put it in the join function,
# in case different join functions want to do it differently.
# (For example, if we combine multiple join functions, we need multiple convolutions.)

# Should the extra output convolution be added in the join function or feature function?
# If we put it in the join function, then we can adapt it to the size of the template.


def _embed_net(x, is_training, trainable, variables_collections,
               weight_decay=0,
               name='embed',
               # Additional arguments:
               arch='alexnet',
               arch_params=None,
               extra_conv_enable=False,
               extra_conv_params=None):
    '''
    Args:
        x: Image of which to compute features. Shape [..., h, w, c]

    Returns:
        Output of network, intermediate layers, variable scope of feature net.
        The variables in the feature scope can be loaded from a pre-trained model.
    '''
    with tf.name_scope(name) as scope:
        arch_params = arch_params or {}
        extra_conv_params = extra_conv_params or {}
        weight_decay = float(weight_decay)

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

        if extra_conv_enable:
            with tf.variable_scope('extra'):
                x = _extra_conv(x, is_training, trainable, variables_collections,
                                **extra_conv_params)
        if num_dims > 4:
            x = cnn.Tensor(unmerge(x.value, 0), x.fields)
        return x, end_points, feature_vs


def _extra_conv(x, is_training, trainable, variables_collections,
                num_outputs=None,
                kernel_size=1,
                stride=1,
                padding='VALID',
                activation='linear'):
    if not trainable:
        raise NotImplementedError('trainable not supported')

    x = cnn.as_tensor(x)
    if num_outputs is None:
        num_outputs = x.value.shape[-1].value
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
        with slim.arg_scope([cnn.slim_conv2d],
                            variables_collections=variables_collections):
            return cnn.slim_conv2d(x, num_outputs, kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   activation_fn=helpers.get_act(activation))


def _branch_net_receptive_field(arch='alexnet',
                                arch_params=None,
                                extra_conv_enable=False,
                                extra_conv_params=None):
    arch_params = arch_params or {}

    graph = tf.Graph()
    with graph.as_default():
        image = tf.placeholder(tf.float32, (None, None, None, 3), name='image')
        is_training = tf.placeholder(tf.bool, (), name='is_training')
        image = cnn.as_tensor(image, add_to_set=True)
        retvals = _embed_net(image, is_training,
                             trainable=True,
                             variables_collections=None,
                             arch=arch,
                             arch_params=arch_params,
                             extra_conv_enable=extra_conv_enable,
                             extra_conv_params=extra_conv_params)
        feat = retvals[0]
        return feat.fields[image.value]


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


def _normalize_scores(scores, method, eps=1e-3):
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
        scores = (1 / tf.maximum(float(eps), mean_val - min_val)) * (scores - min_val)
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
        #         collections=IMAGE_SUMMARIES_COLLECTIONS)
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
            'distance_greater': _cost_distance_greater,
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
    # Force cost of center label to be zero.
    costs = tf.where(is_center, tf.zeros_like(costs), costs)
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


def _cost_distance_greater(r, threshold):
    is_neg = (r >= threshold)
    cost_name = 'distance_greater_{}'.format(str(threshold))
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
