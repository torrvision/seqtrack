from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import numpy as np
import pprint

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

import logging
logger = logging.getLogger(__name__)

from seqtrack import cnn
from seqtrack import geom
from seqtrack import helpers
from seqtrack import lossfunc
from seqtrack import receptive_field
from seqtrack import sample
from seqtrack.models import itermodel

from . import util
from . import feature_nets
from . import join_nets

IMAGE_SUMMARIES_COLLECTIONS = ['image_summaries']
MODE_KEYS_SUPERVISED = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

CONTEXT_SIZE = 195


def default_params():
    return dict(
        target_size=64,
        aspect_method='perimeter',  # TODO: Equivalent to SiamFC?
        pad_with_mean=True,  # Use mean of first image for padding.
        feather=False,
        feather_margin=0.1,
        center_input_range=True,  # Make range [-0.5, 0.5] instead of [0, 1].
        keep_uint8_range=False,  # Use input range of 255 instead of 1.
        response_stride=8,
        response_size=17,
        num_scales=5,
        scale_step=1.02,
        use_predictions=True,  # Use predictions for previous positions?
        scale_update_rate=1,
        arg_max_eps=0.0,
        # Loss parameters:
        wd=0.0,
        loss_params=None,  # kwargs for compute_loss()
    )


class MotionRegressor(object):
    '''Instantiates the TF graph.

    Use either `train` or `start` and `next`.
    '''

    def __init__(self, mode, params, example_type=None):
        '''
        Args:
            mode: tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
                Losses are not computed in PREDICT mode (no labels).
            example_type: sample.ExampleTypeKeys.{CONSECUTIVE, UNORDERED, SEPARATE_INIT}
        '''
        self.mode = mode
        self.example_type = example_type

        params = helpers.update_existing_keys(default_params(), params)
        for key, value in params.items():
            assert not hasattr(self, key)
            setattr(self, key, value)

        # Size of search area relative to object.
        self.context_scale = float(CONTEXT_SIZE) / self.target_size
        # Defaults instead of None.
        self.loss_params = self.loss_params or {}
        # Ensure types are correct.
        self.scale_update_rate = float(self.scale_update_rate)

        self._num_frames = 0
        # For summaries in end():
        self._info = {}

    def train(self, example, run_opts, scope='model'):
        '''
        Args:
            example: ExampleSequence

        Returns: itermodel.OperationsUnroll
        '''
        if self.example_type == sample.ExampleTypeKeys.CONSECUTIVE:
            # Use the default `instantiate_unroll` method.
            return itermodel.instantiate_unroll(self, example, run_opts=run_opts, scope=scope)
        elif self.example_type == sample.ExampleTypeKeys.SEPARATE_INIT:
            raise NotImplementedError()
        else:
            raise ValueError('unsupported example type: {}'.format(self.example_type))

    def _context_rect(self, rect, aspect, amount):
        context_rect, _ = _get_context_rect(
            rect,
            context_amount=amount,
            aspect=aspect,
            aspect_method=self.aspect_method)
        return context_rect

    def _crop(self, image, rect, size, mean_color):
        return util.crop(
            image, rect, size,
            pad_value=mean_color if self.pad_with_mean else 0.5,
            feather=self.feather,
            feather_margin=self.feather_margin)

    def _crop_pyr(self, image, rect, size, scales, mean_color):
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

    def start(self, features_init, run_opts, name='start'):
        with tf.name_scope(name) as scope:
            im = features_init['image']['data']
            aspect = features_init['aspect']
            target_rect = features_init['rect']
            mean_color = tf.reduce_mean(im, axis=(-3, -2), keepdims=True)

            state = {
                'image': im,
                'run_opts': run_opts,
                'aspect': aspect,
                'rect': tf.identity(target_rect),
                'mean_color': tf.identity(mean_color),
            }
            return state

    def next(self, features, labels, state, name='timestep'):
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
            prev_im = state['image']

            # If the label is not valid, there will be no loss for this frame.
            # However, the input image may still be processed.
            # In this case, adopt the previous rectangle as the "ground-truth".
            if self.mode in MODE_KEYS_SUPERVISED:
                gt_rect = tf.where(labels['valid'], labels['rect'], state['rect'])
            else:
                gt_rect = None
            # Use the previous rectangle.
            # This will be the ground-truth rect during training if `use_predictions` is true.
            prev_target_rect = state['rect']

            # Coerce the aspect ratio of the rectangle to construct the context area.
            context_rect = self._context_rect(prev_target_rect, aspect, self.context_scale)
            # Extract same rectangle in past and current images and feed into conv-net.
            ims = tf.stack([
                self._crop(im, context_rect, CONTEXT_SIZE, mean_color),
                self._crop(prev_im, context_rect, CONTEXT_SIZE, mean_color),
            ], axis=1)

            # Extract features, perform search, get receptive field of response wrt image.
            ims_preproc = self._preproc(ims)
            # ims_preproc = cnn.as_tensor(ims_preproc, add_to_set=True)
            with tf.variable_scope('motion', reuse=(self._num_frames > 0)):
                response = _motion_net(ims_preproc, run_opts['is_training'],
                                       response_size=self.response_size,
                                       num_scales=self.num_scales,
                                       weight_decay=self.wd)
            # response = cnn.get_value(response)

            # mid_scale = (self.num_scales - 1) // 2
            # self._info.setdefault('response', []).append(
            #     _to_uint8(util.colormap(tf.sigmoid(response[:, mid_scale]), _COLORMAP)))

            losses = {}
            if self.mode in MODE_KEYS_SUPERVISED:
                # Get ground-truth translation and scale relative to context window.
                gt_rect_in_context = geom.crop_rect(gt_rect, context_rect)
                gt_translation, gt_rect_size = geom.rect_center_size(gt_rect_in_context)
                gt_size = helpers.scalar_size(gt_rect_size, self.aspect_method)

                # base_translations = ((self.response_stride / self.context_size) *
                #                      util.displacement_from_center(self.response_size))
                # scales = util.scale_range(tf.constant(self.num_scales),
                #                           tf.to_float(self.scale_step))
                base_target_size = self.target_size / CONTEXT_SIZE
                translation_stride = self.response_stride / CONTEXT_SIZE

                loss_name, loss = compute_loss(
                    response,
                    self.num_scales,
                    translation_stride,
                    self.scale_step,
                    base_target_size,
                    gt_translation,
                    gt_size,
                    **self.loss_params)

                # if reset_position:
                #     # TODO: Something better!
                #     losses[loss_name] = tf.zeros_like(loss)
                # else:
                #     losses[loss_name] = loss

            scales = util.scale_range(tf.constant(self.num_scales), tf.to_float(self.scale_step))

            # Use pyramid from loss function to obtain position.
            # Get relative translation and scale from response.
            # TODO: Upsample to higher resolution than original image?
            response_resize = cnn.get_value(cnn.upsample(
                response, self.response_stride, method=tf.image.ResizeMethod.BICUBIC))
            response_final = response_resize
            # if self.learn_motion:
            #     response_final = response_resize
            # else:
            #     response_final = apply_motion_penalty(
            #         response_resize, radius=self.window_radius * self.target_size,
            #         **self.window_params)
            translation, scale, in_arg_max = util.find_peak_pyr(
                response_final, scales, eps_abs=self.arg_max_eps)
            # Obtain translation in relative co-ordinates within search image.
            translation = 1 / tf.to_float(CONTEXT_SIZE) * translation
            # Get scalar representing confidence in prediction.
            # Use raw appearance score (before motion penalty).
            confidence = helpers.weighted_mean(response_resize, in_arg_max, axis=(-4, -3, -2))
            # Damp the scale update towards 1 (no change).
            # TODO: Should this be in log space?
            scale = self.scale_update_rate * scale + (1. - self.scale_update_rate) * 1.
            # Get rectangle in search image.
            prev_target_in_search = geom.crop_rect(prev_target_rect, context_rect)
            pred_in_search = _rect_translate_scale(prev_target_in_search, translation,
                                                   tf.expand_dims(scale, -1))
            # Move from search back to original image.
            pred = geom.crop_rect(pred_in_search, geom.crop_inverse(context_rect))

            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            if tf.estimator.ModeKeys.PREDICT or self.use_predictions:
                next_prev_rect = pred
            else:
                next_prev_rect = gt_rect

            self._num_frames += 1
            outputs = {'rect': pred, 'score': confidence}
            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                'image': im,
                'rect': next_prev_rect,
                'mean_color': state['mean_color'],
            }
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


def _motion_net(x, is_training, response_size, num_scales,
                weight_decay=0):
    '''
    Args:
        x: [b, t, h, w, c]

    Returns:
        [b, s, hr, wr, 1]
    '''
    assert len(x.shape) == 5
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
            x, unmerge = helpers.merge_dims(x, 0, 2)  # Merge time into batch.
            # 103 = 11 + (47 - 1) * 2 or
            # 195 = 11 + (47 - 1) * 4
            x = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1')
            # 47 = 3 + (23 - 1) * 2
            x = slim.max_pool2d(x, [3, 3], 2, scope='pool1')
            x = unmerge(x, axis=0)  # Un-merge time from batch.
            # Concatenate the images over time.
            x = tf.concat(tf.unstack(x, axis=1), axis=-1)
            x = slim.conv2d(x, 192, [5, 5], scope='conv2')
            # 23 = 3 + (11 - 1) * 2
            x = slim.max_pool2d(x, [3, 3], 2, scope='pool2')
            x = slim.conv2d(x, 384, [3, 3], scope='conv3')
            x = slim.conv2d(x, 384, [3, 3], scope='conv4')
            x = slim.conv2d(x, 256, [3, 3], scope='conv5')
            # 11 = 3 + (5 - 1) * 2
            x = slim.max_pool2d(x, [3, 3], 2, scope='pool5')
            # 5

            # Add fully-connected layer.
            x = slim.conv2d(x, 4096, [5, 5], padding='VALID', scope='fc6')
            x = tf.squeeze(x, axis=(-3, -2))
            x = slim.fully_connected(x, 4096, scope='fc7')
            # Regress to score map.
            output_shape = [num_scales, response_size, response_size, 1]
            x = slim.fully_connected(x, np.asscalar(np.prod(output_shape)), scope='fc8',
                                     activation_fn=None, normalizer_fn=None)
            x = helpers.split_dims(x, axis=-1, shape=output_shape)
            return x


def _to_uint8(x):
    return tf.image.convert_image_dtype(x, tf.uint8, saturate=True)


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
        response_shape = response.shape[-3:].as_list()
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


def compute_loss(scores, num_scales, translation_stride, scale_step, base_target_size,
                 gt_translation, gt_size,
                 method='sigmoid', params=None):
    '''
    Args:
        scores: [b, h, w, 1]
        See `multiscale_error`.
    '''
    params = params or {}

    try:
        loss_fn = {
            'sigmoid': _multiscale_sigmoid_cross_entropy_loss,
            # 'max_margin': _multiscale_max_margin_loss,
            # 'softmax': _multiscale_softmax_cross_entropy_loss,
        }[method]
    except KeyError as ex:
        raise ValueError('unknown loss method: "{}"'.format(method))
    name, loss = loss_fn(scores, num_scales, translation_stride, scale_step, base_target_size,
                         gt_translation, gt_size, **params)

    return name, loss


def multiscale_error(response_size, num_scales,
                     translation_stride, scale_step, base_target_size,
                     gt_translation, gt_size):
    '''Computes error for each element of multi-scale response.

    Ground-truth is relative to center of response.

    Args:
        response_size: Integer or 2-tuple of integers
        num_scales: Integer
        translation_stride: Float
        scale_step: Float
        base_target_size: Float or tensor with shape [b]
        gt_translation: [b, 2]
        gt_size: [b]

    Returns:
        err_translation: [b, s, h, w, 2]
        err_log_scale: [b, s]
    '''
    response_size = n_positive_integers(2, response_size)
    translation_stride = float(translation_stride)
    scale_step = float(scale_step)

    # TODO: Check if ground-truth is within range!

    base_translations = (
        translation_stride *
        tf.to_float(util.displacement_from_center(response_size)))
    scales = util.scale_range(tf.constant(num_scales), tf.to_float(scale_step))

    gt_scale = gt_size / base_target_size
    # err_log_scale: [b, s]
    err_log_scale = (
        tf.log(scales) -  # [s]
        tf.log(tf.expand_dims(gt_scale, -1)))  # [b] -> [b, 1]
    # translations: [b, s, h, w, 2]
    translations = (
        tf.expand_dims(base_translations, -4) *  # [..., h, w, 2] -> [..., 1, h, w, 2]
        helpers.expand_dims_n(scales, -1, 3))  # [s] -> [s, 1, 1, 1]
    # err_translation: [b, s, h, w, 2]
    err_translation = (
        translations -  # [b, s, h, w, 2]
        helpers.expand_dims_n(gt_translation, -2, 3))  # [b, 2] -> [b, 1, 1, 1, 2]
    return err_translation, err_log_scale


def _multiscale_sigmoid_cross_entropy_loss(
        scores, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size,
        pos_weight=1,
        balanced=False,
        label_method='hard',
        label_params=None):
    '''
    Args:
        scores: Tensor with shape [b, s, h, w, 1]

    Returns:
        Loss name, loss tensor with shape [b].
    '''
    label_params = label_params or {}

    try:
        label_fn = {
            'hard': _multiscale_hard_labels,
            # 'gaussian': _wrap_with_spatial_weight(_multiscale_gaussian_labels),
            'hard_binary': _wrap_with_spatial_weight(_multiscale_hard_binary_labels),
        }[label_method]
    except KeyError as ex:
        raise ValueError('unknown label shape: "{}"'.format(label_method))

    label_name, labels, weights = label_fn(
        scores, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size, **label_params)
    loss = lossfunc.normalized_sigmoid_cross_entropy_with_logits(
        logits=scores, targets=tf.broadcast_to(labels, tf.shape(scores)),
        weights=weights, axis=(-3, -2),
        pos_weight=pos_weight, balanced=balanced)
    # Remove singleton channels dimension.
    loss = tf.squeeze(loss, -1)

    # Reflect all parameters that affect magnitude of loss.
    # (Losses with same name are comparable.)
    loss_name = 'sigmoid_labels_{}_balanced_{}_pos_weight_{}'.format(
        label_name, balanced, pos_weight)
    return loss_name, loss


def _multiscale_hard_binary_labels(
        scores, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size,
        translation_radius,
        scale_radius):
    '''
    Does not support un-labelled examples.
    More suitable for use with softmax.
    '''
    response_size = helpers.known_spatial_dim(scores)
    err_translation, err_log_scale = multiscale_error(
        response_size, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size)
    # err_translation: [..., s, h, w, 2]
    # err_log_scale: [..., s]
    err_log_scale = helpers.expand_dims_n(err_log_scale, -1, 3)

    r2 = (tf.reduce_sum(tf.square(1 / translation_radius * err_translation),
                        axis=-1, keepdims=True) +
          tf.square(1 / tf.log(scale_radius) * err_log_scale))
    is_pos = (r2 <= 1)
    label = tf.to_float(is_pos)
    name = 'hard_binary_translation_{}_scale_{}'.format(translation_radius, scale_radius)
    return name, label


# MultiscaleScoreDims = collections.namedtuple('MultiScaleScore', [
#     'response_size',  # Integer or 2-tuple of integers
#     'num_scales',  # Integer
#     'translation_stride',  # Float
#     'scale_step',  # Float
# ])


def _multiscale_hard_labels(
        scores, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size,
        translation_radius_pos=20,
        translation_radius_neg=50,
        scale_radius_pos=1.1,
        scale_radius_neg=1.3):
    '''
    Args:
    '''
    response_size = helpers.known_spatial_dim(scores)
    err_translation, err_log_scale = multiscale_error(
        response_size, num_scales, translation_stride, scale_step, base_target_size,
        gt_translation, gt_size)
    # err_translation: [..., s, h, w, 2]
    # err_log_scale: [..., s]
    err_log_scale = helpers.expand_dims_n(err_log_scale, -1, 3)
    r2_pos = (tf.reduce_sum(tf.square(1 / translation_radius_pos * err_translation),
                            axis=-1, keepdims=True) +
              tf.square(1 / tf.log(scale_radius_pos) * err_log_scale))
    r2_neg = (tf.reduce_sum(tf.square(1 / translation_radius_neg * err_translation),
                            axis=-1, keepdims=True) +
              tf.square(1 / tf.log(scale_radius_neg) * err_log_scale))

    is_pos = (r2_pos <= 1)
    # TODO: Could force the minimum distance to be a positive?
    # Or exclude examples with no positive labels?
    is_neg = (r2_neg >= 1)
    # Ensure exclusivity and give priority to positive labels.
    is_neg = tf.logical_and(tf.logical_not(is_pos), is_neg)
    label = tf.to_float(is_pos)
    spatial_weight = tf.to_float(tf.logical_or(is_pos, is_neg))
    name = 'hard_translation_{}_{}_scale_{}_{}'.format(
        translation_radius_pos, translation_radius_neg, scale_radius_pos, scale_radius_neg)
    return name, label, spatial_weight


def _wrap_with_spatial_weight(fn):
    return functools.partial(_add_spatial_weight, fn)


def _add_spatial_weight(fn, *args, **kwargs):
    '''Calls fn() and additionally returns uniform spatial weights.'''
    name, label = fn(*args, **kwargs)
    return name, label, tf.ones_like(label)


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
