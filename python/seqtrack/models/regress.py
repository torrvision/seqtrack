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

MODE_KEYS_SUPERVISED = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]

CONTEXT_SIZE = 195


def default_params():
    return dict(
        target_size=64,
        # image_size=640,
        aspect_method='perimeter',
        pad_with_mean=True,  # Use mean of first image for padding.
        feather=False,
        feather_margin=0.1,
        center_input_range=True,  # Make range [-0.5, 0.5] instead of [0, 1].
        keep_uint8_range=False,  # Use input range of 255 instead of 1.
        output_form='discrete',  # 'vector', 'discrete'
        response_stride=8,  # if output_form == 'discrete'
        response_size=17,  # if output_form == 'discrete'
        num_scales=5,  # if output_form == 'discrete'
        log_scale_step=0.02,  # if output_form == 'discrete'
        use_predictions=False,  # Use predictions for previous positions?
        scale_update_rate=1,
        arg_max_eps=0.0,
        stateless=False,  # Ignore previous image.
        # Loss parameters:
        wd=0.0,
        loss_params=None,  # kwargs for compute_loss_xxx()
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
        # self._info = {}

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
                # 'image': tf.image.resize_images(im, [self.image_size, self.image_size]),
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
            # This will be the ground-truth rect during training if `use_predictions` is false.
            prev_target_rect = state['rect']

            # Coerce the aspect ratio of the rectangle to construct the context area.
            context_rect = self._context_rect(prev_target_rect, aspect, self.context_scale)
            # Extract same rectangle in past and current images and feed into conv-net.
            context_curr = self._crop(im, context_rect, CONTEXT_SIZE, mean_color)
            context_prev = self._crop(prev_im, context_rect, CONTEXT_SIZE, mean_color)
            with tf.name_scope('summary_context'):
                tf.summary.image('curr', context_curr)
                tf.summary.image('prev', context_curr)
            ims = [context_curr] if self.stateless else [context_curr, context_prev]
            ims = tf.stack(ims, axis=1)

            if self.output_form == 'discrete':
                output_shapes = {
                    'response': [self.num_scales, self.response_size, self.response_size, 1]}
            elif self.output_form == 'vector':
                output_shapes = {'translation': [2], 'log_scale': [1]}
            else:
                raise ValueError('unknown output form: "{}"'.format(output_form))

            # Extract features, perform search, get receptive field of response wrt image.
            ims_preproc = self._preproc(ims)
            with tf.variable_scope('motion', reuse=(self._num_frames > 0)):
                outputs = _motion_net(ims_preproc, output_shapes, run_opts['is_training'],
                                      weight_decay=self.wd)
            outputs = {k: tf.verify_tensor_all_finite(v, 'output "{}" not finite'.format(k))
                       for k, v in outputs.items()}

            losses = {}
            if self.mode in MODE_KEYS_SUPERVISED:
                # Get ground-truth translation and scale relative to context window.
                gt_rect_in_context = geom.crop_rect(gt_rect, context_rect)
                gt_position, gt_rect_size = geom.rect_center_size(gt_rect_in_context)
                gt_translation = gt_position - 0.5  # Displacement relative to center.
                gt_size = helpers.scalar_size(gt_rect_size, self.aspect_method)
                # Scale is size relative to target_size.
                gt_scale = gt_size / (self.target_size / CONTEXT_SIZE)
                gt_log_scale = tf.log(gt_scale)

                if self.output_form == 'discrete':
                    # base_translations = ((self.response_stride / self.context_size) *
                    #                      util.displacement_from_center(self.response_size))
                    # scales = util.scale_range(tf.constant(self.num_scales),
                    #                           tf.to_float(self.log_scale_step))
                    base_target_size = self.target_size / CONTEXT_SIZE
                    translation_stride = self.response_stride / CONTEXT_SIZE
                    loss_name, loss = compute_loss_discrete(
                        outputs['response'],
                        self.num_scales,
                        translation_stride,
                        self.log_scale_step,
                        base_target_size,
                        gt_translation,
                        gt_size,
                        **self.loss_params)
                else:
                    loss_name, loss = compute_loss_vector(
                        outputs['translation'],
                        outputs['log_scale'],
                        gt_translation,
                        gt_log_scale,
                        **self.loss_params)

                # if reset_position:
                #     # TODO: Something better!
                #     losses[loss_name] = tf.zeros_like(loss)
                # else:
                #     losses[loss_name] = loss
                losses[loss_name] = loss

            if self.output_form == 'discrete':
                response = outputs['response']
                scales = util.scale_range(tf.constant(self.num_scales),
                                          tf.to_float(self.log_scale_step))
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
                scale = tf.expand_dims(scale, -1)  # [b, 1]
                # Obtain translation in relative co-ordinates within search image.
                translation = 1 / tf.to_float(CONTEXT_SIZE) * translation
                # Get scalar representing confidence in prediction.
                # Use raw appearance score (before motion penalty).
                confidence = helpers.weighted_mean(response_resize, in_arg_max, axis=(-4, -3, -2))
            else:
                translation = outputs['translation']  # [b, 2]
                scale = tf.exp(outputs['log_scale'])  # [b, 1]

            # Damp the scale update towards 1 (no change).
            # TODO: Should this be in log space?
            scale = self.scale_update_rate * scale + (1. - self.scale_update_rate) * 1.
            # Get rectangle in search image.
            prev_target_in_context = geom.crop_rect(prev_target_rect, context_rect)
            pred_in_context = _rect_translate_scale(prev_target_in_context, translation, scale)
            # Move from search back to original image.
            pred = geom.crop_rect(pred_in_context, geom.crop_inverse(context_rect))

            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            if self.mode in MODE_KEYS_SUPERVISED:
                next_prev_rect = pred if self.use_predictions else gt_rect
            else:
                next_prev_rect = pred

            self._num_frames += 1
            # outputs = {'rect': pred, 'score': confidence}
            predictions = {'rect': pred}
            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                # 'image': tf.image.resize_images(im, [self.image_size, self.image_size]),
                'image': im,
                'rect': next_prev_rect,
                'mean_color': state['mean_color'],
            }
            return predictions, state, losses

    def end(self):
        losses = {}
        # with tf.name_scope('summary'):
        #     for key in self._info:
        #         _image_sequence_summary(key, tf.stack(self._info[key], axis=1), axis=1)
        return losses

    def init(self, sess):
        pass


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


def _motion_net(x, output_shapes, is_training, weight_decay=0):
    '''
    Args:
        x: [b, t, h, w, c]
        output_shapes: Dict that maps string to iterable of ints.
            e.g. {'response': [5, 17, 17, 1]} for a score-map
            e.g. {'translation': [2]} for translation regression

    Returns:
        Dictionary of outputs with shape [b] + output_shape.
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
            y = {}
            for k in output_shapes.keys():
                with tf.variable_scope('head_{}'.format(k)):
                    y[k] = x
                    output_dim = np.asscalar(np.prod(output_shapes[k]))
                    y[k] = slim.fully_connected(y[k], output_dim, scope='fc8',
                                                activation_fn=None, normalizer_fn=None)
                    if len(output_shapes[k]) > 1:
                        y[k] = helpers.split_dims(y[k], axis=-1, shape=output_shapes[k])

            return y


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


def compute_loss_vector(translation, log_scale, gt_translation, gt_log_scale,
                        translation_radius=1.0,
                        log_scale_radius=1.0):
    '''
    Args:
        translation: [b, 2]
        log_scale: [b, 1]
        gt_translation: [b, 2]
        gt_log_scale: [b, 1]

    Returns:
        name: String
        loss: Tensor of shape [b]
    '''
    tf.summary.histogram('translation', translation)
    tf.summary.histogram('log_scale', log_scale)
    tf.summary.histogram('gt_translation', gt_translation)
    tf.summary.histogram('gt_log_scale', gt_log_scale)
    err_translation = (1 / translation_radius) * (translation - gt_translation)
    err_log_scale = (1 / log_scale_radius) * (log_scale - gt_log_scale)
    loss = tf.norm(err_translation, axis=-1) + tf.norm(err_log_scale, axis=-1)
    loss_name = 'norm_translation_{}_log_scale_{}'.format(translation_radius, log_scale_radius)
    return loss_name, loss


def compute_loss_discrete(scores, num_scales, translation_stride, log_scale_step, base_target_size,
                          gt_translation, gt_size,
                          method='sigmoid', params=None):
    '''
    Args:
        scores: [b, s, h, w, 1]
        See `multiscale_error`.

    Returns:
        name: String
        loss: Tensor of shape [b]
    '''
    params = params or {}
    try:
        loss_fn = {
            'sigmoid': multiscale_sigmoid_cross_entropy_loss,
            # 'max_margin': _multiscale_max_margin_loss,
            # 'softmax': _multiscale_softmax_cross_entropy_loss,
        }[method]
    except KeyError as ex:
        raise ValueError('unknown loss method: "{}"'.format(method))
    scores = tf.verify_tensor_all_finite(scores, 'scores are not finite')
    name, loss = loss_fn(scores, num_scales, translation_stride, log_scale_step, base_target_size,
                         gt_translation, gt_size, **params)
    loss = tf.verify_tensor_all_finite(loss, 'scores are finite but loss is not')
    return name, loss


def multiscale_error(response_size, num_scales,
                     translation_stride, log_scale_step, base_target_size,
                     gt_translation, gt_size):
    '''Computes error for each element of multi-scale response.

    Ground-truth is relative to center of response.

    Args:
        response_size: Integer or 2-tuple of integers
        num_scales: Integer
        translation_stride: Float
        log_scale_step: Float
        base_target_size: Float or tensor with shape [b]
        gt_translation: [b, 2]
        gt_size: [b]

    Returns:
        err_translation: [b, s, h, w, 2]
        err_log_scale: [b, s]
    '''
    response_size = n_positive_integers(2, response_size)
    translation_stride = float(translation_stride)
    log_scale_step = float(log_scale_step)

    # TODO: Check if ground-truth is within range!

    base_translations = (
        translation_stride *
        tf.to_float(util.displacement_from_center(response_size)))
    scales = util.scale_range(tf.constant(num_scales), tf.to_float(log_scale_step))

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


def multiscale_sigmoid_cross_entropy_loss(
        scores, num_scales, translation_stride, log_scale_step, base_target_size,
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
        label_fn = LABEL_FNS[label_method]
    except KeyError as ex:
        raise ValueError('unknown label shape: "{}"'.format(label_method))

    response_size = helpers.known_spatial_dim(scores)
    label_name, labels, weights = label_fn(
        response_size, num_scales, translation_stride, log_scale_step, base_target_size,
        gt_translation, gt_size, **label_params)
    loss = lossfunc.normalized_sigmoid_cross_entropy_with_logits(
        logits=scores, targets=tf.broadcast_to(labels, tf.shape(scores)),
        weights=weights, axis=(-4, -3, -2),
        pos_weight=pos_weight, balanced=balanced)
    # Remove singleton channels dimension.
    loss = tf.squeeze(loss, -1)

    _image_sequence_summary('scores_pyramid', scores, axis=1)
    _image_sequence_summary('labels_pyramid', labels, axis=1)

    # Reflect all parameters that affect magnitude of loss.
    # (Losses with same name are comparable.)
    loss_name = 'sigmoid_labels_{}_balanced_{}_pos_weight_{}'.format(
        label_name, balanced, pos_weight)
    return loss_name, loss


def multiscale_hard_binary_labels(
        response_size, num_scales, translation_stride, log_scale_step, base_target_size,
        gt_translation, gt_size,
        translation_radius=0.2,
        scale_radius=1.03):
    '''
    Does not support un-labelled examples.
    More suitable for use with softmax.
    '''
    err_translation, err_log_scale = multiscale_error(
        response_size, num_scales, translation_stride, log_scale_step, base_target_size,
        gt_translation, gt_size)
    # err_translation: [..., s, h, w, 2]
    # err_log_scale: [..., s]
    err_log_scale = helpers.expand_dims_n(err_log_scale, -1, 3)
    abs_translation_radius = translation_radius * base_target_size
    log_scale_radius = tf.log(tf.to_float(scale_radius))

    r2_translation = tf.reduce_sum(tf.square(1 / abs_translation_radius * err_translation),
                                   axis=-1, keepdims=True)
    r2_scale = tf.square(1 / log_scale_radius * err_log_scale)
    r2 = (r2_translation + r2_scale)
    is_pos = (r2 <= 1)
    label = tf.to_float(is_pos)

    name = 'hard_binary_translation_{}_scale_{}'.format(translation_radius, scale_radius)
    weights = tf.ones_like(label)
    return name, label, weights


# MultiscaleScoreDims = collections.namedtuple('MultiScaleScore', [
#     'response_size',  # Integer or 2-tuple of integers
#     'num_scales',  # Integer
#     'translation_stride',  # Float
#     'log_scale_step',  # Float
# ])


def multiscale_hard_labels(
        response_size, num_scales, translation_stride, log_scale_step, base_target_size,
        gt_translation, gt_size,
        translation_radius_pos=0.2,
        translation_radius_neg=0.5,
        scale_radius_pos=1.03,
        scale_radius_neg=1.1):
    '''
    Args:
    '''
    err_translation, err_log_scale = multiscale_error(
        response_size, num_scales, translation_stride, log_scale_step, base_target_size,
        gt_translation, gt_size)
    # err_translation: [..., s, h, w, 2]
    # err_log_scale: [..., s]
    err_log_scale = helpers.expand_dims_n(err_log_scale, -1, 3)
    r2_pos = (
        tf.reduce_sum(
            tf.square(1 / (translation_radius_pos * base_target_size) * err_translation),
            axis=-1, keepdims=True) +
        tf.square(1 / tf.log(scale_radius_pos) * err_log_scale))
    r2_neg = (
        tf.reduce_sum(
            tf.square(1 / (translation_radius_neg * base_target_size) * err_translation),
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


def _rect_translate_scale(rect, translate, scale, name='rect_translate_scale'):
    '''
    Args:
        rect: [..., 4]
        translate: [..., 2]
        scale: [..., 1]
    '''
    with tf.name_scope(name) as scope:
        center, size = geom.rect_center_size(rect)
        return geom.make_rect_center_size(center + translate, size * scale)


def _clip_rect_size(rect, min_size=None, max_size=None, name='clip_rect_size'):
    with tf.name_scope(name) as scope:
        center, size = geom.rect_center_size(rect)
        if max_size is not None:
            size = tf.minimum(size, max_size)
        if min_size is not None:
            size = tf.maximum(size, min_size)
        return geom.make_rect_center_size(center, size)


def _image_sequence_summary(name, sequence, axis=1, **kwargs):
    with tf.name_scope(name) as scope:
        elems = tf.unstack(sequence, axis=axis)
        with tf.name_scope('elems'):
            for i in range(len(elems)):
                tf.summary.image(str(i), elems[i], **kwargs)


LABEL_FNS = {
    'hard': multiscale_hard_labels,
    # 'gaussian': multiscale_gaussian_labels,
    'hard_binary': multiscale_hard_binary_labels,
}
