'''This model combines long-term appearance and short-term motion.'''

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
from seqtrack import receptive_field
from seqtrack import sample
from seqtrack.models import itermodel

from seqtrack.models import siamfc
from seqtrack.models import regress

from . import util as model_util
from . import feature_nets
from . import join_nets

from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

_COLORMAP = 'viridis'

MODE_KEYS_SUPERVISED = [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]


def default_params():
    return dict(
        target_size=64,
        template_size=127,
        search_size=255,
        aspect_method='perimeter',
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
        # feature_model_file='',
        appearance_model_file='',
        appearance_scope_dst='',  # e.g. 'model/appearance/',
        appearance_scope_src='',  # e.g. 'model/',
        learn_appearance=True,
        use_predictions=True,  # Use predictions for previous positions?
        # Tracking parameters:
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
        appearance_loss=False,
        appearance_loss_params=None,  # kwargs for compute_appearance_loss()
    )


class SiamFlow(object):
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
        self.template_scale = float(self.template_size) / self.target_size
        self.search_scale = float(self.search_size) / self.target_size
        # Defaults instead of None.
        self.join_params = self.join_params or {}
        self.multi_join_layers = self.multi_join_layers or []
        self.window_params = self.window_params or {}
        self.loss_params = self.loss_params or {}
        self.appearance_loss_params = self.appearance_loss_params or {}
        # Ensure types are correct.
        self.scale_update_rate = float(self.scale_update_rate)
        self.num_scales = int(self.num_scales)

        self._num_frames = 0  # Not including start frame.
        self._appearance_var_list = None  # For debug purposes.
        self._appearance_saver = None
        # self._feature_saver = None

    def train(self, example, run_opts, scope='model'):
        '''
        Args:
            example: ExampleSequence

        Returns: itermodel.OperationsUnroll
        '''
        if self.example_type == sample.ExampleTypeKeys.UNORDERED:
            raise ValueError('unordered examples not supported')
        elif self.example_type == sample.ExampleTypeKeys.CONSECUTIVE:
            # Use the default `instantiate_unroll` method.
            return itermodel.instantiate_unroll(self, example, run_opts, scope=scope)
        elif self.example_type == sample.ExampleTypeKeys.SEPARATE_INIT:
            return _instantiate_separate_init(self, example, run_opts, scope=scope)
        else:
            raise ValueError('unknown example type: "{}"'.format(self.example_type))

    def _context_rect(self, rect, aspect, amount):
        template_rect, _ = _get_context_rect(
            rect,
            context_amount=amount,
            aspect=aspect,
            aspect_method=self.aspect_method)
        return template_rect

    def _crop(self, image, rect, size, mean_color):
        return model_util.crop(
            image, rect, size,
            pad_value=mean_color if self.pad_with_mean else 0.5,
            feather=self.feather,
            feather_margin=self.feather_margin)

    def _crop_pyr(self, image, rect, size, scales, mean_color):
        return model_util.crop_pyr(
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

            with tf.variable_scope('appearance', reuse=False):
                template_rect = self._context_rect(target_rect, aspect, self.template_scale)
                template_im = self._crop(im, template_rect, self.template_size, mean_color)
                template_input = self._preproc(template_im)
                template_input = cnn.as_tensor(template_input, add_to_set=True)
                with tf.variable_scope('embed', reuse=False):
                    template_feat, template_layers, feature_scope = self._embed_net(
                        template_input, run_opts['is_training'])
                    # Get names relative to this scope for loading pre-trained.
                    # self._feature_vars = _global_variables_relative_to_scope(feature_scope)
                rf_template = template_feat.fields[template_input.value]
                template_feat = cnn.get_value(template_feat)
                feat_size = template_feat.shape[-3:-1].as_list()
                receptive_field.assert_center_alignment(self.template_size, feat_size, rf_template)

            # self._feature_saver = tf.train.Saver(self._feature_vars)

            with tf.name_scope('summary'):
                tf.summary.image('template', template_im)

            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                'rect': tf.identity(target_rect),
                'template_init': tf.identity(template_feat),
                'mean_color': tf.identity(mean_color),
            }
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
            # In this case, adopt the previous rectangle as the "ground-truth".
            if self.mode in MODE_KEYS_SUPERVISED:
                gt_rect = tf.where(labels['valid'], labels['rect'], state['rect'])
            else:
                gt_rect = None
            # Use the previous rectangle.
            # This will be the ground-truth rect during training if `use_predictions` is false.
            prev_target_rect = state['rect']

            # How to obtain template from previous state?
            template_feat = state['template_init']

            # Coerce the aspect ratio of the rectangle to construct the search area.
            search_rect = self._context_rect(prev_target_rect, aspect, self.search_scale)
            # Extract an image pyramid (use 1 scale when not in tracking mode).
            mid_scale = (self.num_scales - 1) // 2
            if self.num_scales == 1:
                scales = tf.constant([1.0], dtype=tf.float32)
            else:
                scales = model_util.scale_range(tf.constant(self.num_scales),
                                                tf.to_float(self.scale_step))
            search_ims, search_rects = self._crop_pyr(
                im, search_rect, self.search_size, scales, mean_color)

            with tf.name_scope('summary'):
                _image_sequence_summary('search', search_ims)

            with tf.variable_scope('appearance', reuse=False) as appearance_scope:
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

            if self._num_frames == 0:
                # Define appearance model saver.
                if self.appearance_model_file:
                    # Create the graph ops for the saver.
                    var_list = appearance_scope.global_variables()
                    var_list = {var.op.name: var for var in var_list}
                    if self.appearance_scope_dst or self.appearance_scope_src:
                        # Replace 'dst' with 'src'.
                        # Caution: This string replacement is a little dangerous.
                        var_list = {
                            k.replace(self.appearance_scope_dst, self.appearance_scope_src, 1): v
                            for k, v in var_list.items()
                        }
                    self._appearance_var_list = var_list
                    self._appearance_saver = tf.train.Saver(var_list)

            # Post-process scores.
            with tf.variable_scope('output', reuse=(self._num_frames > 0)):
                if not self.learn_appearance:
                    # TODO: Prevent batch-norm updates as well.
                    # TODO: Set trainable=False for all variables above.
                    response = tf.stop_gradient(response)

                # Regress response to translation and log(scale).
                output_shapes = {'translation': [2], 'log_scale': [1]}
                outputs = _output_net(response, output_shapes, run_opts['is_training'],
                                      weight_decay=self.wd)

            _image_sequence_summary('response',
                                    model_util.colormap(tf.sigmoid(response), _COLORMAP))

            losses = {}
            if self.mode in MODE_KEYS_SUPERVISED:
                # Get ground-truth translation and scale relative to search window.
                gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
                gt_position, gt_rect_size = geom.rect_center_size(gt_rect_in_search)
                # Positions in real interval [0, 1] correspond to real interval [0, search_size].
                # Pixel centers range from 0.5 to search_size - 0.5 in [0, search_size].
                gt_translation = gt_position - 0.5  # Displacement relative to center.
                gt_size = helpers.scalar_size(gt_rect_size, self.aspect_method)
                target_size_in_search = self.target_size / self.search_size
                # size = target_size * scale
                gt_scale = gt_size / target_size_in_search
                gt_log_scale = tf.log(gt_scale)

                if self.appearance_loss:
                    target_size_in_response = self.target_size / rf_response.stride
                    loss_name, loss = siamfc.compute_loss(response[:, mid_scale],
                                                          target_size_in_response,
                                                          **self.appearance_loss_params)
                    losses[loss_name] = loss

                loss_name, loss = regress.compute_loss_vector(outputs['translation'],
                                                              outputs['log_scale'],
                                                              gt_translation,
                                                              gt_log_scale,
                                                              **self.loss_params)
                losses[loss_name] = loss

                if reset_position:
                    # TODO: Something better!
                    # TODO: Keep appearance loss even when `reset_position` is true?
                    losses = {k: tf.zeros_like(v) for k, v in losses.items()}

            translation = outputs['translation']  # [b, 2]
            scale = tf.exp(outputs['log_scale'])  # [b, 1]

            # Damp the scale update towards 1 (no change).
            # TODO: Should this be in log space?
            scale = self.scale_update_rate * scale + (1. - self.scale_update_rate) * 1.
            # Get rectangle in search image.
            prev_target_in_search = geom.crop_rect(prev_target_rect, search_rect)
            pred_in_search = _rect_translate_scale(prev_target_in_search, translation, scale)
            # Move from search back to original image.
            pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))

            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            if self.mode in MODE_KEYS_SUPERVISED:
                next_prev_rect = pred if self.use_predictions else gt_rect
            else:
                next_prev_rect = pred

            # outputs = {'rect': pred, 'score': confidence}
            outputs = {'rect': pred}
            state = {
                'run_opts': run_opts,
                'aspect': aspect,
                'rect': next_prev_rect,
                'template_init': state['template_init'],
                'mean_color': state['mean_color'],
            }
            self._num_frames += 1
            return outputs, state, losses

    def end(self):
        losses = {}

        return losses

    def init(self, sess):
        if self.appearance_model_file:
            try:
                self._appearance_saver.restore(sess, self.appearance_model_file)
            except tf.errors.NotFoundError as ex:
                command = ('python -m tensorflow.python.tools.inspect_checkpoint'
                           ' --file_name ' + self.appearance_model_file)
                raise RuntimeError('\n\n'.join([
                    'could not load appearance model: {}'.format(str(ex)),
                    'tried to load variables:\n{}'.format(pprint.pformat(self._appearance_var_list)),
                    'use this command to inspect checkpoint:\n' + command]))


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
    square = model_util.coerce_aspect(rect, im_aspect=aspect, aspect_method=aspect_method)
    context = geom.grow_rect(context_amount, square)
    return context, square


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


# def _global_variables_relative_to_scope(scope):
#     '''
#     Args:
#         scope: VariableScope
#     '''
#     prefix = scope.name + '/'
#     # tf.Saver uses var.op.name to get variable name.
#     # https://stackoverflow.com/a/36156697/1136018
#     return {_remove_prefix(prefix, v.op.name): v for v in scope.global_variables()}


# def _remove_prefix(prefix, x):
#     if not x.startswith(prefix):
#         raise ValueError('does not have prefix "{}": "{}"'.format(prefix, x))
#     return x[len(prefix):]


def _instantiate_separate_init(iter_model_fn, example, run_opts, scope='model'):
    # Obtain the appearance model from the first frame,
    # then start tracking from the position in the second frame.
    # The following code is copied and modified from `itermodel.instantiate_unroll`.
    with tf.variable_scope(scope):
        state_init = iter_model_fn.start(example.features_init, run_opts)
        features = helpers.unstack_structure(example.features, axis=1)
        labels = helpers.unstack_structure(example.labels, axis=1)
        ntimesteps = len(features)
        predictions = [None for _ in range(ntimesteps)]
        losses = [None for _ in range(ntimesteps)]
        state = state_init
        for i in range(ntimesteps):
            # Reset the position in the first step!
            predictions[i], state, losses[i] = iter_model_fn.next(
                features[i], labels[i], state,
                reset_position=(i == 0))
        predictions = helpers.stack_structure(predictions, axis=1)
        losses = helpers.stack_structure(losses, axis=0)
        # Compute mean over frames.
        losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
        state_final = state

        extra_losses = iter_model_fn.end()
        helpers.assert_no_keys_in_common(losses, extra_losses)
        losses.update(extra_losses)
        return itermodel.OperationsUnroll(
            predictions=predictions,
            losses=losses,
            state_init=state_init,
            state_final=state_final,
        )


def _output_net(x, output_shapes, is_training, weight_decay=0):
    '''
    Args:
        x: [b, s, h, w, c]
        output_shapes: Dict that maps string to iterable of ints.
            e.g. {'response': [5, 17, 17, 1]} for a score-map
            e.g. {'translation': [2], 'scale': [1]} for translation regression

    Returns:
        Dictionary of outputs with shape [b] + output_shape.
    '''
    assert len(x.shape) == 5
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # Perform same operation on each scale.
            x, unmerge = helpers.merge_dims(x, 0, 2)  # Merge scale into batch.
            # Spatial dim 17
            x = slim.conv2d(x, 32, 3, padding='SAME', scope='conv1')
            x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool1')
            # Spatial dim 9
            x = slim.conv2d(x, 64, 3, padding='SAME', scope='conv2')
            x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool2')
            # Spatial dim 5
            x = unmerge(x, axis=0) # Unmerge scale from batch.
            x = tf.concat(tf.unstack(x, axis=1), axis=-1)  # Concatenate channels of all scales.
            x = slim.conv2d(x, 512, 5, padding='VALID', scope='fc3')
            # Spatial dim 1
            x = tf.squeeze(x, axis=(-2, -3))
            x = slim.fully_connected(x, 512, scope='fc4')

            # Regress to each output.
            y = {}
            for k in output_shapes.keys():
                with tf.variable_scope('head_{}'.format(k)):
                    y[k] = x
                    output_dim = np.asscalar(np.prod(output_shapes[k]))
                    y[k] = slim.fully_connected(y[k], output_dim, scope='fc5',
                                                activation_fn=None, normalizer_fn=None)
                    if len(output_shapes[k]) > 1:
                        y[k] = helpers.split_dims(y[k], axis=-1, shape=output_shapes[k])
            return y


def _image_sequence_summary(name, sequence, axis=1, **kwargs):
    with tf.name_scope(name) as scope:
        elems = tf.unstack(sequence, axis=axis)
        with tf.name_scope('elems'):
            for i in range(len(elems)):
                tf.summary.image(str(i), elems[i], **kwargs)
