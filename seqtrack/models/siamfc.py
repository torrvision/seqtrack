import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import math

from seqtrack import cnnutil
from seqtrack import geom
from seqtrack import lossfunc
from seqtrack import models
from seqtrack.models import interface
from seqtrack.models import util

from seqtrack.helpers import merge_dims, expand_dims_n, get_act
from tensorflow.contrib.layers.python.layers.utils import n_positive_integers


class SiamFC(interface.IterModel):

    def __init__(
            self,
            template_size=127,
            search_size=255,
            feature_padding='VALID',
            num_scales=5,
            scale_step=1.03,
            scale_update_rate=0.6,
            balance_classes=False):
        self._template_size = template_size
        self._search_size = search_size
        self._feature_padding = feature_padding
        self._num_scales = num_scales
        self._scale_step = scale_step
        self._scale_update_rate = scale_update_rate
        self._balance_classes = balance_classes

        self._debug = {k: [] for k in ['search', 'response', 'labels']}

    def start(self, frame, aspect, run_opts, enable_loss,
              image_summaries_collections=None, name='start'):
        with tf.name_scope(name) as scope:
            self._aspect = aspect
            self._is_training = run_opts['is_training']
            self._is_tracking = run_opts['is_tracking']
            self._enable_loss = enable_loss
            # self._init_rect = frame['y']
            self._image_summaries_collections = image_summaries_collections

            # TODO: frame['image'] and template_im have a viewport
            template_rect = util.context_rect(frame['y'], scale=2,
                                              im_aspect=self._aspect, aspect_method='perimeter')
            template_im = util.crop(frame['x'], template_rect, self._template_size)
            with tf.variable_scope('feature_net', reuse=False):
                template_feat, _ = _feature_net(template_im, padding=self._feature_padding,
                                                is_training=self._is_training)

            # TODO: Avoid passing template_feat to and from GPU (or re-computing).
            state = {
                'y': frame['y'],
                'template_feat': template_feat,
            }
            return state

    def next(self, frame, prev_state, name='timestep'):
        with tf.name_scope(name) as scope:
            # TODO: Should be 255 / 64 instead of 4?
            # TODO: Is 'perimeter' equivalent to SiamFC?

            # During training, use the true location of THIS frame.
            # If this label is not valid, use the previous location from the state.
            gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_state['y'])
            prev_rect = tf.cond(self._is_training, lambda: prev_state['y'], lambda: gt_rect)
            search_rect = util.context_rect(prev_rect, scale=4,
                                            im_aspect=self._aspect, aspect_method='perimeter')
            # Only extract an image pyramid in tracking mode.
            num_scales = tf.cond(self._is_tracking, lambda: self._num_scales, lambda: 1)
            mid_scale = (num_scales - 1) / 2
            scales = util.scale_range(num_scales, self._scale_step)
            search_ims, search_rects = util.crop_pyr(frame['x'], search_rect, self._search_size, scales)
            self._debug['search'].append(
                tf.image.convert_image_dtype(search_ims[:, mid_scale], tf.float32, saturate=True))
            rfs = {'search': cnnutil.identity_rf()}
            with tf.variable_scope('feature_net', reuse=True):
                search_feat, rfs = _feature_net(search_ims, rfs, padding=self._feature_padding,
                                                is_training=self._is_training)
            template_feat = prev_state['template_feat']
            response, rfs = util.diag_xcorr_rf(search_feat, template_feat, rfs)
            # Reduce to a scalar response.
            response = tf.reduce_sum(response, axis=-1, keep_dims=True)
            response = slim.batch_norm(response, scale=True, is_training=self._is_training)
            output_size = response.shape.as_list()[-3:-1]
            assert_alignment_correct(self._search_size, output_size, rfs['search'])
            sigmoid_response = tf.sigmoid(response)
            self._debug['response'].append(
                tf.image.convert_image_dtype(sigmoid_response[:, mid_scale], tf.uint8, saturate=True))

            loss = 0.
            if self._enable_loss:
                # response is [b, s, h, w, 1] with s = 1
                # TODO: Manage is_valid=False.

                # Obtain displacement from center of search image.
                disp = util.displacement_from_center(output_size)
                disp = tf.to_float(disp) * rfs['search'].stride / self._search_size
                grid = 0.5 + disp

                gt_rect_in_search = geom.crop_rect(gt_rect, search_rect)
                # labels, has_label = lossfunc.translation_labels(grid, gt_rect_in_search, **kwargs)
                labels, has_label = lossfunc.foreground_labels(
                    grid, gt_rect_in_search, shape='rect', sigma=0.3)
                self._debug['labels'].append(
                    tf.image.convert_image_dtype(tf.expand_dims(labels, -1), tf.uint8, saturate=True))
                if self._balance_classes:
                    weights = lossfunc.make_balanced_weights(labels, has_label, axis=(-2, -1))
                else:
                    weights = lossfunc.make_uniform_weights(has_label, axis=(-2, -1))
                # Note: This squeeze will fail if there is an image pyramid (is_tracking=True).
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels,                            # [b, h, w]
                    logits=tf.squeeze(response, axis=(1, 4))) # [b, s, h, w, 1] -> [b, h, w]
                loss = tf.where(frame['y_is_valid'], loss, tf.zeros_like(loss))
                loss = tf.reduce_sum(weights * loss)
                # ce_loss = lossfunc.cross_entropy(response, search_rect, frame['y'])

            prev_target_in_search = geom.crop_rect(prev_rect, search_rect)

            # Get relative translation and scale from response.
            translation_in_search, scale = _relative_motion_from_multiscale_scores(
                response, rfs['search'].stride, self._search_size, scales)
            # Damp the scale update towards 1.
            scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.
            # Get rectangle in search image.
            pred_in_search = _rect_translate_scale(prev_target_in_search, translation_in_search,
                                                   tf.expand_dims(scale, -1))
            # Move from search back to original image.
            # TODO: Test that this is equivalent to scaling translation?
            pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            # gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_rect)
            next_prev_rect = tf.cond(self._is_training, lambda: gt_rect, lambda: pred)

            outputs = {'y': pred}
            state = {
                'y': next_prev_rect,
                'template_feat': prev_state['template_feat'],
            }
            return pred, state, loss

    def end(self):
        extra_loss = 0.
        image_sequence_summary('search', tf.stack(self._debug['search'], axis=1),
                               collections=self._image_summaries_collections)
        image_sequence_summary('response', tf.stack(self._debug['response'], axis=1),
                               collections=self._image_summaries_collections)
        if self._enable_loss:
            image_sequence_summary('labels', tf.stack(self._debug['labels'], axis=1),
                                   collections=self._image_summaries_collections)
        return extra_loss


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


def _relative_motion_from_multiscale_scores(response, stride, search_size, scales, name='relative_motion'):
    '''
    Args:
        response: [b, s, h, w, c] where c is 1 or 2.
    '''
    with tf.name_scope(name) as scope:
        response_size = response.shape.as_list()[-3:-1]
        assert all(response_size)
        response_size = np.array(response_size)
        assert all(response_size % 2 == 1)
        # TODO: stride is (x, y) and size is (y, x)
        stride = np.array(n_positive_integers(2, stride))
        assert all(stride == stride[0])
        upsample_size = (response_size - 1) * stride + 1
        # Upsample.
        response, restore_fn = merge_dims(response, 0, 2)
        response = tf.image.resize_images(
            response, upsample_size, method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
        response = restore_fn(response, 0)
        # Map to range [0, 1]. Assume c == 1.
        response = tf.sigmoid(tf.squeeze(response, -1))
        # Apply motion penalty at all scales.
        response = response * hann_2d(upsample_size)
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


def hann(n):
    n = tf.convert_to_tensor(n)
    x = tf.to_float(tf.range(n)) / tf.to_float(n - 1)
    return 0.5 * (1. - tf.cos(2.*math.pi*x))


def hann_2d(im_size):
    size_0, size_1 = im_size
    window_0 = hann(size_0)
    window_1 = hann(size_1)
    return tf.expand_dims(window_0, 1) * tf.expand_dims(window_1, 0)


def assert_alignment_correct(input_size, output_size, rf):
    '''
    Args:
        input_size: (height, width)
        output_size: (height, width)
        rf: cnnutil.ReceptiveField, which uses (width, height)
    '''
    input_size = np.array(n_positive_integers(2, input_size))[[1, 0]]
    output_size = np.array(n_positive_integers(2, output_size))[[1, 0]]

    min_pt = rf.rect.int_center()
    max_pt = min_pt + rf.stride * (output_size - 1) + 1
    gap_before = min_pt
    gap_after = input_size - max_pt
    # If gap_before is equal to gap_after, then center of response map
    # corresponds to center of search image.
    assert np.array_equal(gap_before, gap_after)
