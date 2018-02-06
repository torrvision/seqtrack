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


class MotionPredictor(models_interface.IterModel):

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
            sigma_absolute=0.3,
            balance_classes=True,
            enable_margin_loss=False,
            margin_cost='iou',
            margin_reduce_method='max',
            enable_motion_loss=True,
            ):
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
        self._report_square = report_square
        self._hann_method = hann_method
        self._hann_coeff = hann_coeff
        self._arg_max_eps_rel = arg_max_eps_rel
        self._wd = wd
        self._enable_ce_loss = enable_ce_loss
        self._ce_label = ce_label
        self._sigma = sigma
        self._sigma_absolute = sigma_absolute
        self._balance_classes = balance_classes
        self._enable_margin_loss = enable_margin_loss
        self._margin_cost = margin_cost
        self._margin_reduce_method = margin_reduce_method
        self._enable_motion_loss = enable_motion_loss 

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

            y = frame['y']
            mean_color = tf.reduce_mean(frame['x'], axis=(-3, -2), keep_dims=True)
            # TODO: frame['image'] and template_im have a viewport
            template_rect, y_square = _get_context_rect(
                y, context_amount=self._template_scale, aspect=self._aspect,
                aspect_method=self._aspect_method)
            if self._report_square:
                y = y_square
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
                    initializer=tf.ones_initializer(),
                    collections=['siamese', tf.GraphKeys.GLOBAL_VARIABLES])
                template_feat *= template_mask

            # initial states for encoder and decoder
            cell_size = [tf.shape(template_feat)[0], 3, 3, 512] # TODO: Avoid manual dimensions?
            with tf.variable_scope('rnn_state'):
                encoder_state_init = {k: tf.fill(cell_size, 0.0, name='{}'.format(k)) for k in ['h', 'c']}
                decoder_state_init = {k: tf.fill(cell_size, 0.0, name='{}'.format(k)) for k in ['h', 'c']}

            # Create 'response_g' using 'y' and ground-truth generation function.
            centers = util.make_grid_centers((257, 257))
            response_g, _ = lossfunc.translation_labels(
                centers, y, shape='gaussian', sigma=self._sigma_absolute, absolute_translation=True)
            response_g = tf.expand_dims(response_g, -1)

            with tf.name_scope('summary'):
                tf.summary.image('template', _to_uint8(template_im[0:1]),
                                 max_outputs=1, collections=self._image_summaries_collections)

            # TODO: Avoid passing template_feat to and from GPU (or re-computing).
            state = {
                'y':             tf.identity(y),
                'template_feat': tf.identity(template_feat),
                'mean_color':    tf.identity(mean_color),
                'encoder':       encoder_state_init,
                'decoder':       decoder_state_init,
                'response_g':    response_g,
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
                        cost_method=self._margin_cost, reduce_method=self._margin_reduce_method)
                    self._info.setdefault('margin_cost', []).append(
                        _to_uint8(util.colormap(tf.expand_dims(cost, -1), _COLORMAP)))

            
            # -------------------------------------------------------------------------------------
            # Motion prediction model (in image frame)
            # 1. Encoder for motion history; potential extention to DNC, VAE, etc.
            # 2. Decoder for motion prediction; potential extention to attention mechanism.
            # 3. Use response_by_motion! (addition, probabilistic search space, etc.)
            # -------------------------------------------------------------------------------------
            # Encoder-decoder
            # TODO: test resizing to reduce computation
            input_to_encoder = tf.stop_gradient(prev_state['response_g'])
            input_to_encoder = tf.image.resize_images(input_to_encoder,
                                                      [65, 65], align_corners=True)
            with tf.variable_scope('encoder', reuse=(self._num_frames > 0)):
                motion_summary, encoder_state = _motion_encoder(
                    input_to_encoder, prev_state['encoder'],
                    tanh=False, # TODO: check
                    is_training=self._is_training)
            # TODO: I WAS USING DECODER WRONGLY!!!!!!
            decoder_type = 'cnn'
            with tf.variable_scope('decoder', reuse=(self._num_frames > 0)):
                response_by_motion, decoder_state = _motion_decoder(
                    motion_summary, prev_state['decoder'], decoder_type=decoder_type,
                    is_training=self._is_training)
            assert response_by_motion.shape.as_list()[-3] == 65
            response_by_motion = tf.image.resize_images(response_by_motion,
                                                        [257, 257], align_corners=True)

            # loss.
            if self._enable_motion_loss:
                centers = util.make_grid_centers((257, 257))
                labels, has_label = lossfunc.translation_labels(
                    centers, gt_rect, shape='gaussian', sigma=self._sigma_absolute,
                    absolute_translation=True)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels, logits=tf.squeeze(response_by_motion, -1))
                if self._balance_classes:
                    weights = lossfunc.make_balanced_weights(labels, has_label, axis=(-2, -1))
                else:
                    weights = lossfunc.make_uniform_weights(has_label, axis=(-2, -1))
                loss = tf.reduce_sum(weights * loss, axis=(1, 2))
                loss = tf.where(frame['y_is_valid'], loss, tf.zeros_like(loss))
                loss = tf.reduce_mean(loss)
                losses['ce_motion'] = loss
                self._info.setdefault('ce_labels_global', []).append(
                    _to_uint8(util.colormap(tf.expand_dims(labels, -1), _COLORMAP)))

            # Convert response in image frame: 'response_by_appearance'
            upsample_response = _finalize_scores(response, rfs['search'].stride,
                                                 self._hann_method, self._hann_coeff, sigmoid=False)
            rf_centers = _find_rf_centers(self._search_size, response_size, rfs['search'])
            upsample_response_size = upsample_response.shape.as_list()[-3:-1]
            response_rect_in_search = _align_corner_centers(rf_centers, upsample_response_size)
            response_rect_in_image = geom.crop_rect(response_rect_in_search,
                                                    geom.crop_inverse(search_rect))
            # crop multi-scale responses
            response_rect_in_image = tf.tile(tf.expand_dims(response_rect_in_image, 1),
                                             tf.stack([1, tf.shape(upsample_response)[1], 1]))
            response_rect_in_image, restore_fn_rect = merge_dims(response_rect_in_image, 0, 2)
            upsample_response, restore_fn_response = merge_dims(upsample_response, 0, 2)
            response_by_appearance = util.crop(upsample_response,
                                               geom.crop_inverse(response_rect_in_image),
                                               frame['x'].shape.as_list()[-3:-1],
                                               tf.reduce_min(upsample_response, (-3,-2,-1),
                                                             keep_dims=True),
                                               feather=True)
            response_rect_in_image = restore_fn_rect(response_rect_in_image, 0)
            upsample_response = restore_fn_response(upsample_response, 0)
            response_by_appearance = restore_fn_response(response_by_appearance, 0)

            # Final response -> Use responses from appearance and motion together.
            response_final = tf.sigmoid(response_by_appearance + response_by_motion)

            # Get 'absolute' translation and scale from response
            translation, scale = util.find_peak_pyr(response_final, scales,
                                                    absolute_translation=True,
                                                    eps_rel=self._arg_max_eps_rel)
            # Damp the scale update towards 1.
            scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.

            # Output rectangle.
            pred = geom.rect_translate(geom.grow_rect(tf.expand_dims(scale, -1), prev_rect),
                                       translation - geom.rect_center(prev_rect))
            # Limit size of object.
            pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # visualize in summary
            self._info.setdefault('response_by_appearance', []).append(
                    _to_uint8(util.colormap(tf.sigmoid(response_by_appearance[:, mid_scale]), _COLORMAP)))
            self._info.setdefault('response_by_motion', []).append(
                _to_uint8(util.colormap(tf.sigmoid(response_by_motion), _COLORMAP)))
            self._info.setdefault('response_final', []).append(
                _to_uint8(util.colormap(response_final[:, mid_scale], _COLORMAP)))
            vis = _paste_image(frame['x'],
                               util.colormap(response_final[:, mid_scale], _COLORMAP), alpha=0.5)
            self._info.setdefault('vis', []).append(_to_uint8(vis))


            ## Get relative translation and scale from response.
            #response_final = _finalize_scores(response, rfs['search'].stride,
            #                                  self._hann_method, self._hann_coeff)
            ## upsample_response_size = response_final.shape.as_list()[-3:-1]
            ## assert np.all(upsample_response_size <= self._search_size)
            #translation, scale = util.find_peak_pyr(response_final, scales,
            #                                        eps_rel=self._arg_max_eps_rel)
            #translation = translation / self._search_size

            #vis = _visualize_response(
            #    response[:, mid_scale], response_final[:, mid_scale],
            #    search_ims[:, mid_scale], rfs['search'], frame['x'], search_rect)
            #self._info.setdefault('vis', []).append(_to_uint8(vis))

            ## Damp the scale update towards 1.
            #scale = self._scale_update_rate * scale + (1. - self._scale_update_rate) * 1.
            ## Get rectangle in search image.
            #prev_target_in_search = geom.crop_rect(prev_rect, search_rect)
            #pred_in_search = _rect_translate_scale(prev_target_in_search, translation,
            #                                       tf.expand_dims(scale, -1))
            ## Move from search back to original image.
            ## TODO: Test that this is equivalent to scaling translation?
            #pred = geom.crop_rect(pred_in_search, geom.crop_inverse(search_rect))
            ## Limit size of object.
            #pred = _clip_rect_size(pred, min_size=0.001, max_size=10.0)

            # Rectangle to use in next frame for search area.
            # If using gt and rect not valid, use previous.
            # gt_rect = tf.where(frame['y_is_valid'], frame['y'], prev_rect)
            if self._use_gt:
                next_prev_rect = tf.cond(self._is_tracking, lambda: pred, lambda: gt_rect)
                next_prev_response = tf.cond(self._is_tracking,
                                             lambda: response_final[:, mid_scale],
                                             lambda: tf.expand_dims(labels, -1))
            else:
                next_prev_rect = pred
                next_prev_response = response_final[:, mid_scale]

            self._num_frames += 1
            outputs = {'y': pred, 'vis': vis}
            state = {
                'y': next_prev_rect,
                'template_feat': prev_state['template_feat'],
                'mean_color':    prev_state['mean_color'],
                'encoder':       encoder_state,
                'decoder':       decoder_state,
                'response_g':    next_prev_response,
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


def _motion_encoder(x, state_prev, tanh=True, is_training=None, name='motion_encoder'):
    with tf.name_scope(name) as scope:
        # motion embedding
        with slim.arg_scope([slim.conv2d],
                padding='SAME',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training, 'fused': True},
                ):
            # Assuming the input is resized to 65x65.
            x = slim.conv2d(x, 32, 5, 2, scope='conv1')
            x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool1')
            x = slim.conv2d(x, 64, 5, 2, scope='conv2')
            x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool2')
            x = slim.conv2d(x, 128, 3, 1, scope='conv3')
            x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool3')

            ## Assuming the input is original size, i.e. 257x257
            #x = slim.conv2d(x, 32, 5, 2, scope='conv1')
            #x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool1')
            #x = slim.conv2d(x, 64, 5, 2, scope='conv2')
            #x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool2')
            #x = slim.conv2d(x, 128, 5, 2, scope='conv3')
            #x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool3')
            #x = slim.conv2d(x, 256, 3, 1, scope='conv4')
            #x = slim.max_pool2d(x, kernel_size=2, padding='SAME', scope='pool4')
        output, state_next = _conv_lstm(x, state_prev, is_training)
        # Put tanh to create summary; following original paper.
        if tanh:
            motion_summary = slim.conv2d(output, output.shape.as_list()[-1], 1, 1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         activation_fn=tf.nn.tanh, scope='context')
        else:
            motion_summary = output
        return motion_summary, state_next

def _motion_decoder(x, state_prev, decoder_type='rnn', is_training=None, name='motion_decoder'):
    with tf.name_scope(name) as scope:
        if decoder_type == 'rnn':
            # initialize hidden states using summary.
            state_prev = {k: slim.conv2d(x, x.shape.as_list()[-1], 1, 1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={'is_training': is_training},
                                         activation_fn=tf.nn.tanh, scope='conv_'+k)
                          for k in state_prev.keys()} # TODO: only H?
            output, state_next = _conv_lstm(x, state_prev, is_training)
            # Deconvolution
            with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    ):
                # Assuming the input is resized to 65x65.
                output = slim.conv2d(tf.image.resize_images(output, [9, 9], align_corners=True),
                                     64, 3, 1, scope='conv2')
                output = slim.conv2d(tf.image.resize_images(output, [33, 33], align_corners=True),
                                     32, 3, 1, scope='conv3')
                output = slim.conv2d(tf.image.resize_images(output, [65, 65], align_corners=True),
                                     1, 3, 1, activation_fn=None, normalizer_fn=None, scope='conv4') # NOTE: with bnorm -> not work.

                ## Assuming the input is original size, i.e. 257x257
                #output = slim.conv2d(tf.image.resize_images(output, [9, 9], align_corners=True),
                #                     128, 3, 1, scope='conv1')
                #output = slim.conv2d(tf.image.resize_images(output, [33, 33], align_corners=True),
                #                     64, 3, 1, scope='conv2')
                #output = slim.conv2d(tf.image.resize_images(output, [129, 129], align_corners=True),
                #                     32, 3, 1, scope='conv3')
                #output = slim.conv2d(tf.image.resize_images(output, [257, 257], align_corners=True),
                #                     1, 3, 1, activation_fn=None, normalizer_fn=None, scope='conv4') # NOTE: with bnorm -> not work.
        elif decoder_type == 'cnn':
            with slim.arg_scope([slim.conv2d],
                    padding='SAME',
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    ):
                # Assuming the input is resized to 65x65.
                output = x
                output = slim.conv2d(tf.image.resize_images(output, [5, 5], align_corners=True),
                                     128, 3, 1, scope='conv1')
                output = slim.conv2d(tf.image.resize_images(output, [9, 9], align_corners=True),
                                     64, 3, 1, scope='conv2')
                output = slim.conv2d(tf.image.resize_images(output, [33, 33], align_corners=True),
                                     32, 3, 1, scope='conv3')
                output = slim.conv2d(tf.image.resize_images(output, [65, 65], align_corners=True),
                                     1, 3, 1, activation_fn=None, normalizer_fn=None, scope='conv4') # NOTE: with bnorm -> not work.
            state_next = state_prev
        else:
            raise ValueError('Not available decoder type.')
        return output, state_next

def _conv_lstm(x, state, is_training, name='conv_lstm'):
    with tf.name_scope(name) as scope:
        assert is_training is not None
        h_prev, c_prev = state['h'], state['c']
        with slim.arg_scope([slim.conv2d],
                # TODO: no batchnorm here?
                #num_outputs=h_prev.shape.as_list()[-1], kernel_size=3, activation_fn=None):
                num_outputs=h_prev.shape.as_list()[-1], kernel_size=3, activation_fn=None,
                normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            # To avoid concat error due to dimension difference between x and h for multi-scale case.
            it = tf.nn.sigmoid(slim.conv2d(x, scope='xi') + slim.conv2d(h_prev, scope='hi'))
            ft = tf.nn.sigmoid(slim.conv2d(x, scope='xf') + slim.conv2d(h_prev, scope='hf'))
            ct_tilda = tf.nn.tanh(slim.conv2d(x, scope='xc') + slim.conv2d(h_prev, scope='hc'))
            ct = (ft * c_prev) + (it * ct_tilda)
            ot = tf.nn.sigmoid(slim.conv2d(x, scope='xo') + slim.conv2d(h_prev, scope='ho'))
            ht = ot * tf.nn.tanh(ct)
            #it = tf.nn.sigmoid(slim.conv2d(tf.concat([x, h_prev], -1), scope='i'))
            #ft = tf.nn.sigmoid(slim.conv2d(tf.concat([x, h_prev], -1), scope='f'))
            #ct_tilda = tf.nn.tanh(slim.conv2d(tf.concat([x, h_prev], -1), scope='c'))
            #ct = (ft * c_prev) + (it * ct_tilda)
            #ot = tf.nn.sigmoid(slim.conv2d(tf.concat([x, h_prev], -1), scope='o'))
            #ht = ot * tf.nn.tanh(ct)
        output = tf.identity(ht)
        state['h'] = ht
        state['c'] = ct
        return output, state


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
        search_size, search_scale, cost_method='iou', reduce_method='max',
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
        loss = _max_margin(score, cost, axis=[-2, -1], reduce_method=reduce_method)
        # TODO: Is this the best way to handle gt_is_valid?
        # (set loss to zero and include in mean)
        loss = tf.where(gt_is_valid, loss, tf.zeros_like(loss))
        loss = tf.reduce_mean(loss)
        return loss, cost


def _max_margin(score, cost, axis=None, reduce_method='max', name='max_margin'):
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
        if reduce_method == 'max':
            # Structured output loss.
            loss = tf.reduce_max(violation, axis=axis)
        elif reduce_method == 'mean':
            # Mean over all hinges; like triplet loss.
            loss = tf.reduce_mean(violation, axis=axis)
        elif reduce_method == 'sum':
            loss = tf.reduce_sum(violation, axis=axis)
        else:
            raise ValueError('unknown reduce method: {}'.format(reduce_method))
        return loss


def _finalize_scores(
        response, stride, hann_method, hann_coeff, sigmoid=True, name='finalize_scores'):
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
            response = tf.sigmoid(response) if sigmoid else response
        elif hann_method == 'mul_prob':
            response = tf.sigmoid(response) if sigmoid else response
            response *= tf.expand_dims(hann_2d(upsample_size), -1)
        elif hann_method == 'none' or not hann_method:
            response = tf.sigmoid(response) if sigmoid else response
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


def _clip_rect_size(rect, min_size=None, max_size=None, name='clip_rect_size'):
    with tf.name_scope(name) as scope:
        center, size = geom.rect_center_size(rect)
        if max_size is not None:
            size = tf.minimum(size, max_size)
        if min_size is not None:
            size = tf.maximum(size, min_size)
        return geom.make_rect_center_size(center, size)