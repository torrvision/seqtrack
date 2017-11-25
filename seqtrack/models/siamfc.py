import tensorflow as tf
import tensorflow.contrib.slim as slim

from seqtrack import cnnutil
from seqtrack.models import util

from seqtrack.helpers import merge_dims, get_act


class SiamFC(object):

    def __init__(
            self,
            template_size=127,
            search_size=255,
            num_scales=5,
            scale_step=1.03):
        self._template_size = template_size
        self._search_size = search_size
        self._num_scales = num_scales
        self._scale_step = scale_step

    def start(self, frame, is_training, enable_loss):
        self._is_training = is_training
        self._enable_loss = enable_loss
        self._init_rect = frame['rect']
        self._prev_rect = tf.identity(_init_rect) # May be part of state.

        # TODO: frame['image'] and template_im have a viewport
        template_rect = util.context_rect(frame['rect'], self._target_scale,
                                          frame['aspect'], aspect_method='perimeter')
        template_im = crop(frame['image'], template_rect, self._template_size)
        with tf.variable_scope('feature_net', reuse=False):
            template_feat, _ = _feature_net(template_im, padding='VALID',
                                            is_training=self._is_training)

        # TODO: Avoid passing template_feat to and from GPU (or re-computing).
        state = {
            'rect': frame['rect'],
            'template_feat': template_feat,
        }
        return state

    def next(self, frame, prev_state):
        # TODO: Should be 255 / 64 instead of 4?
        # TODO: Is 'perimeter' equivalent to SiamFC?
        search_rect = util.context_rect(prev_state['rect'], self._search_scale,
                                        frame['aspect'], aspect_method='perimeter')
        # Extract an image pyramid in testing mode.
        num_scales = tf.cond(self._is_training, 1, self._num_scales)
        scales = util.scale_range(num_scales, self._scale_step)
        search_ims, search_rects = util.crop_pyr(frame['x'], search_rect, self._search_size, scales)
        rfs = {'search': cnnutil.identity_rf()}
        with tf.variable_scope('feature_net', reuse=True):
            search_feat, rfs = _feature_net(search_ims, rfs, padding='VALID',
                                            is_training=self._is_training)
        template_feat = prev_state['template_feat']
        response, rfs = util.diag_conv_rf(search_feat, template_feat, rfs)
        # Reduce to a scalar response.
        response = tf.sum(response, axis=-1)
        response = slim.batch_norm(response, scale=True, is_training=self._is_training)

        if self._enable_loss:
            pass

        # Rectangle to use in next frame for search area.
        # If using gt and rect not valid, use previous.
        gt_rect = tf.cond(frame['rect_valid'], frame['rect'], prev_state['rect'])
        next_prev_rect = tf.cond(self._is_training, gt_rect, pred_rect)

        state = {
            'rect': next_prev_rect,
            'template_feat': prev_state['template_feat'],
        }
        return pred_rect, state


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
        x = _feature_net(x, rfs, padding, output_act, enable_output_bnorm, is_training, name)
        x = restore(x, 0)
        return x

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
