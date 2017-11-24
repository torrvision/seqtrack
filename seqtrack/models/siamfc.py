import tensorflow as tf
import tensorflow.contrib.slim as slim

from seqtrack import cnnutil
from seqtrack.models import util


# class SiamFC(object):
# 
#     def __init__(
#             self,
#             template_size=127,
#             search_size=255):
#         self._template_size = template_size
#         self._search_size = search_size
# 
#     def start(self, frame):
#         # TODO: frame['image'] and template_im have a viewport
#         template_rect = _compute_template_rect(frame['rect'], frame['aspect'])
#         template_im = crop(frame['image'], template_rect, self._template_size)
#         with tf.variable_scope('feature_net', reuse=False):
#             template_feat = _feature_net(template_im, padding='VALID')
# 
#     def next(self, frame):
#         search_rect = _search_rect(y_prev, frame['aspect'])
#         search_im = crop(frame['x'], search_rect)
#         with tf.variable_scope('feature_net', reuse=True):
#             search_feat = _feature_net(search_im, padding='VALID')
# 
#         response = tf.nn.conv2d(


def _feature_net(x, padding):
    # AlexNet from TF slim models.
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
    # x = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1')
    # x = slim.max_pool2d(x, [3, 3], 2, scope='pool1')
    # x = slim.conv2d(x, 192, [5, 5], scope='conv2')
    # x = slim.max_pool2d(x, [3, 3], 2, scope='pool2')
    # x = slim.conv2d(x, 384, [3, 3], scope='conv3')
    # x = slim.conv2d(x, 384, [3, 3], scope='conv4')
    # x = slim.conv2d(x, 256, [3, 3], scope='conv5')
    # x = slim.max_pool2d(x, [3, 3], 2, scope='pool5')

    # https://github.com/bertinetto/siamese-fc/blob/master/training/vid_create_net.m
    # with slim.arg_scope(
    #         [slim.conv2d, slim.max_pool2d],
    #         padding=padding):
    #     rfs = {'image': cnnutil.identity_rf()}
    #     x, rfs = util.conv2d(x, rfs, 96, [11, 11], 2, scope='conv1')
    #     x, rfs = util.max_pool2d(x, rfs, [3, 3], 2, scope='pool1')
    #     x, rfs = util.conv2d(x, rfs, 256, [5, 5], scope='conv2')
    #     x, rfs = util.max_pool2d(x, rfs, [3, 3], 2, scope='pool2')
    #     x, rfs = util.conv2d(x, rfs, 384, [3, 3], scope='conv3')
    #     x, rfs = util.conv2d(x, rfs, 384, [3, 3], scope='conv4')
    #     x, rfs = util.conv2d(x, rfs, 256, [3, 3], scope='conv5')

    rfs = {'image': cnnutil.identity_rf()}
    x, rfs = util.conv2d(x, rfs, 96, [11, 11], 2, padding=padding, scope='conv1')
    x, rfs = util.max_pool2d(x, rfs, [3, 3], 2, padding=padding, scope='pool1')
    x, rfs = util.conv2d(x, rfs, 256, [5, 5], padding=padding, scope='conv2')
    x, rfs = util.max_pool2d(x, rfs, [3, 3], 2, padding=padding, scope='pool2')
    x, rfs = util.conv2d(x, rfs, 384, [3, 3], padding=padding, scope='conv3')
    x, rfs = util.conv2d(x, rfs, 384, [3, 3], padding=padding, scope='conv4')
    x, rfs = util.conv2d(x, rfs, 256, [3, 3], padding=padding, scope='conv5')
    return x, rfs
