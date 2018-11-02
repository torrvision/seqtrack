# https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/alexnet.py

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

from seqtrack import cnn
from . import util

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# Replace operations with version that calculates receptive field.
conv2d = cnn.slim_conv2d  # slim.conv2d
max_pool2d = cnn.slim_max_pool2d  # slim.max_pool2d
dropout = cnn.slim_dropout  # slim.dropout


def alexnet_v2_arg_scope(weight_decay=0.0005, conv_padding='VALID', pool_padding='VALID'):
    with slim.arg_scope([conv2d],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([conv2d], padding=conv_padding):
            with slim.arg_scope([max_pool2d], padding=pool_padding) as arg_sc:
                return arg_sc


def alexnet_v2(inputs,
               is_training=True,
               conv1_stride=4,
               output_layer='conv5',
               output_activation_fn=tf.nn.relu,
               freeze_until_layer=None,
               scope='alexnet_v2'):
    """AlexNet version 2.

    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224 or set
          global_pool=True. To use in fully convolutional mode, set
          spatial_squeeze to false.
          The LRN layers have been removed and change the initializers from
          random_normal_initializer to xavier_initializer.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: the number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer are returned instead.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.
      spatial_squeeze: whether or not should squeeze the spatial dimensions of the
        logits. Useful to remove unnecessary dimensions for classification.
      scope: Optional scope for the variables.
      global_pool: Optional boolean flag. If True, the input to the classification
        layer is avgpooled to size 1x1, for any input size. (This is not part
        of the original AlexNet.)

    Returns:
      net: the output of the logits layer (if num_classes is a non-zero integer),
        or the non-dropped-out input to the logits layer (if num_classes is 0
        or None).
      end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([conv2d, max_pool2d],
                            outputs_collections=[end_points_collection]):
            layers = [
                ('conv1', util.partial(conv2d, 64, [11, 11], conv1_stride, padding='VALID')),
                ('pool1', util.partial(max_pool2d, [3, 3], 2)),
                ('conv2', util.partial(conv2d, 192, [5, 5])),
                ('pool2', util.partial(max_pool2d, [3, 3], 2)),
                ('conv3', util.partial(conv2d, 384, [3, 3])),
                ('conv4', util.partial(conv2d, 384, [3, 3])),
                ('conv5', util.partial(conv2d, 256, [3, 3])),
                ('pool5', util.partial(max_pool2d, [3, 3], 2)),
            ]
            net = util.evaluate_until(layers, inputs, output_layer,
                                      output_kwargs=dict(activation_fn=output_activation_fn),
                                      freeze_until_layer=freeze_until_layer)

            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


# alexnet_v2.default_image_size = 224
