from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

import functools

from seqtrack import cnn


def partial(f, *a, **b):
    '''Returns a function h such that h(x) = f(x, *a, **b).'''
    def caller(x, *u, **v):
        g = functools.partial(f, x, *a, **b)
        return g(*u, **v)
    return caller


def evaluate_until(layers, inputs, output_layer, output_kwargs=None, freeze_until_layer=None):
    '''
    Args:
        layers: List of (name, function) tuples.
        inputs: Tensor
        output_layer: Name of layer to take as output.
            Following layers are not instantiated.
            Layer is evaluated with **output_kwargs.
        frozen_until_layer: Freeze all layers up to (and including) this layer.
            Following layers will return to the default behaviour.
            The following layers will be frozen (using slim.arg_scope):
                cnn.slim_conv2d, slim.batch_norm
    '''
    output_kwargs = output_kwargs or {}
    layer_names = [name for name, func in layers]
    if output_layer not in layer_names:
        raise ValueError('output layer not found: "{}"'.format(output_layer))

    if freeze_until_layer and freeze_until_layer not in layer_names:
        raise ValueError('frozen layer not found: "{}"'.format(freeze_until_layer))
    # Freeze initial layers if freeze_until_layer is specified.
    frozen = bool(freeze_until_layer)

    net = inputs
    end_points = {}
    for name, func in layers:
        # Use arg_scope to achieve trainable=False because:
        # 1) It is compatible for max_pool2d layers (no trainable parameter).
        # 2) It makes it possible to freeze internal batch_norm layers too.
        # Set trainable to False if frozen, otherwise take default behaviour.
        args = dict(trainable=False) if frozen else {}
        with slim.arg_scope([cnn.slim_conv2d, slim.batch_norm], **args):
            if name == output_layer:
                net = func(net, scope=name, **output_kwargs)
                break
            else:
                net = func(net, scope=name)
                end_points[name] = net
        # If this is the last frozen layer, then un-freeze for next layer.
        if name == freeze_until_layer:
            frozen = False
    return net, end_points
