from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

import functools


def partial(f, *a, **b):
    '''Returns a function h such that h(x) = f(x, *a, **b).'''
    def caller(x, *u, **v):
        g = functools.partial(f, x, *a, **b)
        return g(*u, **v)
    return caller


def evaluate_until(layers, inputs, output_layer, output_kwargs=None):
    '''
    Args:
        layers: List of (name, function) tuples.
    '''
    output_kwargs = output_kwargs or {}
    if output_layer not in [name for name, func in layers]:
        raise ValueError('output layer not found: "{}"'.format(output_layer))
    net = inputs
    for name, func in layers:
        if name == output_layer:
            net = func(net, scope=name, **output_kwargs)
            break
        else:
            net = func(net, scope=name)
    return net
