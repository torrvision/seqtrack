from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers
from seqtrack import sample
from seqtrack.models import itermodel


class NullTracker(object):

    def __init__(self, mode, params, example_type=None):
        self.mode = mode
        self.example_type = example_type

    def train(self, example, run_opts, scope='model'):
        raise RuntimeError('not trainable')

    def start(self, features_init, run_opts, name='start'):
        with tf.name_scope(name) as scope:
            rect = features_init['rect']
            state = {
                'run_opts': run_opts,
                'rect': rect,
            }
            return state

    def next(self, features, labels, state, name='timestep'):
        with tf.name_scope(name) as scope:
            run_opts = state['run_opts']
            rect = state['rect']
            losses = {}
            outputs = {'rect': rect}
            state = {
                'run_opts': run_opts,
                'rect': rect,
            }
            return outputs, state, losses

    def end(self):
        losses = {}
        return losses
