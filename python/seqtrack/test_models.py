from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slim_nets

from tensorflow.contrib.layers.python.layers.utils import n_positive_integers

import collections
import functools

from seqtrack import cnn
from seqtrack import sample
from seqtrack import train
from seqtrack import models
from seqtrack.helpers import trySubTest


class TestModels(tf.test.TestCase):

    def test_instantiate(self):
        for model_name, create_iter_model_fn in models.BY_NAME.items():
            with trySubTest(self, model=model_name):
                tf.reset_default_graph()
                iter_model_fn = create_iter_model_fn(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    example_type=sample.ExampleTypeKeys.CONSECUTIVE,
                    params=dict(),
                )
                b, n, h, w = 8, 2, 480, 480
                example = sample.ExampleSequence(
                    features_init={
                        'image': {'data': tf.placeholder(tf.float32, [b, h, w, 3])},
                        'aspect': tf.placeholder(tf.float32, [b]),
                        'rect': tf.placeholder(tf.float32, [b, 4]),
                    },
                    features={
                        'image': {'data': tf.placeholder(tf.float32, [b, n, h, w, 3])},
                    },
                    labels={
                        'valid': tf.placeholder(tf.bool, [b, n]),
                        'rect': tf.placeholder(tf.float32, [b, n, 4]),
                    },
                )
                run_opts = train._make_run_opts_placeholders()

                with tf.variable_scope('model', reuse=False) as vs:
                    ops = iter_model_fn.train(example, run_opts, scope=vs)
