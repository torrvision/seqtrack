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
from seqtrack import util_test
from seqtrack.models import regress


class TestRegress(tf.test.TestCase):

    def test_instantiate(self):
        iter_model_fn = regress.MotionRegressor(
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
            ops = iter_model_fn.train(example, run_opts=run_opts, scope=vs)

    def test_multiscale_error_translation(self):
        response_size = 5
        num_scales = 1
        translation_stride = 10
        scale_step = 2
        base_target_size = 30

        gt_translation = [-10, -20]
        gt_size = 30

        # The error is `pred - gt` accounting for stride.
        translation_want = [[
            [[-10,  0], [ 0,  0], [10,  0], [20,  0], [30,  0]],
            [[-10, 10], [ 0, 10], [10, 10], [20, 10], [30, 10]],
            [[-10, 20], [ 0, 20], [10, 20], [20, 20], [30, 20]],
            [[-10, 30], [ 0, 30], [10, 30], [20, 30], [30, 30]],
            [[-10, 40], [ 0, 40], [10, 40], [20, 40], [30, 40]],
        ]]  # [s=1, h=5, w=5, 2]
        scale_want = [0]  # [s=1]

        gt_translation = _make_constant_batch(gt_translation)
        gt_size = _make_constant_batch(gt_size)
        translation_want = _make_constant_batch(translation_want)
        scale_want = _make_constant_batch(scale_want)

        translation_actual, scale_actual = regress.multiscale_error(
            response_size, num_scales, translation_stride, scale_step, base_target_size,
            gt_translation, gt_size)

        self.assertAllClose(translation_want, translation_actual)
        self.assertAllClose(scale_want, scale_actual)

    def test_multiscale_error_scale(self):
        response_size = 1
        num_scales = 5
        translation_stride = 10
        scale_step = 2
        base_target_size = 30

        gt_translation = [0, 0]
        gt_size = 60

        # The error is `pred - gt`.
        translation_want = [[
            [[0, 0]],
        ]] * num_scales  # [s=5, h=1, w=1, 2]
        scale_want = np.log([2 ** i for i in [-3, -2, -1, 0, 1]])  # [s=5]

        gt_translation = _make_constant_batch(gt_translation)
        gt_size = _make_constant_batch(gt_size)
        translation_want = _make_constant_batch(translation_want)
        scale_want = _make_constant_batch(scale_want)

        translation_actual, scale_actual = regress.multiscale_error(
            response_size, num_scales, translation_stride, scale_step, base_target_size,
            gt_translation, gt_size)

        self.assertAllClose(translation_want, translation_actual)
        self.assertAllClose(scale_want, scale_actual)

    def test_losses(self):
        response_size = 7
        num_scales = 3
        translation_stride = 10
        scale_step = 2
        base_target_size = 30
        scores_shape = (1, num_scales) + n_positive_integers(2, response_size) + (1,)

        gt_translation = [-20, -40]
        gt_size = 60
        scores = tf.random.normal(scores_shape, dtype=tf.float32)

        losses = {
            'sigmoid_hard': dict(
                method='sigmoid',
                params=dict(balanced=True,
                            label_method='hard',
                            label_params=dict(translation_radius_pos=0.2,
                                              translation_radius_neg=0.5,
                                              scale_radius_pos=1.1,
                                              scale_radius_neg=1.3))),
            'sigmoid_hard_binary': dict(
                method='sigmoid',
                params=dict(balanced=True,
                            label_method='hard_binary',
                            label_params=dict(translation_radius=0.2,
                                              scale_radius=1.2))),
        }

        for loss_name, loss_kwargs in losses.items():
            with util_test.try_sub_test(self, loss=loss_name):
                _, loss = regress.compute_loss(
                    scores, num_scales, translation_stride, scale_step, base_target_size,
                    _make_constant_batch(gt_translation),
                    _make_constant_batch(gt_size),
                    **loss_kwargs)
                self.assertEqual(len(loss.shape), 1)
                with self.test_session():
                    self.assertTrue(np.all(np.isfinite(loss.eval())))


    def test_label_fns(self):
        response_size = 7
        num_scales = 3
        translation_stride = 10
        scale_step = 2
        base_target_size = 30
        scores_shape = (1, num_scales) + n_positive_integers(2, response_size) + (1,)

        gt_translation = [-20, -40]
        gt_size = 60

        label_fns = {
            'hard': dict(
                translation_radius_pos=0.2,
                translation_radius_neg=0.5,
                scale_radius_pos=1.1,
                scale_radius_neg=1.3,
            ),
            'hard_binary': dict(
                translation_radius=0.2,
                scale_radius=1.2,
            ),
        }

        for name, kwargs in label_fns.items():
            with util_test.try_sub_test(self, label_fn=name):
                with self.test_session():
                    label_fn = regress.LABEL_FNS[name]
                    _, labels, weights = label_fn(
                        response_size, num_scales, translation_stride, scale_step, base_target_size,
                        _make_constant_batch(gt_translation),
                        _make_constant_batch(gt_size),
                        **kwargs)

                    # labels: [b, s, h, w, c]
                    assert(len(labels.shape) == 5)
                    self.assertAllGreaterEqual(weights, 0)
                    sum_positive = tf.reduce_sum(weights * labels, axis=(-4, -3, -2, -1))
                    sum_negative = tf.reduce_sum(weights * (1 - labels), axis=(-4, -3, -2, -1))
                    self.assertAllGreater(sum_positive, 0)
                    self.assertAllGreater(sum_negative, 0)


def _make_constant_batch(x):
    return tf.expand_dims(tf.constant(x, dtype=tf.float32), axis=0)
