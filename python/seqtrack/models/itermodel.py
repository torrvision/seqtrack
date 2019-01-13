from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
nest = tf.contrib.framework.nest

from seqtrack import helpers


# Do not include `run_opts` in ExampleXxx because examples are what gets batched.
# That is, each batch contains a collection of examples, but `run_opts` is global.


ExampleUnroll = collections.namedtuple('ExampleUnroll', [
    'features_init',
    'features',
    'labels',
])


OperationsUnroll = collections.namedtuple('OperationsUnroll', [
    'predictions',
    'losses',
    'state_init',
    'state_final',
])


def instantiate_unroll(iter_model_fn, example, run_opts, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        example: ExampleUnroll

    Returns:
        OperationsUnroll

    Can feed either `example.features_init` or `ops.state_init` when obtaining predictions.
    '''
    with tf.variable_scope(scope):
        state_init = iter_model_fn.start(example.features_init, run_opts, enable_loss=True)
        features = helpers.unstack_structure(example.features, axis=1)
        labels = helpers.unstack_structure(example.labels, axis=1)
        ntimesteps = len(features)
        predictions = [None for _ in range(ntimesteps)]
        losses = [None for _ in range(ntimesteps)]
        state = state_init
        for i in range(ntimesteps):
            predictions[i], state, losses[i] = iter_model_fn.next(features[i], labels[i], state)
        predictions = helpers.stack_structure(predictions, axis=1)
        losses = helpers.stack_structure(losses, axis=0)
        # Compute mean over frames.
        losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
        state_final = state

        extra_losses = iter_model_fn.end()
        _assert_no_keys_in_common(losses, extra_losses)
        losses.update(extra_losses)
        return OperationsUnroll(
            predictions=predictions,
            losses=losses,
            state_init=state_init,
            state_final=state_final,
        )


def _assert_no_keys_in_common(a, b):
    intersection = set(a.keys()).intersection(set(b.keys()))
    if intersection:
        raise ValueError('keys in common: {}'.format(str(intersection)))


ExampleIter = collections.namedtuple('ExampleIter', [
    'features_init',
    'features_curr',
    'labels_curr',
])


OperationsIterAssign = collections.namedtuple('OperationsIterAssign', [
    'assign_state_init',
    'assign_state_curr',
    'predictions_curr',
])


def instantiate_iter_assign(iter_model_fn, example, run_opts, local_scope, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        example: ExampleIter

    Returns:
        OperationsIterAssign
    '''
    with tf.variable_scope(scope):
        # TODO: Add `mode`? (like `enable_loss`)
        # Initialize model in first frame.
        state_init = iter_model_fn.start(example.features_init, run_opts)
        with tf.variable_scope(local_scope):  # Outside the model scope.
            # Local variables are ignored by the saver.
            state_prev = _get_local_variable_like_structure(state_init, scope='state')
        assign_state_init = _assign_structure(state_prev, state_init)
        # Make prediction in current frame.
        predictions_curr, state_curr, _ = iter_model_fn.next(example.features_curr, None, state_prev)
        assign_state_curr = _assign_structure(state_prev, state_curr, validate_shape=False)
        return OperationsIterAssign(
            assign_state_init=assign_state_init,
            assign_state_curr=assign_state_curr,
            predictions_curr=predictions_curr,
        )


def _get_local_variable_like_structure(structure, scope=None):
    with tf.variable_scope(scope, 'like_structure'):
        return nest.map_structure(_get_local_variable_like, structure)


def _get_local_variable_like(x):
    '''
    Shape of `x` must be known!
    '''
    return tf.get_local_variable(_escape(x.name),
                                 initializer=tf.zeros(shape=x.shape, dtype=x.dtype))


def _escape(s):
    return s.replace('/', '_').replace(':', '_')


def _assign_structure(ref, value, validate_shape=None, name='assign_structure'):
    with tf.name_scope(name) as scope:
        nest.assert_same_structure(ref, value)
        assign_ops = [tf.assign(r, v, validate_shape=validate_shape)
                      for r, v in zip(nest.flatten(ref), nest.flatten(value))]
        # with tf.control_dependencies(update_ops):
        #     return tf.no_op(name=scope)
        return assign_ops


OperationsIterFeed = collections.namedtuple('OperationsIterFeed', [
    'state_init',
    'state_prev',
    'predictions_curr',
    'state_curr',
])


def instantiate_iter_feed(iter_model_fn, example, run_opts, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        example: ExampleIter

    Returns:
        OperationsIterFeed
    '''
    with tf.variable_scope(scope):
        # TODO: Add `mode`? (like `enable_loss`)
        # Initialize model in first frame.
        state_init = iter_model_fn.start(example.features_init, run_opts)
        with tf.name_scope('state_prev'):
            state_prev = nest.map_structure(_placeholder_like, state_init)
        # Make prediction in current frame.
        predictions_curr, state_curr, _ = iter_model_fn.next(example.features_curr, None, state_prev)
        return OperationsIterFeed(
            state_init=state_init,
            state_prev=state_prev,
            predictions_curr=predictions_curr,
            state_curr=state_curr,
        )


def _placeholder_like(x):
    return tf.placeholder(x.dtype, x.shape, name=_escape(x.name))


# Provide a Tracker interface for using an instantiated model.
# The `run_opts` params are external to this interface.


class TrackerAssign(object):

    def __init__(self, example, ops):
        '''
        Args:
            example: ExampleIter
            ops: OperationsIterAssign
        '''
        self.example = example
        self.ops = ops

    def start(self, sess, features_init):
        sess.run(
            self.ops.assign_state_init,
            feed_dict=helpers.flatten_dict(self.example.features_init, features_init),
        )

    def next(self, sess, features_curr):
        predictions_curr, _ = sess.run(
            (self.ops.predictions_curr, self.ops.assign_state_curr),
            feed_dict=helpers.flatten_dict(self.example.features_curr, features_curr),
        )
        return predictions_curr


class TrackerFeed(object):

    def __init__(self, example, ops):
        '''
        Args:
            example: ExampleIter
            ops: OperationsIterFeed
        '''
        self.example = example
        self.ops = ops

        self._state = None

    def start(self, sess, features_init):
        self._state = sess.run(
            self.ops.state_init,
            feed_dict=helpers.flatten_dict(self.example.features_init, features_init),
        )

    def next(self, sess, features_curr):
        predictions_curr, self._state = sess.run(
            (self.ops.predictions_curr, self.ops.state_curr),
            feed_dict=helpers.merge_dicts(
                helpers.flatten_dict(self.example.features_curr, features_curr),
                helpers.flatten_dict(self.ops.state_prev, self._state)))
        return predictions_curr
