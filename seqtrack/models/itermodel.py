from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
nest = tf.contrib.framework.nest

from seqtrack import helpers
from seqtrack.models import interface as models_interface


ArgsUnroll = collections.namedtuple('ArgsUnroll', [
    'inputs_init',
    'inputs',
    'labels',
])

OperationsUnroll = collections.namedtuple('OperationsUnroll', [
    'predictions',
    'losses',
    'state_init',
    'state_final',
])


def instantiate_unroll(iter_model_fn, args, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        args: ArgsUnroll

    Returns:
        OperationsUnroll
    '''
    with tf.variable_scope(scope):
        # TODO: Aspect.
        # frame = {'x': example['x0'], 'y': example['y0']}
        state_init = iter_model_fn.start(args.inputs_init, enable_loss=True)
        inputs = helpers.unstack_structure(args.inputs, axis=1)
        labels = helpers.unstack_structure(args.labels, axis=1)
        ntimesteps = len(inputs)
        predictions = [None for _ in range(ntimesteps)]
        losses = [None for _ in range(ntimesteps)]
        state = state_init
        for i in range(ntimesteps):
            predictions[i], state, losses[i] = iter_model_fn.next(inputs[i], labels[i], state)
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


# class ModelFromIterModel(models_interface.Model):
# 
#     def __init__(self, iter_model):
#         self._model = iter_model
# 
#     def instantiate(self, inputs_init, #example, run_opts,
#                     enable_loss=True,
#                     image_summaries_collections=None):
#         '''Instantiates the graph of a model.
# 
#         Returns prediction in the image reference frame,
#         and prediction_crop in the window reference frame.
#         The prediction in the window reference frame can be used to apply the loss.
# 
#         Example:
#             prediction_crop, prediction, init_state, final_state = model_graph.process_sequence(
#                 model, example, run_opts, batchsz, ntimesteps, im_size)
# 
#         Args:
#             example has fields:
#                 x0, y0, x, y, y_is_valid
# 
#         Returns:
#             prediction_crop: Model prediction.
#             prediction: Model prediction in image reference frame.
#             init_state: Initial state.
#             final_state: Final state.
#                 Should be possible to feed final state to initial state to continue
#                 from end of sequence.
#         '''
#         # TODO: Aspect.
#         frame = {'x': example['x0'], 'y': example['y0']}
#         init_state = self._model.start(frame, example['aspect'], run_opts, enable_loss=enable_loss,
#                                        image_summaries_collections=image_summaries_collections)
# 
#         frames = {
#             'x': tf.unstack(example['x'], axis=1),
#             'y': tf.unstack(example['y'], axis=1),
#             'y_is_valid': tf.unstack(example['y_is_valid'], axis=1),
#         }
#         ntimesteps = len(frames['x'])
#         frames = [{k: frames[k][i] for k in frames} for i in range(ntimesteps)]
# 
#         outputs = [None for _ in range(ntimesteps)]
#         losses = [None for _ in range(ntimesteps)]
#         state = init_state
#         for i in range(ntimesteps):
#             outputs[i], state, losses[i] = self._model.next(frames[i], state)
#             assert 'y' in outputs[i]
#         outputs = helpers.stack_dict(outputs, axis=1)
#         losses = helpers.stack_dict(losses)
#         # Compute mean over frames.
#         losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
#         final_state = state
# 
#         extra_losses = self._model.end()
#         _assert_no_keys_in_common(losses, extra_losses)
#         losses.update(extra_losses)
#         return outputs, losses, init_state, final_state
# 
#     def init(self, sess):
#         self._model.init(sess)


def _assert_no_keys_in_common(a, b):
    intersection = set(a.keys()).intersection(set(b.keys()))
    if intersection:
        raise ValueError('keys in common: {}'.format(str(intersection)))


ArgsIter = collections.namedtuple('ArgsIter', [
    'inputs_init',
    'inputs_curr',
    'labels_curr',
])


OperationsIterAssign = collections.namedtuple('OperationsIterAssign', [
    'assign_state_init',
    'assign_state_curr',
    'predictions_curr',
])


def instantiate_iter_assign(iter_model_fn, args, local_scope, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        args: ArgsIter

    Returns:
        OperationsIterAssign
    '''
    with tf.variable_scope(scope):
        # TODO: Add `mode`? (like `enable_loss`)
        # Initialize model in first frame.
        state_init = iter_model_fn.start(args.inputs_init)
        with tf.variable_scope(local_scope):  # Outside the model scope.
            # Local variables are ignored by the saver.
            state_prev = _get_local_variable_like_structure(state_init, scope='state')
        assign_state_init = _assign_structure(state_prev, state_init)
        # Make prediction in current frame.
        predictions_curr, state_curr, _ = iter_model_fn.next(args.inputs_curr, None, state_prev)
        assign_state_curr = _assign_structure(state_prev, state_curr, validate_shape=False)
        return OperationsIterAssign(
            assign_state_init=assign_state_init,
            assign_state_curr=assign_state_curr,
            predictions_curr=predictions_curr,
        )


class InstanceAssign(object):

    def __init__(self, args, ops):
        '''
        Args:
            args: ArgsIter
            ops: OperationsIterAssign
        '''
        self.args = args
        self.ops = ops

    def start(self, sess, inputs_init):
        sess.run(
            self.ops.assign_state_init,
            feed_dict=flatten_dict(self.args.inputs_init, inputs_init),
        )

    def next(self, sess, inputs_curr):
        predictions_curr, _ = sess.run(
            (self.ops.predictions_curr, self.ops.assign_state_curr),
            feed_dict=flatten_dict(self.args.inputs_curr, inputs_curr),
        )
        return predictions_curr


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


def flatten_dict(keys, values):
    nest.assert_shallow_structure(keys, values)
    return dict(zip(nest.flatten(keys),
                    nest.flatten_up_to(keys, values)))


OperationsIterFeed = collections.namedtuple('OperationsIterFeed', [
    'state_init',
    'state_prev',
    'predictions_curr',
    'state_curr',
])


def instantiate_iter_feed(iter_model_fn, args, scope='model'):
    '''Like model_fn() in tf.Estimator.

    Args:
        iter_model_fn: IterModel
        args: ArgsIter

    Returns:
        OperationsIterFeed
    '''
    with tf.variable_scope(scope):
        # TODO: Add `mode`? (like `enable_loss`)
        # Initialize model in first frame.
        state_init = iter_model_fn.start(args.inputs_init)
        with tf.name_scope('state_prev'):
            state_prev = nest.map_structure(_placeholder_like, state_init)
        # Make prediction in current frame.
        predictions_curr, state_curr, _ = iter_model_fn.next(args.inputs_curr, None, state_prev)
        return OperationsIterFeed(
            state_init=state_init,
            state_prev=state_prev,
            predictions_curr=predictions_curr,
            state_curr=state_curr,
        )


def _placeholder_like(x):
    return tf.placeholder(x.dtype, x.shape, name=_escape(x.name))


class InstanceFeed(object):

    def __init__(self, args, ops):
        '''
        Args:
            args: ArgsIter
            ops: OperationsIterFeed
        '''
        self.args = args
        self.ops = ops

        self._state = None

    def start(self, sess, inputs_init):
        self._state = sess.run(
            self.ops.state_init,
            feed_dict=flatten_dict(self.args.inputs_init, inputs_init),
        )

    def next(self, sess, inputs_curr):
        predictions_curr, self._state = sess.run(
            (self.ops.predictions_curr, self.ops.state_curr),
            feed_dict=helpers.merge_dicts(
                flatten_dict(self.args.inputs_curr, inputs_curr),
                flatten_dict(self.ops.state_prev, self._state)))
        return predictions_curr
