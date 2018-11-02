from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seqtrack.models import interface as models_interface
from seqtrack.helpers import stack_dict


class ModelFromIterModel(models_interface.Model):

    def __init__(self, iter_model):
        self._model = iter_model

    def instantiate(self, example, run_opts, enable_loss=True,
                    image_summaries_collections=None):
        '''Instantiates the graph of a model.

        Returns prediction in the image reference frame,
        and prediction_crop in the window reference frame.
        The prediction in the window reference frame can be used to apply the loss.

        Example:
            prediction_crop, viewport, prediction, init_state, final_state = model_graph.process_sequence(
                model, example, run_opts, batchsz, ntimesteps, im_size)

        Args:
            example has fields:
                x0, y0, x, y, y_is_valid

        Returns:
            prediction_crop: Model prediction in reference frame of viewport.
            viewport: Search region of model.
            prediction: Model prediction in image reference frame.
            init_state: Initial state.
            final_state: Final state.
                Should be possible to feed final state to initial state to continue
                from end of sequence.
        '''
        # TODO: Aspect, viewport.
        frame = {'x': example['x0'], 'y': example['y0']}
        init_state = self._model.start(frame, example['aspect'], run_opts, enable_loss=enable_loss,
                                       image_summaries_collections=image_summaries_collections)

        frames = {
            'x': tf.unstack(example['x'], axis=1),
            'y': tf.unstack(example['y'], axis=1),
            'y_is_valid': tf.unstack(example['y_is_valid'], axis=1),
        }
        ntimesteps = len(frames['x'])
        frames = [{k: frames[k][i] for k in frames} for i in range(ntimesteps)]

        outputs = [None for _ in range(ntimesteps)]
        losses = [None for _ in range(ntimesteps)]
        state = init_state
        for i in range(ntimesteps):
            outputs[i], state, losses[i] = self._model.next(frames[i], state)
            assert 'y' in outputs[i]
        outputs = stack_dict(outputs, axis=1)
        losses = stack_dict(losses)
        # Compute mean over frames.
        losses = {k: tf.reduce_mean(v) for k, v in losses.items()}
        final_state = state

        extra_losses = self._model.end()
        _assert_no_keys_in_common(losses, extra_losses)
        losses.update(extra_losses)
        return outputs, losses, init_state, final_state

    def init(self, sess):
        self._model.init(sess)


def _assert_no_keys_in_common(a, b):
    intersection = set(a.keys()).intersection(set(b.keys()))
    if intersection:
        raise ValueError('keys in common: {}'.format(str(intersection)))
