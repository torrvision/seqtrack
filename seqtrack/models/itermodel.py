import tensorflow as tf

from seqtrack.models import interface


class ModelFromIterModel(interface.Model):

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

        predictions = [None for _ in range(ntimesteps)]
        losses = [None for _ in range(ntimesteps)]
        state = init_state
        for i in range(ntimesteps):
            predictions[i], state, losses[i] = self._model.next(frames[i], state)
        final_state = state

        extra_loss = self._model.end()
        loss = tf.reduce_mean(losses) + extra_loss

        predictions = tf.stack(predictions, axis=1)
        outputs = {'y': predictions}
        return outputs, loss, init_state, final_state
