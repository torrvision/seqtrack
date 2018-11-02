from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Model(object):
    '''Model can instantiate a model.'''

    def instantiate(self, example, run_opts, enable_loss,
                    image_summaries_collections=None):
        '''
        Args:
            example has fields:
                example['x0']
                example['y0']
                example['x']
                example['y']
                example['y_is_valid']
                example['aspect']

            run_opts has fields:
                run_opts['is_training']

            enable_loss is True to add loss terms to the graph.

        Returns:
            outputs, losses, init_state, final_state

        The losses are returned as a dictionary.
        '''
        raise NotImplementedError()

    def init(self, sess):
        raise NotImplementedError()


class IterModel(object):
    '''
    IterModel has the following interface.
    Note that these functions are called to instantiate the TF graph.
    At test time, we simply do sess.run() using the resulting tensors.

    The interface is used like this:

        state = model.start(frames[0], run_opts, enable_loss)
        for i in range(ntimesteps):
            outputs[i], state, losses[i] = model.next(frames[i], state)
        extra_loss = model.end()

    Each frame is a dict with:
        frame['x'] is an image
        frame['y']
        frame['y_is_valid']

    Each state is a nested structure (dicts, lists, tuples) of tensors.
    '''

    def start(self, frame, aspect, run_opts, enable_loss,
              image_summaries_collections=None):
        '''Processes the first frame.

        Args:
            frame is a frame dict

        Returns:
            Initial state.

        Beware: the initial state will be fed using feed_dict during tracking.
        It may be necessary to use tf.identity() to prevent undesired effects.
        '''
        raise NotImplementedError()

    def next(self, frame, prev_state):
        '''next() processes another frame and returns a prediction.

        Args:
            frame is a frame dict
                It may have 'y' (and 'y_is_valid') but at test time these will be invalid.

        Returns:
            outputs, state, losses

        The method next() should not modify member variables of the model.
        Instead, it should pass all information to the next frame using state.

        The method next() should only use frame['y'] during training.

        The losses are returned as a dict in each frame.
        The overall loss will be the mean of the per-frame losses.

        The loss operations are only added to the graph if enable_loss is true.
        '''
        raise NotImplementedError()

    def end(self):
        '''end() provides a final chance to add summaries, losses, etc.

        Returns:
            An additional loss dict.
        '''
        raise NotImplementedError()
