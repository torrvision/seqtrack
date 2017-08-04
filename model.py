'''This file describes several different models.

A Model has the following interface::

    state_init = model.init(example_init, run_opts)

        example_init['x0']
        example_init['y0']
        run_opts['use_gt']

    prediction_t, window_t, state_t = model.step(example_t, state_{t-1})

        example_t['x']
        example_t['y'] (optional)
        prediction_t['y']
        prediction_t['hmap'] (optional)
        prediction_t['hmap_softmax'] (optional)

Note that step() returns a window in the current frame,
and specifies its prediction within this rectangle.
The window represents a search area in the current image.
This enables the loss to be computed in the reference frame of the search area.
(Particularly important for the heatmap loss.)
The model is free to choose its own window.

It is then possible to process a long sequence by dividing it into chunks
of length k and feeding state_{k-1} to state_init.

A Model also has the following properties::

    model.batch_size    # Batch size of model instance, either None or an integer.
    model.image_size    # Tuple of image size.
'''

import pdb
import functools
import itertools
import tensorflow as tf
from tensorflow.contrib import slim
import math
import numpy as np
import os

import cnnutil
import geom
from helpers import merge_dims, most_static_shape
from upsample import upsample

concat = tf.concat if hasattr(tf, 'concat') else tf.concat_v2

def convert_rec_to_heatmap(rec, frmsz, dtype=tf.float32, min_size=None):
    '''Create heatmap from rectangle
    Args:
        rec: [batchsz x ntimesteps x 4] ground-truth rectangle labels
    Return:
        heatmap: [batchsz x ntimesteps x o.frmsz x o.frmsz x 2] # fg + bg
    '''
    with tf.name_scope('heatmaps') as scope:
        # JV: This causes a seg-fault in save when two loss functions are constructed?!
        # masks = []
        # for t in range(o.ntimesteps):
        #     masks.append(get_masks_from_rectangles(rec[:,t], o, kind='bg'))
        # return tf.stack(masks, axis=1, name=scope)
        rec, unmerge = merge_dims(rec, 0, 2)
        masks = get_masks_from_rectangles(rec, frmsz, dtype=dtype, kind='bg', min_size=min_size)
        return unmerge(masks, 0)

def get_masks_from_rectangles(rec, frmsz, dtype=tf.float32, kind='fg', typecast=True, min_size=None, name='mask'):
    with tf.name_scope(name) as scope:
        # create mask using rec; typically rec=y_prev
        # rec -- [b, 4]
        rec *= float(frmsz)
        # x1, y1, x2, y2 -- [b]
        x1, y1, x2, y2 = tf.unstack(rec, axis=1)
        if min_size is not None:
            x1, y1, x2, y2 = enforce_min_size(x1, y1, x2, y2, min_size=min_size)
        # grid_x -- [1, frmsz]
        # grid_y -- [frmsz, 1]
        grid_x = tf.expand_dims(tf.cast(tf.range(frmsz), dtype), 0)
        grid_y = tf.expand_dims(tf.cast(tf.range(frmsz), dtype), 1)
        # resize tensors so that they can be compared
        # x1, y1, x2, y2 -- [b, 1, 1]
        x1 = tf.expand_dims(tf.expand_dims(x1, -1), -1)
        x2 = tf.expand_dims(tf.expand_dims(x2, -1), -1)
        y1 = tf.expand_dims(tf.expand_dims(y1, -1), -1)
        y2 = tf.expand_dims(tf.expand_dims(y2, -1), -1)
        # masks -- [b, frmsz, frmsz]
        masks = tf.logical_and(
            tf.logical_and(tf.less_equal(x1, grid_x), 
                           tf.less_equal(grid_x, x2)),
            tf.logical_and(tf.less_equal(y1, grid_y), 
                           tf.less_equal(grid_y, y2)))

        if kind == 'fg': # only foreground mask
            masks = tf.expand_dims(masks, 3) # to have channel dim
        elif kind == 'bg': # add background mask
            masks_bg = tf.logical_not(masks)
            masks = concat(
                    (tf.expand_dims(masks,3), tf.expand_dims(masks_bg,3)), 3)
        if typecast: # type cast so that it can be concatenated with x
            masks = tf.cast(masks, dtype)
        return masks

def enforce_min_size(x1, y1, x2, y2, min_size, name='min_size'):
    with tf.name_scope(name) as scope:
        # Ensure that x2-x1 > 1
        xc, xs = 0.5*(x1 + x2), x2-x1
        yc, ys = 0.5*(y1 + y2), y2-y1
        # TODO: Does this propagate NaNs?
        xs = tf.maximum(min_size, xs)
        ys = tf.maximum(min_size, ys)
        x1, x2 = xc-xs/2, xc+xs/2
        y1, y2 = yc-ys/2, yc+ys/2
        return x1, y1, x2, y2


class SimpleSearch:

    # NOTE: To reduce confusion, avoid putting tensors in member variables.

    # TODO: Make stat part of the model (i.e. a variable?)
    def __init__(self, ntimesteps, frmsz, batchsz, output_mode, stat, weight_decay=0.0,
            summaries_collections=None,
            image_summaries_collections=None,
            # Model parameters:
            # use_rnn=True,
            use_heatmap=False,
            use_batch_norm=False, # Caution when use_rnn is True.
            object_centric=False,
            # normalize_size=False,
            # normalize_first_only=False
            ):
        # TODO: Possible to automate this? Yuck!
        self.ntimesteps                  = ntimesteps
        self.frmsz                       = frmsz
        self.batchsz                     = batchsz
        self.output_mode                 = output_mode
        self.stat                        = stat
        self.weight_decay                = weight_decay
        self.summaries_collections       = summaries_collections
        self.image_summaries_collections = image_summaries_collections
        # Model parameters:
        # self.use_rnn        = use_rnn
        self.use_heatmap    = use_heatmap
        self.use_batch_norm = use_batch_norm
        self.object_centric = object_centric

        # Public model properties:
        self.batch_size = None # Model accepts variable batch size.

        # Model state.
        self._run_opts = None

        if self.object_centric:
            self._window_model = ConditionalWindow(
                train_model=InitialWindow(),
                test_model=MovingAverageWindow(0.5),
            )
        else:
            # TODO: May be more efficient to avoid cropping if using whole window?
            self._window_model = ConditionalWindow(
                train_model=WholeImageWindow(batchsz=batchsz),
                test_model=WholeImageWindow(batchsz=batchsz),
            )
        self._window_state_keys = None


    def init(self, example, run_opts):
        state = {}
        self._run_opts = run_opts
        self._init_example = example

        with tf.name_scope('extract_window'):
            window_state = self._window_model.init(example, run_opts['is_training'])
            self._window_state_keys = window_state.keys()
            # Use the search window for the second frame to crop the initial example.
            window = self._window_model.window(window_state)
            example = {
                'x0': self._extract_window(example['x0'], window),
                'y0': geom.crop_rect(example['y0'], window),
            }
            # Do not pass gradients back to window.
            example = {k: tf.stop_gradient(v) for k, v in example.items()}

        example['x0'] = self._whiten(example['x0'])

        # Visualize supervision rectangle in window.
        tf.summary.image('frame_0',
            tf.image.draw_bounding_boxes(
                _normalize_image_range(example['x0'][0:1]),
                geom.rect_to_tf_box(tf.expand_dims(example['y0'][0:1], 1))),
            collections=self.image_summaries_collections)

        with slim.arg_scope(self._arg_scope(is_training=self._run_opts['is_training'])):
            # Process initial image and label to get "template".
            with tf.variable_scope('template'):
                p0 = get_masks_from_rectangles(example['y0'], frmsz=self.frmsz)
                first_image_with_mask = concat([example['x0'], p0], axis=3)
                state['template'] = self._template_net(first_image_with_mask)

        # Ensure that there is no key collision.
        assert len(set(state.keys()).intersection(set(window_state.keys()))) == 0
        state.update(window_state)
        # Do not over-write elements of state used in init().
        state = {k: tf.identity(v) for k, v in state.items()}
        return state

    def step(self, example, prev_state):
        state = {}
        state['template'] = prev_state['template']

        with tf.name_scope('extract_window'):
            # Extract the window state by taking a subset of the state dictionary.
            prev_window_state = {k: prev_state[k] for k in self._window_state_keys}
            window = self._window_model.window(prev_window_state)
            # Use the window chosen by the previous frame.
            example = {
                'x': self._extract_window(example['x'], window),
                'y': geom.crop_rect(example['y'], window),
            }
            example = {k: tf.stop_gradient(v) for k, v in example.items()}

        example['x'] = self._whiten(example['x'])

        x = example['x']
        with slim.arg_scope(self._arg_scope(is_training=self._run_opts['is_training'])):
            # Process all images from all sequences with feature net.
            with tf.variable_scope('features'):
                feat = self._feat_net(x)
            # Search each image using result of template network.
            with tf.variable_scope('search'):
                similarity = self._search_net(feat, prev_state['template'])
            # if self.use_rnn:
            #     # Update abstract position likelihood of object.
            #     with tf.variable_scope('track'):
            #         init_state = self._initial_state_net(example['x0'], example['y0'])
            #         curr_state = init_state
            #         similarity = tf.unstack(similarity, axis=1)
            #         position_map = [None] * self.ntimesteps
            #         for t in range(self.ntimesteps):
            #             with tf.variable_scope('update', reuse=(t > 0)):
            #                 position_map[t], curr_state = self._update_net(similarity[t], curr_state)
            #         position_map = tf.stack(position_map, axis=1)
            # else:
            #     position_map = similarity
            position_map = similarity
            # Transform abstract position position_map into rectangle.
            with tf.variable_scope('output'):
                prediction = self._output_net(position_map, self._init_example['y0'], window)

            if self.use_heatmap:
                with tf.variable_scope('foreground'):
                    # Map abstract position_map to probability of foreground.
                    prediction['hmap'] = self._foreground_net(position_map)
                    prediction['hmap_full'] = tf.image.resize_images(prediction['hmap'],
                        size=[self.frmsz, self.frmsz],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        align_corners=True)
                    prediction['hmap_softmax'] = tf.nn.softmax(prediction['hmap_full'])

        # Visualize rectangle in window.
        if 'y' in prediction:
            tf.summary.image('frame_1_to_n',
                tf.image.draw_bounding_boxes(
                    _normalize_image_range(example['x'][0:1]),
                    geom.rect_to_tf_box(tf.expand_dims(prediction['y'][0:1], 1))),
                collections=self.image_summaries_collections)

        # TODO: How to update window state from score map?
        window_state = prev_window_state
        # # Update window state for next frame.
        # window_state = {}
        # with tf.name_scope('update_window'):
        #     # Obtain rectangle in image co-ordinates.
        #     prediction_uncrop = {
        #         'y': geom.crop_rect(prediction['y'], geom.crop_inverse(window)),
        #     }
        #     window_state = self._window_model.update(prediction_uncrop, prev_window_state)

        # if self.use_rnn:
        #     state = {k: (init_state[k], curr_state[k]) for k in curr_state}
        # else:
        #     state = {}

        state.update(window_state)
        return prediction, window, state

    def _arg_scope(self, is_training):
        batch_norm_opts = {} if not self.use_batch_norm else {
            'normalizer_fn': slim.batch_norm,
            'normalizer_params': {
                'is_training': is_training,
                'fused': True,
            },
        }
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            **batch_norm_opts) as arg_sc:
            return arg_sc

    def _extract_window(self, x, window):
        return geom.crop_image(x, window,
                               crop_size=[self.frmsz, self.frmsz],
                               pad_value=self.stat.get('mean', 0.0))

    def _whiten(self, x):
        return _whiten_image(x, mean=self.stat.get('mean', 0.0),
                                std=self.stat.get('std', 1.0))

    def _feat_net(self, x):
        assert len(x.shape) == 4
        with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='SAME'):
            # conv1
            x = slim.conv2d(x, 64, 11, stride=4)
            x = slim.max_pool2d(x)
            # conv2
            x = slim.conv2d(x, 128, 5)
            x = slim.max_pool2d(x)
            # conv3
            x = slim.conv2d(x, 192, 3)
            # conv4
            x = slim.conv2d(x, 192, 3)
            # conv5
            x = slim.conv2d(x, 128, 3, activation_fn=None)
        return x

    def _template_net(self, x):
        assert len(x.shape) == 4
        with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='SAME'):
            x = self._feat_net(x)
            x = tf.nn.relu(x) # No activation_fn at output of _feat_net.
            # x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            x = slim.conv2d(x, 128, 3)
            x = slim.max_pool2d(x)
            x = slim.conv2d(x, 128, 3)
            x = slim.max_pool2d(x)
            print 'template: shape at fc layer:', x.shape.as_list()
            x = slim.conv2d(x, 128, 4, padding='VALID', activation_fn=None)
            assert x.shape.as_list()[1:3] == [1, 1]
        return x

    def _search_net(self, x, f):
        assert len(x.shape) == 4
        assert len(f.shape) == 4
        dim = 256
        # x.shape is [b, hx, wx, c]
        # f.shape is [b, hf, wf, c] = [b, 1, 1, c]
        # Search for f in x.
        x = tf.nn.relu(x + f)
        x = slim.conv2d(x, dim, 1)
        x = slim.conv2d(x, dim, 1)
        return x

    # def _initial_state_net(self, x0, y0, state_dim=16):
    #     assert len(x0.shape) == 4
    #     f = get_masks_from_rectangles(y0, frmsz=self.frmsz)
    #     f = _feat_net(f)
    #     h = slim.conv2d(f, state_dim, 3, activation_fn=None)
    #     c = slim.conv2d(f, state_dim, 3, activation_fn=None)
    #     return {'h': h, 'c': c}

    # def _update_net(self, x, prev_state):
    #     assert len(x.shape) == 4
    #     '''Convert response maps to rectangle.'''
    #     h = prev_state['h']
    #     c = prev_state['c']
    #     h, c = self._conv_lstm(x, h, c, state_dim=16)
    #     x = h
    #     state = {'h': h, 'c': c}
    #     return x, state

    # def _conv_lstm(self, x, h_prev, c_prev, state_dim, name='clstm'):
    #     assert len(x.shape) == 4
    #     with tf.name_scope(name) as scope:
    #         with slim.arg_scope([slim.conv2d],
    #                             num_outputs=state_dim,
    #                             kernel_size=3,
    #                             padding='SAME',
    #                             activation_fn=None,
    #                             normalizer_fn=None):
    #             i = tf.nn.sigmoid(slim.conv2d(x, scope='xi') +
    #                               slim.conv2d(h_prev, scope='hi', biases_initializer=None))
    #             f = tf.nn.sigmoid(slim.conv2d(x, scope='xf') +
    #                               slim.conv2d(h_prev, scope='hf', biases_initializer=None))
    #             y = tf.nn.sigmoid(slim.conv2d(x, scope='xo') +
    #                               slim.conv2d(h_prev, scope='ho', biases_initializer=None))
    #             c_tilde = tf.nn.tanh(slim.conv2d(x, scope='xc') +
    #                                  slim.conv2d(h_prev, scope='hc', biases_initializer=None))
    #             c = (f * c_prev) + (i * c_tilde)
    #             h = y * tf.nn.tanh(c)
    #     return h, c

    def _foreground_net(self, x):
        assert len(x.shape) == 4
        # Map output of LSTM to a heatmap.
        x = slim.conv2d(x, 2, kernel_size=1, activation_fn=None, normalizer_fn=None)
        return x

    def _output_net(self, x, init_rect, window):
        assert len(x.shape) == 4
        output = {}
        if self.output_mode == 'rectangle':
            with tf.variable_scope('rectangle'):
                with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='SAME'):
                    # Map output of LSTM to a rectangle.
                    x = slim.conv2d(x, 64, 3)
                    x = slim.max_pool2d(x)
                    x = slim.conv2d(x, 128, 3)
                    x = slim.max_pool2d(x)
                    print 'output: shape at fc layer:', x.shape.as_list()
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 512)
                    x = slim.fully_connected(x, 512)
                    x = slim.fully_connected(x, 4, activation_fn=None, normalizer_fn=None)
            output = {'y': x}
        elif self.output_mode == 'score_map':
            with tf.variable_scope('score_map'):
                score = slim.conv2d(x, 1, 1, activation_fn=None, normalizer_fn=None)
                score = tf.squeeze(score, axis=-1)
                # Find arg max in score map.
                # Score map is [n, h, w, 1].
                score_dim = tf.shape(score)[1:3]
                score_vec, _ = merge_dims(score, 1, 3)
                argmax_vec = tf.argmax(score_vec, axis=1)
                # Note: Co-ordinates in (i, j) order.
                argmax_i, argmax_j = tf.py_func(np.unravel_index, [argmax_vec, score_dim], [tf.int64, tf.int64])
                argmax_i.set_shape([None])
                argmax_j.set_shape([None])
                argmax = tf.stack([argmax_i, argmax_j], -1)
                # New position is centered at arg max in window.
                # Assume centers of receptive fields align with first and last pixels.
                center = tf.to_float(argmax) / tf.to_float(score_dim - 1)
                center = center[:, ::-1]
                # Get size of initial rectangle in reference frame of search window.
                old_rect_min, old_rect_max = geom.rect_min_max(geom.crop_rect(init_rect, window))
                rect_size = old_rect_max - old_rect_min
                # Use a rectangle of the same dimension at the new position.
                # NOTE: If window uses size of rectangle to get next window, then this is unstable!!
                pred_rect = geom.make_rect(center - 0.5*rect_size, center + 0.5*rect_size)
            output = {'score': score, 'y': pred_rect}
        else:
            raise ValueError('unknown output mode: {}'.format(self.output_mode))
        return output


class ConditionalWindow:
    def __init__(self, train_model, test_model):
        self._models = {'train': train_model, 'test': test_model}
        self._keys = {k: None for k in self._models}
        self._is_training = None

    def init(self, example, is_training):
        self._is_training = is_training
        model_states = {k: model.init(example) for k, model in self._models.items()}
        self._keys = {k: model_states[k].keys() for k in self._models}
        state = self._merge(model_states)
        return state

    def update(self, prediction, prev_state):
        prev_model_states = self._split(prev_state)
        model_states = {
            k: model.update(prediction, prev_model_states[k])
            for k, model in self._models.items()
        }
        state = self._merge(model_states)
        return state

    def window(self, state):
        model_states = self._split(state)
        return tf.cond(self._is_training,
                       lambda: self._models['train'].window(model_states['train']),
                       lambda: self._models['test'].window(model_states['test']))

    def _merge(self, states):
        return dict(itertools.chain(*[
            [(model_key+'/'+var_key, var) for var_key, var in model_state.items()]
            for model_key, model_state in states.items()
        ]))

    def _split(self, states):
        return {
            model_key: {
                var_key: states[model_key+'/'+var_key]
                for var_key in self._keys[model_key]
            }
            for model_key in self._models
        }


def mlp(example, ntimesteps, frmsz,
        summaries_collections=None,
        hidden_dim=1024):
    z0 = tf.concat([slim.flatten(example['x0']), example['y0']], axis=1)
    z0 = slim.fully_connected(z0, hidden_dim)
    z = []
    x = tf.unstack(example['x'], axis=1)
    for t in range(ntimesteps):
        with tf.variable_scope('frame', reuse=(t > 0)):
            zt = slim.flatten(x[t])
            zt = slim.fully_connected(zt, hidden_dim)
            zt = zt + z0
            zt = slim.fully_connected(zt, 4, activation_fn=None)
            z.append(zt)
    z = tf.stack(z, axis=1)

    class Model:
        pass
    model = Model()
    model.outputs = {'y': z}
    model.state   = {}
    # Properties of instantiated model:
    model.image_size   = (frmsz, frmsz)
    model.sequence_len = ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model


'''
    state = model.init(example)

    window = model.window(state)

    state = model.update(prediction, prev_state)
        In the future, prediction may include presence/absence of object.

The final state will be fed to the initial state to process long sequences.
'''


class WholeImageWindow:
    def __init__(self, batchsz):
        self.batchsz = batchsz

    def init(self, example):
        state = {}
        return state

    def update(self, prediction, prev_state):
        state = {}
        return state

    def _window(self, state):
        return self._window(**state)

    def window(self, state):
        rect = [0.0, 0.0, 1.0, 1.0]
        # Use same window for every image in batch.
        return tf.tile(tf.expand_dims(rect, 0), [self.batchsz, 1])


class InitialWindow:
    def __init__(self, relative_size=4.0):
        self.relative_size = relative_size

    def init(self, example):
        init_obj_rect = example['y0']
        state = {'init_obj_rect': init_obj_rect}
        return state

    def update(self, prediction, prev_state):
        state = dict(prev_state)
        return state

    def window(self, state):
        return self._window(**state)

    def _window(self, init_obj_rect):
        return geom.object_centric_window(init_obj_rect,
            relative_size=self.relative_size)


class MovingAverageWindow:
    def __init__(self, decay, relative_size=4.0):
        self.decay = decay
        self.relative_size = relative_size
        self.eps = 0.01

    def init(self, example):
        center, log_diameter = self._center_log_diameter(example['y0'])
        state = {'center': center, 'log_diameter': log_diameter}
        return state

    def update(self, prediction, prev_state):
        center, log_diameter = self._center_log_diameter(prediction['y'])
        # Moving average.
        center = self.decay * prev_state['center'] + (1 - self.decay) * center
        log_diameter = self.decay * prev_state['log_diameter'] + (1 - self.decay) * log_diameter
        state = {'center': center, 'log_diameter': log_diameter}
        return state

    def window(self, state):
        return self._window(**state)

    def _window(self, center, log_diameter):
        window_diameter = self.relative_size * tf.exp(log_diameter)
        window_size = tf.expand_dims(window_diameter, -1)
        window_min = center - 0.5*window_size
        window_max = center + 0.5*window_size
        return geom.make_rect(window_min, window_max)

    def _center_log_diameter(self, rect):
        min_pt, max_pt = geom.rect_min_max(rect)
        center = 0.5 * (min_pt + max_pt)
        size = tf.maximum(0.0, max_pt - min_pt)
        log_diameter = tf.reduce_mean(tf.log(size + self.eps), axis=-1)
        return center, log_diameter


def _normalize_image_range(x):
    return 0.5 * (1 + x / tf.reduce_max(tf.abs(x)))


def _whiten_image(x, mean, std, name='whiten_image'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)


def load_model(o):
    '''
    example is a dictionary that maps strings to Tensors.
    Its keys should include 'inputs', 'labels', 'x0', 'y0'.
    '''
    # if o.model == 'RNN_dual':
    #     model = functools.partial(RNN_dual, o=o, **model_params)
    # elif o.model == 'RNN_conv_asymm':
    #     model = functools.partial(rnn_conv_asymm, o=o, **model_params)
    # elif o.model == 'RNN_multi_res':
    #     model = functools.partial(rnn_multi_res, o=o, **model_params)
    if o.model == 'simple_search':
        model = functools.partial(SimpleSearch,
                                  ntimesteps=o.ntimesteps,
                                  frmsz=o.frmsz,
                                  batchsz=o.batchsz,
                                  output_mode=o.output_mode,
                                  weight_decay=o.wd,
                                  **o.model_params)
    else:
        raise ValueError ('model not available')
    return model

if __name__ == '__main__':
    '''Test model 
    '''

    from opts import Opts
    o = Opts()

    o.mode = 'train'
    o.dataset = 'ILSVRC'
    o._set_dataset_params()

    o.batchsz = 4

    o.losses = ['l1'] # 'l1', 'iou', etc.

    o.model = 'RNN_new' # RNN_basic, RNN_a 

    # data setting (since the model requires stat, I need this to test)
    import data
    loader = data.load_data(o)

    if o.model == 'RNN_basic':
        o.pass_yinit = True
        m = RNN_basic(o)
    elif o.model == 'RNN_new':
        o.losses = ['ce', 'l2']
        m = RNN_new(o, loader.stat['train'])

    pdb.set_trace()

