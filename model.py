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

The function step() returns the prediction in the reference frame of the window!
This enables the loss to be computed in that reference frame if desired.
(Particularly important for the heatmap loss.)

It is then possible to process a long sequence by dividing it into chunks
of length k and feeding state_{k-1} to state_init.

A Model also has the following properties::

    model.batch_size    # Batch size of model instance, either None or an integer.
    model.image_size    # Tuple of image size.
'''

import pdb
import functools
import tensorflow as tf
from tensorflow.contrib import slim
import math
import numpy as np
import os

import cnnutil
import geom
from helpers import merge_dims
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


class RNN_dual(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 lstm1_nlayers=1,
                 lstm2_nlayers=1,
                 use_cnn3=False,
                 pass_hmap=False,
                 dropout_rnn=False,
                 dropout_cnn=False,
                 keep_prob=0.2, # following `Recurrent Neural Network Regularization, Zaremba et al.
                 init_memory=False,
                 ):
        # model parameters
        self.lstm1_nlayers = lstm1_nlayers
        self.lstm2_nlayers = lstm2_nlayers
        self.use_cnn3      = use_cnn3
        self.pass_hmap     = pass_hmap
        self.dropout_rnn   = dropout_rnn
        self.dropout_cnn   = dropout_cnn
        self.keep_prob     = keep_prob
        self.init_memory   = init_memory
        # Ignore sumaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.memory, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_init_lstm2(x, name='pass_init_lstm2'):
            ''' CNN for memory states in lstm2. Used to initialize.
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 2, [7, 7], stride=3, scope='conv1')
                    x = slim.conv2d(x, 2, [1, 1], stride=1, activation_fn=None, scope='conv2')
            return x

        def pass_cnn1(x, name):
            ''' CNN for search space
            '''
            out = []
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1'); out.append(x)
                    x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool1'); out.append(x)
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3'); out.append(x)
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv4'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool2'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv5'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv6'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv7'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool3'); out.append(x)
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv8'); out.append(x)
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv9'); out.append(x)
            return out

        def pass_cnn2(x, outsize=1024, name='pass_cnn2'):
            ''' CNN for appearance
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
                    x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
                    x = slim.max_pool2d(x, 2, scope='pool1')
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv4')
                    x = slim.max_pool2d(x, 2, scope='pool2')
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv5')
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv6')
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv7')
                    x = slim.max_pool2d(x, 2, scope='pool3')
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv8')
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv9')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1')
                    if self.dropout_cnn:
                        x = slim.dropout(x, keep_prob=self.keep_prob, is_training=is_training, scope='dropout1')
                    x = slim.fully_connected(x, outsize, scope='fc2')
            return x

        def pass_cnn3(x, name):
            ''' CNN for flow
            '''
            out = []
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1'); out.append(x)
                    x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool1'); out.append(x)
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3'); out.append(x)
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv4'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool2'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv5'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv6'); out.append(x)
                    x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv7'); out.append(x)
                    x = slim.max_pool2d(x, 2, scope='pool3'); out.append(x)
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv8'); out.append(x)
                    x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv9'); out.append(x)
            return out

        def pass_lstm1(x, h_prev, c_prev, name):
            with tf.name_scope(name):
                with slim.arg_scope([slim.fully_connected],
                        num_outputs=o.nunits,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    # NOTE: `An Empirical Exploration of Recurrent Neural Network Architecture`.
                    # Initialize forget bias to be 1.
                    # They also use `tanh` instead of `sigmoid` for input gate. (yet not employed here)
                    ft = slim.fully_connected(concat((h_prev, x), 1), biases_initializer=tf.ones_initializer(), scope='hf')
                    it = slim.fully_connected(concat((h_prev, x), 1), scope='hi')
                    ct_tilda = slim.fully_connected(concat((h_prev, x), 1), scope='hc')
                    ot = slim.fully_connected(concat((h_prev, x), 1), scope='ho')
                    ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
                    ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
            return ht, ct

        def pass_multi_level_cross_correlation(search, filt, name):
            ''' Multi-level cross-correlation function producing scoremaps.
            Option 1: depth-wise convolution
            Option 2: similarity score (-> doesn't work well)
            Note that depth-wise convolution with 1x1 filter is actually same as
            channel-wise (and element-wise) multiplication.
            '''
            # TODO: sigmoid or softmax over scoremap?
            # channel-wise l2 normalization as in Universal Correspondence Network?
            scoremap = []
            with tf.name_scope(name):
                with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    for i in range(len(search)):
                        depth = search[i].shape.as_list()[-1]
                        scoremap.append(search[i] *
                                tf.expand_dims(tf.expand_dims(slim.fully_connected(filt, depth), 1), 1))
            return scoremap

        def pass_multi_level_integration_correlation_and_flow(correlation, flow, name):
            ''' Multi-level integration of correlation and flow outputs.
            Using sum.
            '''
            with tf.name_scope(name):
                scoremap = [correlation[i]+flow[i] for i in range(len(correlation))]
            return scoremap

        def pass_multi_level_deconvolution(x, name):
            ''' Multi-level deconvolutions.
            This is in a way similar to HourglassNet.
            Using sum.
            '''
            deconv = x[-1]
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        kernel_size=[3,3],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    for i in range(len(x)-1):
                        shape_to = x[len(x)-2-i].shape.as_list()
                        deconv = slim.conv2d(
                                tf.image.resize_images(deconv, shape_to[1:3]),
                                num_outputs=shape_to[-1],
                                scope='deconv{}'.format(i+1))
                        deconv = deconv + x[len(x)-2-i] # TODO: try concat
                        deconv = slim.conv2d(deconv,
                                num_outputs=shape_to[-1],
                                kernel_size=[1,1],
                                scope='conv{}'.format(i+1)) # TODO: pass conv before addition
            return deconv

        def pass_lstm2(x, h_prev, c_prev, name):
            ''' ConvLSTM
            h and c have the same spatial dimension as x.
            '''
            # TODO: increase size of hidden
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        num_outputs=2,
                        kernel_size=3,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    it = tf.nn.sigmoid(slim.conv2d(x, scope='xi') + slim.conv2d(h_prev, scope='hi'))
                    ft = tf.nn.sigmoid(slim.conv2d(x, scope='xf') + slim.conv2d(h_prev, scope='hf'))
                    ct_tilda = tf.nn.tanh(slim.conv2d(x, scope='xc') + slim.conv2d(h_prev, scope='hc'))
                    ct = (ft * c_prev) + (it * ct_tilda)
                    ot = tf.nn.sigmoid(slim.conv2d(x, scope='xo') + slim.conv2d(h_prev, scope='ho'))
                    ht = ot * tf.nn.tanh(ct)
            return ht, ct

        def pass_out_rectangle(x, name):
            ''' Regress output rectangle.
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    if not self.lstm2_nlayers > 0:
                        x = slim.conv2d(x, 2, 1, scope='conv1')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1')
                    x = slim.fully_connected(x, 1024, scope='fc2')
                    x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        def pass_out_heatmap(x, name):
            ''' Upsample and generate spatial heatmap.
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        #num_outputs=x.shape.as_list()[-1],
                        num_outputs=2, # NOTE: hmap before lstm2 -> reduce the output channel to 2 here.
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(tf.image.resize_images(x, [241, 241]),
                                    kernel_size=[3, 3], scope='deconv')
                    x = slim.conv2d(x, kernel_size=[1, 1], scope='conv1')
                    x = slim.conv2d(x, kernel_size=[1, 1], activation_fn=None, scope='conv2')
            return x


        x           = inputs['x']  # shape [b, ntimesteps, h, w, 3]
        x0          = inputs['x0'] # shape [b, h, w, 3]
        y0          = inputs['y0'] # shape [b, 4]
        y           = inputs['y']  # shape [b, ntimesteps, 4]
        use_gt      = inputs['use_gt']
        gt_ratio    = inputs['gt_ratio']
        is_training = inputs['is_training']

        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0)
        hmap_init = tf.identity(get_masks_from_rectangles(y0, o, kind='bg'))

        # lstm initial memory states. {random or CNN}.
        h1_init = [None] * self.lstm1_nlayers
        c1_init = [None] * self.lstm1_nlayers
        h2_init = [None] * self.lstm2_nlayers
        c2_init = [None] * self.lstm2_nlayers
        if not self.init_memory:
            with tf.name_scope('lstm_initial'):
                with slim.arg_scope([slim.model_variable],
                        initializer=tf.truncated_normal_initializer(stddev=0.01),
                        regularizer=slim.l2_regularizer(o.wd)):
                    for i in range(self.lstm1_nlayers):
                        h1_init_single = slim.model_variable('lstm1_h_init_{}'.format(i+1), shape=[o.nunits])
                        c1_init_single = slim.model_variable('lstm1_c_init_{}'.format(i+1), shape=[o.nunits])
                        h1_init[i] = tf.stack([h1_init_single] * o.batchsz)
                        c1_init[i] = tf.stack([c1_init_single] * o.batchsz)
                    for i in range(self.lstm2_nlayers):
                        h2_init_single = slim.model_variable('lstm2_h_init_{}'.format(i+1), shape=[81, 81, 2]) # TODO: adaptive
                        c2_init_single = slim.model_variable('lstm2_c_init_{}'.format(i+1), shape=[81, 81, 2])
                        h2_init[i] = tf.stack([h2_init_single] * o.batchsz)
                        c2_init[i] = tf.stack([c2_init_single] * o.batchsz)
        else:
            with tf.name_scope('lstm_initial'):
                # lstm1
                hmap_from_rec = get_masks_from_rectangles(y_init, o)
                if self.pass_hmap:
                    xy = concat([x_init, hmap_from_rec, hmap_init], axis=3)
                    xy = tf.stop_gradient(xy)
                else:
                    xy = concat([x_init, hmap_from_rec], axis=3)
                for i in range(self.lstm1_nlayers):
                    with tf.variable_scope('lstm1_layer_{}'.format(i+1)):
                        with tf.variable_scope('h_init'):
                            h1_init[i] = pass_cnn2(xy, o.nunits)
                        with tf.variable_scope('c_init'):
                            c1_init[i] = pass_cnn2(xy, o.nunits)
                # lstm2
                for i in range(self.lstm2_nlayers):
                    with tf.variable_scope('lstm2_layer_{}'.format(i+1)):
                        with tf.variable_scope('h_init'):
                            h2_init[i] = pass_init_lstm2(hmap_init)
                        with tf.variable_scope('c_init'):
                            c2_init[i] = pass_init_lstm2(hmap_init)

        with tf.name_scope('noise'):
            noise = tf.truncated_normal(tf.shape(y), mean=0.0, stddev=0.05,
                                        dtype=o.dtype, seed=o.seed_global, name='noise')


        x_prev = x_init
        y_prev = y_init
        hmap_prev = hmap_init
        h1_prev, c1_prev = h1_init, c1_init
        h2_prev, c2_prev = h2_init, c2_init

        y_pred = []
        hmap_pred = []
        memory_h2 = []
        memory_c2 = []

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]
            with tf.name_scope('cnn1_{}'.format(t)) as scope:
                with tf.variable_scope('cnn1', reuse=(t > 0)):
                    cnn1out = pass_cnn1(x_curr, scope)

            with tf.name_scope('cnn2_{}'.format(t)) as scope:
                with tf.variable_scope('cnn2', reuse=(t > 0)):
                    # use both `hmap_prev` along with `y_prev_{GT or pred}`
                    hmap_from_rec = get_masks_from_rectangles(y_prev, o)
                    if self.pass_hmap:
                        xy = concat([x_prev, hmap_from_rec, hmap_prev], axis=3) # TODO: backpropagation-able?
                        xy = tf.stop_gradient(xy)
                    else:
                        xy = concat([x_prev, hmap_from_rec], axis=3)
                    cnn2out = pass_cnn2(xy, name=scope)

            if self.use_cnn3:
                with tf.name_scope('cnn3_{}'.format(t)) as scope:
                    with tf.variable_scope('cnn3', reuse=(t > 0)):
                        cnn3out = pass_cnn3(tf.concat([x_prev, x_curr], axis=3), scope)

            h1_curr = [None] * self.lstm1_nlayers
            c1_curr = [None] * self.lstm1_nlayers
            with tf.name_scope('lstm1_{}'.format(t)) as scope:
                with tf.variable_scope('lstm1', reuse=(t > 0)):
                    input_to_lstm1 = tf.identity(cnn2out)
                    for i in range(self.lstm1_nlayers):
                        with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                            h1_curr[i], c1_curr[i] = pass_lstm1(input_to_lstm1, h1_prev[i], c1_prev[i], scope)
                        if self.dropout_rnn:
                            input_to_lstm1 = slim.dropout(h1_curr[i],
                                                          keep_prob=self.keep_prob,
                                                          is_training=is_training, scope='dropout')
                        else:
                            input_to_lstm1 = h1_curr[i]


            with tf.name_scope('multi_level_cross_correlation_{}'.format(t)) as scope:
                with tf.variable_scope('multi_level_cross_correlation', reuse=(t > 0)):
                    scoremap = pass_multi_level_cross_correlation(cnn1out, h1_curr[-1], scope) # multi-layer lstm1

            if self.use_cnn3:
                with tf.name_scope('multi_level_integration_correlation_and_flow_{}'.format(t)) as scope:
                    with tf.variable_scope('multi_level_integration_correlation_and_flow', reuse=(t > 0)):
                        scoremap = pass_multi_level_integration_correlation_and_flow(
                                scoremap, cnn3out, scope)

            with tf.name_scope('multi_level_deconvolution_{}'.format(t)) as scope:
                with tf.variable_scope('multi_level_deconvolution', reuse=(t > 0)):
                    scoremap = pass_multi_level_deconvolution(scoremap, scope)

            with tf.name_scope('cnn_out_hmap_{}'.format(t)) as scope:
                with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                    hmap_curr = pass_out_heatmap(scoremap, scope)

            h2_curr = [None] * self.lstm2_nlayers
            c2_curr = [None] * self.lstm2_nlayers
            with tf.name_scope('lstm2_{}'.format(t)) as scope:
                with tf.variable_scope('lstm2', reuse=(t > 0)):
                    input_to_lstm2 = tf.identity(scoremap)
                    for i in range(self.lstm2_nlayers):
                        with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                            h2_curr[i], c2_curr[i] = pass_lstm2(input_to_lstm2, h2_prev[i], c2_prev[i], scope)
                        if self.dropout_rnn:
                            input_to_lstm2 = slim.dropout(h2_curr[i],
                                                          keep_prob=self.keep_prob,
                                                          is_training=is_training, scope='dropout')
                        else:
                            input_to_lstm2 = h2_curr[i]

            with tf.name_scope('cnn_out_rec_{}'.format(t)) as scope:
                with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                    if self.lstm2_nlayers > 0:
                        y_curr_pred = pass_out_rectangle(h2_curr[-1], scope) # multi-layer lstm2
                    else:
                        y_curr_pred = pass_out_rectangle(scoremap, scope) # No LSTM2

            #with tf.name_scope('cnn_out_hmap_{}'.format(t)) as scope:
            #    with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
            #        hmap_curr = pass_out_heatmap(h2_curr[-1], scope) # multi-layer lstm2

            x_prev = x_curr
            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            y_prev = tf.cond(gt_condition, lambda: y_curr + noise[:,t], # TODO: should noise be gone?
                                           lambda: y_curr_pred)
            h1_prev, c1_prev = h1_curr, c1_curr
            h2_prev, c2_prev = h2_curr, c2_curr
            hmap_prev = hmap_curr

            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr)
            memory_h2.append(h2_curr[-1] if self.lstm2_nlayers > 0 else None)
            memory_c2.append(c2_curr[-1] if self.lstm2_nlayers > 0 else None)

        y_pred = tf.stack(y_pred, axis=1) # list to tensor
        hmap_pred = tf.stack(hmap_pred, axis=1)
        if self.lstm2_nlayers > 0:
            memory_h2 = tf.stack(memory_h2, axis=1)
            memory_c2 = tf.stack(memory_c2, axis=1)

        outputs = {'y': y_pred, 'hmap': hmap_pred}
        state = {}
        state.update({'h1_{}'.format(i+1): (h1_init[i], h1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'c1_{}'.format(i+1): (c1_init[i], c1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'h2_{}'.format(i+1): (h2_init[i], h2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'c2_{}'.format(i+1): (c2_init[i], c2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'x': (x_init, x_prev), 'y': (y_init, y_prev)})
        state.update({'hmap': (hmap_init, hmap_prev)})
        memory = {'h2': memory_h2, 'c2': memory_c2}

        #dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        dbg = {}
        return outputs, state, memory, dbg


def rnn_conv_asymm(example, o,
                   summaries_collections=None,
                   # Model parameters:
                   input_num_layers=3,
                   input_kernel_size=[7, 5, 3],
                   input_num_channels=[16, 32, 64],
                   input_stride=[2, 1, 1],
                   input_pool=[True, True, True],
                   input_pool_stride=[2, 2, 2],
                   input_pool_kernel_size=[3, 3, 3],
                   input_batch_norm=False,
                   lstm_num_channels=64,
                   lstm_num_layers=1):
                   # lstm_kernel_size=[3]):

    images = example['x']
    x0     = example['x0']
    y0     = example['y0']
    is_training = example['is_training']
    masks = get_masks_from_rectangles(y0, o)
    if o.debugmode:
        with tf.name_scope('input_preview'):
            tf.summary.image('x', images[0], collections=summaries_collections)
            target = concat([images[0, 0], masks[0]], axis=2)
            tf.summary.image('target', tf.expand_dims(target, axis=0),
                             collections=summaries_collections)
    if o.activ_histogram:
        with tf.name_scope('input_histogram'):
            tf.summary.histogram('x', images, collections=summaries_collections)
    init_input = concat([x0, masks], axis=3)

    assert(len(input_kernel_size)      == input_num_layers)
    assert(len(input_num_channels)     == input_num_layers)
    assert(len(input_stride)           == input_num_layers)
    assert(len(input_pool)             == input_num_layers)
    assert(len(input_pool_stride)      == input_num_layers)
    assert(len(input_pool_kernel_size) == input_num_layers)
    assert(lstm_num_layers >= 1)
    # assert(len(lstm_kernel_size) == lstm_num_layers)

    def input_cnn(x, num_outputs, name='input_cnn'):
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                normalizer_fn = None
                conv2d_params = {'weights_regularizer': slim.l2_regularizer(o.wd)}
                if input_batch_norm:
                    conv2d_params.update({
                        'normalizer_fn': slim.batch_norm,
                        'normalizer_params': {
                            'is_training': is_training,
                        }})
                with slim.arg_scope([slim.conv2d], **conv2d_params):
                    layers = {}
                    for i in range(input_num_layers):
                        conv_name = 'conv{}'.format(i+1)
                        x = slim.conv2d(x, input_num_channels[i],
                                           kernel_size=input_kernel_size[i],
                                           stride=input_stride[i],
                                           scope=conv_name)
                        layers[conv_name] = x
                        if input_pool[i]:
                            pool_name = 'pool{}'.format(i+1)
                            x = slim.max_pool2d(x, kernel_size=input_pool_kernel_size[i],
                                                   stride=input_pool_stride[i],
                                                   scope=pool_name)
                    # if o.activ_histogram:
                    #     with tf.name_scope('summary'):
                    #         for k, v in layers.iteritems():
                    #             tf.summary.histogram(k, v, collections=summaries_collections)
        return x

    def conv_lstm(x, h_prev, c_prev, state_dim, name='conv_lstm'):
        with tf.name_scope(name) as scope:
            with slim.arg_scope([slim.conv2d],
                                num_outputs=state_dim,
                                kernel_size=3,
                                padding='SAME',
                                activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(o.wd)):
                i = tf.nn.sigmoid(slim.conv2d(x, scope='xi') +
                                  slim.conv2d(h_prev, scope='hi', biases_initializer=None))
                f = tf.nn.sigmoid(slim.conv2d(x, scope='xf') +
                                  slim.conv2d(h_prev, scope='hf', biases_initializer=None))
                y = tf.nn.sigmoid(slim.conv2d(x, scope='xo') +
                                  slim.conv2d(h_prev, scope='ho', biases_initializer=None))
                c_tilde = tf.nn.tanh(slim.conv2d(x, scope='xc') +
                                     slim.conv2d(h_prev, scope='hc', biases_initializer=None))
                c = (f * c_prev) + (i * c_tilde)
                h = y * tf.nn.tanh(c)
                # layers = {'i': i, 'f': f, 'o': y, 'c_tilde': c, 'c': c, 'h': h}
                # if o.activ_histogram:
                #     with tf.name_scope('summary'):
                #         for k, v in layers.iteritems():
                #             tf.summary.histogram(k, v, collections=summaries_collections)
        return h, c

    def output_cnn(x, name='output_cnn'):
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_regularizer=slim.l2_regularizer(o.wd)):
                    layers = {}
                    x = slim.conv2d(x, 128, kernel_size=3, stride=2, scope='conv1')
                    layers['conv1'] = x
                    x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool1')
                    x = slim.conv2d(x, 256, kernel_size=3, stride=1, scope='conv2')
                    layers['conv2'] = x
                    x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool2')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 4, scope='predict')
                    layers['predict'] = x
                    # if o.activ_histogram:
                    #     with tf.name_scope('summary'):
                    #         for k, v in layers.iteritems():
                    #             tf.summary.histogram(k, v, collections=summaries_collections)
        return x

    # At start of sequence, compute hidden state from first example.
    # Feed (h_init, c_init) to resume tracking from previous state.
    # Do NOT feed (h_init, c_init) when starting new sequence.
    # TODO: Share some layers?
    h_init = [None] * lstm_num_layers
    c_init = [None] * lstm_num_layers
    with tf.variable_scope('lstm_init'):
        for j in range(lstm_num_layers):
            with tf.variable_scope('layer_{}'.format(j+1)):
                with tf.variable_scope('h_init'):
                    h_init[j] = input_cnn(init_input, num_outputs=lstm_num_channels)
                with tf.variable_scope('c_init'):
                    c_init[j] = input_cnn(init_input, num_outputs=lstm_num_channels)

    # # TODO: Process all frames together in training (when sequences are equal length)
    # # (because it enables batch-norm to operate on whole sequence)
    # # but not during testing (when sequences are different lengths)
    # x, unmerge = merge_dims(images, 0, 2)
    # with tf.name_scope('frame_cnn') as scope:
    #     with tf.variable_scope('frame_cnn'):
    #         # Pass name scope from above, otherwise makes new name scope
    #         # within name scope created by variable scope.
    #         r = input_cnn(x, num_outputs=lstm_num_channels, name=scope)
    # r = unmerge(r, 0)

    y = []
    ht, ct = h_init, c_init
    for t in range(o.ntimesteps):
        xt = images[:, t]
        with tf.name_scope('frame_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('frame_cnn', reuse=(t > 0)):
                # Pass name scope from above, otherwise makes new name scope
                # within name scope created by variable scope.
                xt = input_cnn(xt, num_outputs=lstm_num_channels, name=scope)
        with tf.name_scope('conv_lstm_{}'.format(t)):
            with tf.variable_scope('conv_lstm', reuse=(t > 0)):
                for j in range(lstm_num_layers):
                    layer_name = 'layer_{}'.format(j+1)
                    with tf.variable_scope(layer_name, reuse=(t > 0)):
                        ht[j], ct[j] = conv_lstm(xt, ht[j], ct[j],
                                                 state_dim=lstm_num_channels)
                    xt = ht[j]
        with tf.name_scope('out_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('out_cnn', reuse=(t > 0)):
                yt = output_cnn(xt, name=scope)
        y.append(yt)
        # tf.get_variable_scope().reuse_variables()
    h_last, c_last = ht, ct
    y = tf.stack(y, axis=1) # list to tensor

    # with tf.name_scope('out_cnn') as scope:
    #     with tf.variable_scope('out_cnn', reuse=(t > 0)):
    #         y = output_cnn(z, name=scope)

    with tf.name_scope('summary'):
        if o.activ_histogram:
            tf.summary.histogram('rect', y, collections=summaries_collections)
        if o.param_histogram:
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(v.name, v, collections=summaries_collections)

    outputs = {'y': y}
    state = {}
    state.update({'h_{}'.format(j+1): (h_init[j], h_last[j])
                  for j in range(lstm_num_layers)})
    state.update({'c_{}'.format(j+1): (c_init[j], c_last[j])
                  for j in range(lstm_num_layers)})

    class Model:
        pass
    model = Model()
    model.outputs = outputs
    model.state   = state
    # Properties of instantiated model:
    model.image_size   = (o.frmsz, o.frmsz)
    model.sequence_len = o.ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model


def rnn_multi_res(example, ntimesteps, frmsz, weight_decay=0.0, heatmap_stride=1,
                  summaries_collections=None,
                  # Model options:
                  kind='vgg',
                  use_heatmap=False,
                  **model_params):

    images = example['x']
    x0     = example['x0']
    y0     = example['y0']
    is_training = example['is_training']
    masks = get_masks_from_rectangles(y0, frmsz=frmsz)
    # if o.debugmode:
    #     with tf.name_scope('input_preview'):
    #         tf.summary.image('x', images[0], collections=summaries_collections)
    #         target = concat([images[0, 0], masks[0]], axis=2)
    #         tf.summary.image('target', tf.expand_dims(target, axis=0),
    #                          collections=summaries_collections)
    # if o.activ_histogram:
    #     with tf.name_scope('input_histogram'):
    #         tf.summary.histogram('x', images, collections=summaries_collections)
    init_input = concat([x0, masks], axis=3)

    # net_fn(x, None, init=True, ...) returns None, h_init.
    # net_fn(x, h_prev, init=False, ...) returns y, h.
    if kind == 'vgg':
        net_fn = multi_res_vgg
    elif kind == 'resnet':
        raise Exception('not implemented')
    else:
        raise ValueError('unknown net type: {}'.format(kind))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            # TODO: Share some layers?
            with tf.variable_scope('rnn_init'):
                _, s_init = net_fn(init_input, None, init=True,
                    use_heatmap=use_heatmap, heatmap_stride=heatmap_stride,
                    **model_params)

            y, heatmap = [], []
            s_prev = s_init
            for t in range(ntimesteps):
                with tf.name_scope('t{}'.format(t)):
                    with tf.variable_scope('frame', reuse=(t > 0)):
                        outputs_t, s = net_fn(images[:, t], s_prev, init=False,
                            use_heatmap=use_heatmap, heatmap_stride=heatmap_stride,
                            **model_params)
                        s_prev = s
                        y.append(outputs_t['y'])
                        if use_heatmap:
                            heatmap.append(outputs_t['hmap'])
            y = tf.stack(y, axis=1) # list to tensor
            if use_heatmap:
                heatmap = tf.stack(heatmap, axis=1) # list to tensor

    outputs = {'y': y}
    if use_heatmap:
        outputs['hmap'] = heatmap
    assert(set(s_init.keys()) == set(s.keys()))
    state = {k: (s_init[k], s[k]) for k in s}

    class Model:
        pass
    model = Model()
    model.outputs = outputs
    model.state   = state
    # Properties of instantiated model:
    model.image_size   = (frmsz, frmsz)
    model.sequence_len = ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model


def conv_lstm(x, h_prev, c_prev, state_dim, name='clstm'):
    with tf.name_scope(name) as scope:
        with slim.arg_scope([slim.conv2d],
                            num_outputs=state_dim,
                            kernel_size=3,
                            padding='SAME',
                            activation_fn=None):
            i = tf.nn.sigmoid(slim.conv2d(x, scope='xi') +
                              slim.conv2d(h_prev, scope='hi', biases_initializer=None))
            f = tf.nn.sigmoid(slim.conv2d(x, scope='xf') +
                              slim.conv2d(h_prev, scope='hf', biases_initializer=None))
            y = tf.nn.sigmoid(slim.conv2d(x, scope='xo') +
                              slim.conv2d(h_prev, scope='ho', biases_initializer=None))
            c_tilde = tf.nn.tanh(slim.conv2d(x, scope='xc') +
                                 slim.conv2d(h_prev, scope='hc', biases_initializer=None))
            c = (f * c_prev) + (i * c_tilde)
            h = y * tf.nn.tanh(c)
            # layers = {'i': i, 'f': f, 'o': y, 'c_tilde': c, 'c': c, 'h': h}
            # if o.activ_histogram:
            #     with tf.name_scope('summary'):
            #         for k, v in layers.iteritems():
            #             tf.summary.histogram(k, v, collections=summaries_collections)
    return h, c


def multi_res_vgg(x, prev, init, use_heatmap, heatmap_stride,
        # Model parameters:
        use_bnorm=False,
        conv_num_groups=5,
        conv_num_layers=[2, 2, 3, 3, 3],
        conv_kernel_size=[3, 3, 3, 3, 3],
        conv_stride=[1, 1, 1, 1, 1],
        conv_rnn_depth=[0, 0, 1, 0, 0],
        conv_dim_first=16,
        conv_dim_last=256,
        fc_num_layers=0,
        fc_dim=256,
        heatmap_input_stride=16,
        ):
    '''
    Layers are conv[1], ..., conv[N], fc[N+1], ..., fc[N+M].
    '''

    assert(len(conv_num_layers) == conv_num_groups)
    assert(len(conv_rnn_depth) == conv_num_groups)
    conv_use_rnn = map(lambda x: x > 0, conv_rnn_depth)

    dims = np.logspace(math.log10(conv_dim_first),
                       math.log10(conv_dim_last),
                       conv_num_groups)
    dims = np.round(dims).astype(np.int)

    with slim.arg_scope([slim.batch_norm], fused=True):
        # Dictionary of state.
        curr = {}
        # Array of (tensor before op with stride, stride) tuples.
        stride_steps = []
        if init and not any(conv_use_rnn):
            return None, curr
        bnorm_params = {'normalizer_fn': slim.batch_norm} if use_bnorm else {}
        for j in range(conv_num_groups):
            # Group of conv plus a pool.
            with slim.arg_scope([slim.conv2d], **bnorm_params):
                conv_name = lambda k: 'conv{}_{}'.format(j+1, k+1)
                for k in range(conv_num_layers[j]):
                    stride = conv_stride[j] if k == conv_num_layers[j]-1 else 1
                    if stride != 1:
                        stride_steps.append((x, stride))
                    x = slim.conv2d(x, dims[j], conv_kernel_size[j], stride=stride,
                                    scope=conv_name(k))
                stride_steps.append((x, 2))
                x = slim.max_pool2d(x, 3, padding='SAME', scope='pool{}'.format(j+1))

            # LSTM at end of group.
            if conv_use_rnn[j]:
                rnn_name = 'rnn{}'.format(j+1)
                if init:
                    # Produce initial state of RNNs.
                    # TODO: Why not variable_scope here? Only expect one instance?
                    for d in range(conv_rnn_depth[j]):
                        layer_name = 'layer{}'.format(d+1)
                        h = '{}_{}_{}'.format(rnn_name, layer_name, 'h')
                        c = '{}_{}_{}'.format(rnn_name, layer_name, 'c')
                        curr[h] = slim.conv2d(x, dims[j], 3, activation_fn=None, scope=h)
                        curr[c] = slim.conv2d(x, dims[j], 3, activation_fn=None, scope=c)
                else:
                    # Different scope for different RNNs.
                    with tf.variable_scope(rnn_name):
                        for d in range(conv_rnn_depth[j]):
                            layer_name = 'layer{}'.format(d+1)
                            with tf.variable_scope(layer_name):
                                h = '{}_{}_{}'.format(rnn_name, layer_name, 'h')
                                c = '{}_{}_{}'.format(rnn_name, layer_name, 'c')
                                curr[h], curr[c] = conv_lstm(x, prev[h], prev[c], state_dim=dims[j])
                                x = curr[h]
            if init and not any(conv_use_rnn[j+1:]):
                # Do not add layers to init network that will not be used.
                return None, curr

        y = x

        outputs = {}
        if use_heatmap:
            # Upsample and convolve to get to heatmap.
            total_stride = np.cumprod([stride for _, stride in stride_steps])
            input_ind = np.asscalar(np.flatnonzero(np.array(total_stride) == heatmap_input_stride))
            output_ind = np.asscalar(np.flatnonzero(np.array(total_stride) == heatmap_stride))
            # TODO: Handle case of using final stride step.
            heatmap, _ = stride_steps[input_ind+1]
            # Now total_stride[output_ind] = (___, heatmap_stride).
            # Our current stride is total_stride[-1].
            # Work back from current stride to restore resolution.
            # Note that we don't take the features from stride_steps[output_ind] because
            # these are the features before the desired stride.
            # TODO: Make sure this works with heatmap_stride = 1.
            for j in range(input_ind, output_ind, -1):
                # Combine current features with features before stride.
                before_stride, stride = stride_steps[j]
                heatmap = upsample(heatmap, stride)
                heatmap = concat([heatmap, before_stride], axis=3)
                num_outputs = 2 if j == output_ind+1 else before_stride.shape.as_list()[3]
                activation_fn = None if j == output_ind+1 else tf.nn.relu
                # TODO: Should the kernel_size be 1, 3, or something else?
                heatmap = slim.conv2d(heatmap, num_outputs, 1, activation_fn=activation_fn,
                                      scope='merge{}'.format(j+1))
            outputs['hmap'] = heatmap

        # Fully-connected stage to get rectangle.
        y = slim.flatten(y)
        fc_name_fn = lambda ind: 'fc{}'.format(conv_num_groups + ind + 1)
        with slim.arg_scope([slim.fully_connected], **bnorm_params):
            for j in range(fc_num_layers):
                y = slim.fully_connected(y, fc_dim, scope=fc_name_fn(j))
        # Make prediction.
        y = slim.fully_connected(y, 4, activation_fn=None, normalizer_fn=None,
                                 scope=fc_name_fn(fc_num_layers))
        outputs['y'] = y
        return outputs, curr


class SimpleSearch:

    # TODO: Make stat part of the model (i.e. a variable?)
    def __init__(self, ntimesteps, frmsz, batchsz, stat, weight_decay=0.0,
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
        self.stat                        = stat
        self.weight_decay                = weight_decay
        self.summaries_collections       = summaries_collections
        self.image_summaries_collections = image_summaries_collections
        # Model parameters:
        # self.use_rnn              = use_rnn
        self.use_heatmap          = use_heatmap
        self.use_batch_norm       = use_batch_norm
        self.object_centric       = object_centric
        # self.normalize_size       = normalize_size
        # self.normalize_first_only = normalize_first_only

        # Public model properties:
        # self.image_size   = (self.frmsz, self.frmsz)
        # self.sequence_len = self.ntimesteps # Static length of unrolled RNN.
        self.batch_size   = None # Model accepts variable batch size.

        # Model state.
        self._template = None
        self._run_opts = None

        if self.object_centric:
            # self._window_model = MovingAverageWindow(0.5)
            self._window_model = InitialWindow()
        else:
            # TODO: May be more efficient to avoid cropping if using whole window?
            self._window_model = WholeImageWindow(batchsz=batchsz)
        self._window_state_keys = None

    def init(self, example, run_opts):
        state = {}
        self._run_opts = run_opts

        with tf.name_scope('extract_window'):
            # Get window for next frame.
            window_state = self._window_model.init(example)
            self._window_state_keys = window_state.keys()
            # For this model, use window of the next frame in the initial frame.
            window = self._window_model.window(window_state)
            # Crop the object in the first frame.
            example = {
                'x0': geom.crop_image(example['x0'],
                                      window,
                                      crop_size=[self.frmsz, self.frmsz],
                                      pad_value=self.stat['mean']),
                'y0': geom.crop_rect(example['y0'], window),
            }
            example = {k: tf.stop_gradient(v) for k, v in example.items()}

        example['x0'] = _whiten_image(example['x0'], self.stat['mean'], self.stat['std'])

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
                self._template = self._template_net(first_image_with_mask)

        # Ensure that there is no key collision.
        assert len(set(state.keys()).intersection(set(window_state.keys()))) == 0
        state.update(window_state)
        return state

    def step(self, example, prev_state):
        state = {}

        with tf.name_scope('extract_window'):
            # Extract the window state by taking a subset of the state dictionary.
            prev_window_state = {k: prev_state[k] for k in self._window_state_keys}
            window = self._window_model.window(prev_window_state)
            # Use the window chosen by the previous frame.
            example = {
                'x': geom.crop_image(example['x'],
                                     window,
                                     crop_size=[self.frmsz, self.frmsz],
                                     pad_value=None),
                'y': geom.crop_rect(example['y'], window),
            }
            example = {k: tf.stop_gradient(v) for k, v in example.items()}

        example['x'] = _whiten_image(example['x'], self.stat['mean'], self.stat['std'])

        x = example['x']
        with slim.arg_scope(self._arg_scope(is_training=self._run_opts['is_training'])):
            # Process all images from all sequences with feature net.
            with tf.variable_scope('features'):
                feat = self._feat_net(x)
            # Search each image using result of template network.
            with tf.variable_scope('search'):
                similarity = self._search_net(feat, self._template)
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
                position = self._output_net(position_map)

            prediction = {'y': position}
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
        tf.summary.image('frame_1_to_n',
            tf.image.draw_bounding_boxes(
                _normalize_image_range(example['x'][0:1]),
                geom.rect_to_tf_box(tf.expand_dims(prediction['y'][0:1], 1))),
            collections=self.image_summaries_collections)

        # Update window state for next frame.
        window_state = {}
        with tf.name_scope('update_window'):
            # Obtain rectangle in image co-ordinates.
            prediction_uncrop = {
                'y': geom.crop_rect(prediction['y'], geom.crop_inverse(window)),
            }
            window_state = self._window_model.update(prediction_uncrop, prev_window_state)

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

    def _output_net(self, x):
        assert len(x.shape) == 4
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
        return x


def _normalize_image_range(x):
    return 0.5 * (1 + x / tf.reduce_max(tf.abs(x)))


def _whiten_image(x, mean, std, name='whiten_image'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)


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

