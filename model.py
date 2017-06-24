'''This file describes several different models.

A model is a class with the properties::

    model.outputs      # Dictionary of tensors
    model.state        # Dictionary of 2-tuples of tensors
    model.batch_size   # Batch size of instance.
    model.sequence_len # Size of instantiated RNN.

The model constructor should take a dictionary of tensors::

    'x'  # Tensor of images [b, t, h, w, c]
    'x0' # Tensor of initial images [b, h, w, c]
    'y0' # Tensor of initial rectangles [b, 4]

It may also have 'target' if required.
Images input to the model are already normalized (e.g. have dataset mean subtracted).

The `outputs` dictionary should have a key 'y' and may have other keys such as 'hmap'.
'''

import pdb
import functools
import tensorflow as tf
from tensorflow.contrib import slim
import math
import numpy as np
import os

import cnnutil
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


def simple_search(original_example, ntimesteps, frmsz, batchsz, weight_decay=0.0,
        summaries_collections=None,
        image_summaries_collections=None,
        # Model parameters:
        use_rnn=True,
        use_heatmap=False,
        use_batch_norm=False, # Caution when use_rnn is True.
        normalize_size=False,
        normalize_first_only=False,
        ):

    def feat_net(x):
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

    def template_net(x):
        with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='SAME'):
            x = feat_net(x)
            x = tf.nn.relu(x) # No activation_fn at output of feat_net.
            # x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
            x = slim.conv2d(x, 128, 3)
            x = slim.max_pool2d(x)
            x = slim.conv2d(x, 128, 3)
            x = slim.max_pool2d(x)
            print 'template: shape at fc layer:', x.shape.as_list()
            x = slim.conv2d(x, 128, 4, padding='VALID', activation_fn=None)
            assert x.shape.as_list()[1:3] == [1, 1]
        return x

    def search_all(x, f):
        dim = 256
        # # x.shape is [b, t, hx, wx, c]
        # # f.shape is [b, hf, wf, c] = [b, 1, 1, c]
        # # relu(linear(concat(a, b)) = relu(linear(a) + linear(b))
        # f = slim.conv2d(f, dim, 1, activation_fn=None)
        # f = tf.expand_dims(f, 1)
        # x, unmerge = merge_dims(x, 0, 2)
        # x = slim.conv2d(x, dim, 1, activation_fn=None)
        # x = unmerge(x, 0)
        # # Search for f in x.

        # x = tf.nn.relu(tf.add(x, f))
        # x.shape is [b, t, hx, wx, c]
        # f.shape is [b, hf, wf, c] = [b, 1, 1, c]
        # relu(linear(concat(a, b)) = relu(linear(a) + linear(b))
        f = tf.expand_dims(f, 1)
        # Search for f in x.
        x = tf.nn.relu(x + f)

        # Post-process appearance "similarity".
        x, unmerge = merge_dims(x, 0, 2)
        x = slim.conv2d(x, dim, 1)
        x = slim.conv2d(x, dim, 1)
        x = unmerge(x, 0)
        return x

    def initial_state_net(x0, y0, state_dim=16):
        f = get_masks_from_rectangles(y0, frmsz=frmsz)
        f = feat_net(f)
        h = slim.conv2d(f, state_dim, 3, activation_fn=None)
        c = slim.conv2d(f, state_dim, 3, activation_fn=None)
        return {'h': h, 'c': c}

    def update(x, prev_state):
        '''Convert response maps to rectangle.'''
        h = prev_state['h']
        c = prev_state['c']
        h, c = conv_lstm(x, h, c, state_dim=16)
        x = h
        state = {'h': h, 'c': c}
        return x, state

    def foreground_net(x):
        # Map output of LSTM to a heatmap.
        x = slim.conv2d(x, 2, kernel_size=1, activation_fn=None, normalizer_fn=None)
        return x

    def output_net(x):
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

    def conv_lstm(x, h_prev, c_prev, state_dim, name='clstm'):
        with tf.name_scope(name) as scope:
            with slim.arg_scope([slim.conv2d],
                                num_outputs=state_dim,
                                kernel_size=3,
                                padding='SAME',
                                activation_fn=None,
                                normalizer_fn=None):
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
        return h, c

    batch_norm_opts = {} if not use_batch_norm else {
        'normalizer_fn': slim.batch_norm,
        'normalizer_params': {
            'is_training': original_example['is_training'],
            'fused': True,
        },
    }

    if normalize_size:
        window_rect = object_centric_window(original_example['y0'])
        example = crop_example(original_example, window_rect, normalize_first_only)
    else:
        example = dict(original_example)
    with tf.name_scope('model_summary') as summary_scope:
        # Visualize rectangle on normalized image.
        tf.summary.image('frame_0',
            tf.image.draw_bounding_boxes(
                normalize_image_range(example['x0'][0:1]),
                rect_to_tf_box(tf.expand_dims(example['y0'][0:1], 1))), # Just one box per image.
            collections=image_summaries_collections)

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        **batch_norm_opts):
        # Process initial image and label to get "template".
        with tf.variable_scope('template'):
            p0 = get_masks_from_rectangles(example['y0'], frmsz=frmsz)
            first_image_with_mask = concat([example['x0'], p0], axis=3)
            template = template_net(first_image_with_mask)
        # Process all images from all sequences with feature net.
        with tf.variable_scope('features'):
            x, unmerge = merge_dims(example['x'], 0, 2)
            feat = feat_net(x)
            feat = unmerge(feat, 0)
        # Search each image using result of template network.
        with tf.variable_scope('search'):
            similarity = search_all(feat, template)
        if use_rnn:
            # Update abstract position likelihood of object.
            with tf.variable_scope('track'):
                init_state = initial_state_net(example['x0'], example['y0'])
                curr_state = init_state
                similarity = tf.unstack(similarity, axis=1)
                position_map = [None] * ntimesteps
                for t in range(ntimesteps):
                    with tf.variable_scope('update', reuse=(t > 0)):
                        position_map[t], curr_state = update(similarity[t], curr_state)
                position_map = tf.stack(position_map, axis=1)
        else:
            position_map = similarity
        if use_heatmap:
            with tf.variable_scope('foreground'):
                hmap = foreground_net(position_map)
        # Transform abstract position position_map into rectangle.
        with tf.variable_scope('output'):
            position_map, unmerge = merge_dims(position_map, 0, 2)
            position = output_net(position_map)
            position = unmerge(position, 0)

    with tf.name_scope(summary_scope):
        # Visualize rectangle on normalized image.
        tf.summary.image('frame_1_to_n',
            tf.image.draw_bounding_boxes(
                normalize_image_range(example['x'][0]),
                rect_to_tf_box(tf.expand_dims(position[0], 1))), # Just one box per image.
            collections=image_summaries_collections)
    if normalize_size and not normalize_first_only:
        inv_window_rect = crop_inverse(window_rect)
        position = crop_rect_sequence(position, inv_window_rect)

    if use_rnn:
        state = {k: (init_state[k], curr_state[k]) for k in curr_state}
    else:
        state = {}

    class Model:
        pass
    model = Model()
    model.outputs = {'y': position}
    if use_heatmap:
        model.outputs['hmap'] = hmap
        hmap, unmerge = merge_dims(hmap, 0, 2)
        # TODO: Would prefer to use NEAREST_NEIGHBOR here.
        # However, it produces incorrect alignment:
        # https://github.com/tensorflow/tensorflow/issues/10989
        hmap_full = tf.image.resize_images(hmap, size=[frmsz, frmsz],
            method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        hmap_full = unmerge(hmap_full, 0)
        model.outputs['hmap_full'] = hmap_full
        model.outputs['hmap_softmax'] = tf.nn.softmax(hmap_full)
    model.state = state
    # Properties of instantiated model:
    model.image_size   = (frmsz, frmsz)
    model.sequence_len = ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model

def normalize_image_range(x):
    return 0.5 * (1 + x / tf.reduce_max(tf.abs(x)))

def crop_example(example, window_rect, first_only=False):
    '''
    Args:
        example -- Dictionary.
            example['x0'] -- [n, h, w, c]
            example['y0'] -- [n, 4]
            example['x'] -- [n, t, h, w, c]
        window_rect -- [n, 4]
    '''
    xs = tf.expand_dims(example['x0'], 1)
    if not first_only:
        xs = tf.concat([xs, example['x']], axis=1)
    im_size = xs.shape.as_list()[2:4] # Require static for now.
    xs = crop_image_sequence(xs, window_rect, crop_size=im_size)
    out = dict(example)
    out['x0'] = xs[:, 0]
    out['y0'] = crop_rects(example['y0'], window_rect)
    if not first_only:
        out['x'] = xs[:, 1:]
    return out

def object_centric_window(obj_rect, relative_size=4.0):
    eps = 0.01
    obj_min, obj_max = rect_min_max(obj_rect)
    obj_size = tf.maximum(0.0, obj_max - obj_min)
    center = 0.5 * (obj_min + obj_max)
    obj_diam = tf.exp(tf.reduce_mean(tf.log(obj_size + eps), axis=-1))
    context_diam = relative_size * obj_diam
    window_min = center - 0.5*tf.expand_dims(context_diam, -1)
    window_max = center + 0.5*tf.expand_dims(context_diam, -1)
    return make_rect(window_min, window_max)

def crop_rects(rects, window_rect):
    '''Returns each rectangle relative to a window.
    
    Args:
        rects -- [..., 4]
        window_rect -- [..., 4]
    '''
    eps = 0.01
    window_min, window_max = rect_min_max(window_rect)
    window_size = window_max - window_min
    window_size = tf.sign(window_size) * (tf.abs(window_size) + eps)
    rects_min, rects_max = rect_min_max(rects)
    out_min = (rects_min - window_min) / window_size
    out_max = (rects_max - window_min) / window_size
    return make_rect(out_min, out_max)

def crop_rect_sequence(rects, window_rect):
    '''Returns each rectangle relative to a window.
    
    Args:
        rects -- [n, t, 4]
        window_rect -- [n, 4]
    '''
    # Same rectangle in every image.
    sequence_len = tf.shape(rects)[1]
    window_rect = tf.expand_dims(window_rect, 1)
    window_rect = tf.tile(window_rect, [1, sequence_len, 1])
    # Now dimensions match.
    return crop_rects(rects, window_rect)

def crop_image_sequence(ims, window_rect, crop_size, pad_value=None):
    '''
    Extracts 

    Args:
        ims -- [n, t, h, w, c]
        window_rect -- [n, 4]
    '''
    # Same rectangle in every image.
    sequence_len = tf.shape(ims)[1]
    window_rect = tf.expand_dims(window_rect, 1)
    window_rect = tf.tile(window_rect, [1, sequence_len, 1])
    # Flatten.
    ims, unmerge = merge_dims(ims, 0, 2)
    window_rect, _ = merge_dims(window_rect, 0, 2)
    boxes = rect_to_tf_box(window_rect)
    num_images = tf.shape(ims)[0]
    crop_ims = tf.image.crop_and_resize(ims, boxes, box_ind=tf.range(num_images),
        crop_size=crop_size,
        method='bilinear',
        extrapolation_value=pad_value)
    # Un-flatten.
    crop_ims = unmerge(crop_ims, 0)
    return crop_ims

def crop_inverse(rect):
    '''Returns the rectangle that reverses the crop.

    If q = crop_inverse(r), then crop(crop(im, r), q) restores the image.
    That is, cropping is a group operation with an inverse.

    CAUTION: Epsilon means that inverse is not exact?
    '''
    eps = 0.01
    # x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    rect_min, rect_max = rect_min_max(rect)
    # TODO: Support reversed rectangle.
    rect_size = tf.abs(rect_max - rect_min) + eps
    # x_size = tf.abs(x_max - x_min) + eps
    # y_size = tf.abs(y_max - y_min) + eps
    inv_min = -rect_min / rect_size
    # u_min = -x_min / x_size
    # v_min = -y_min / y_size
    inv_max = (1 - rect_min) / rect_size
    # inv_max = inv_min + 1 / rect_size
    # u_max = u_min + 1 / x_size
    # v_max = v_min + 1 / y_size
    return make_rect(inv_min, inv_max)

def rect_min_max(rect):
    x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    min_pt = tf.stack([x_min, y_min], axis=-1)
    max_pt = tf.stack([x_max, y_max], axis=-1)
    return min_pt, max_pt

def make_rect(min_pt, max_pt):
    x_min, y_min = tf.unstack(min_pt, axis=-1)
    x_max, y_max = tf.unstack(max_pt, axis=-1)
    return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

def rect_to_tf_box(rect):
    x_min, y_min, x_max, y_max = tf.unstack(rect, axis=-1)
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)


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


def load_model(o, model_params=None):
    '''
    example is a dictionary that maps strings to Tensors.
    Its keys should include 'inputs', 'labels', 'x0', 'y0'.
    '''
    model_params = model_params or {}
    assert('summaries_collections' not in model_params)
    if o.model == 'RNN_dual':
        model = functools.partial(RNN_dual, o=o, **model_params)
    elif o.model == 'RNN_conv_asymm':
        model = functools.partial(rnn_conv_asymm, o=o, **model_params)
    elif o.model == 'RNN_multi_res':
        model = functools.partial(rnn_multi_res, o=o, **model_params)
    elif o.model == 'simple_search':
        model = functools.partial(simple_search,
            ntimesteps=o.ntimesteps,
            frmsz=o.frmsz,
            batchsz=o.batchsz,
            weight_decay=o.wd,
            **model_params)
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

