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

The `outputs` dictionary should have a key 'y' and may have other keys such as 'heatmap'.
'''

import pdb
import functools
import tensorflow as tf
from tensorflow.contrib import slim
import os

import numpy as np

import cnnutil
from helpers import merge_dims


def convert_rec_to_heatmap(rec, o, min_size=None):
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
        masks = get_masks_from_rectangles(rec, o, kind='bg', min_size=min_size)
        return unmerge(masks, 0)

def get_masks_from_rectangles(rec, o, kind='fg', typecast=True, min_size=None, name='mask'):
    with tf.name_scope(name) as scope:
        # create mask using rec; typically rec=y_prev
        # rec -- [b, 4]
        rec *= float(o.frmsz)
        # x1, y1, x2, y2 -- [b]
        x1, y1, x2, y2 = tf.unstack(rec, axis=1)
        if min_size is not None:
            x1, y1, x2, y2 = enforce_min_size(x1, y1, x2, y2, min_size=min_size)
        # grid_x -- [1, frmsz]
        # grid_y -- [frmsz, 1]
        grid_x = tf.expand_dims(tf.cast(tf.range(o.frmsz), o.dtype), 0)
        grid_y = tf.expand_dims(tf.cast(tf.range(o.frmsz), o.dtype), 1)
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
            masks = tf.concat(
                    (tf.expand_dims(masks,3), tf.expand_dims(masks_bg,3)), 3)
        if typecast: # type cast so that it can be concatenated with x
            masks = tf.cast(masks, o.dtype)
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
                 is_training=True,
                 summaries_collections=None):
        # Ignore is_training  - model does not change in training vs testing.
        # Ignore sumaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_cnn1(x, name):
            # NOTE: whether severe intermediate downsampling was truly necessary? 
            # -> Maybe not for cnn1 -> Dilated convolution.
            # Dilated convolution will remove the need of pooling or subsampling.
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
                    x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
                    x = slim.max_pool2d(x, 2, stride=2, scope='pool1')
                    x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
            return x

        def pass_cnn2(x, name):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    with slim.arg_scope([slim.conv2d],
                            padding='VALID'):
                        x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
                        x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
                        x = slim.max_pool2d(x, 2, scope='pool1')
                        x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
                        x = slim.max_pool2d(x, 2, scope='pool2')
                        x = slim.flatten(x)
                        x = slim.fully_connected(x, 256, scope='fc1')
            return x

        def pass_lstm1(x, h_prev, c_prev, name):
            # TODO: multiple layers
            with tf.name_scope(name):
                ft, it, ct_tilda, ot = tf.split(
                    slim.fully_connected(
                        tf.concat((h_prev, x), 1),
                        o.nunits*4,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(o.wd)),
                    4, 1)
                ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
                ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
            return ht, ct

        def pass_project_for_cross_correlation(h, name):
            # TODO: maybe do not reduce the feature vector size to much.
            with tf.name_scope(name):
                out = slim.fully_connected(h, 64)
                out = tf.expand_dims(out, 1)
                out = tf.expand_dims(out, 1)
            return out

        def pass_cross_correlation(search, filt, name):
            # TODO: mutli-scale cross-correlation.
            # TODO: How can I achieve this without using o.batchsz?
            with tf.name_scope(name):
                scoremap = []
                for i in range(o.batchsz):
                    searchimg = tf.expand_dims(search[i],0)
                    filtimg = tf.expand_dims(filt[i],3)
                    scoremap.append(
                        tf.nn.depthwise_conv2d(searchimg, filtimg, [1,1,1,1], 'SAME'))
                scoremap = tf.squeeze(tf.stack(scoremap, axis=0), squeeze_dims=1)
                scoremap.set_shape([None, None, None, 64]) # TODO: is this working?
            return scoremap

        def pass_project_for_lstm2(x, name):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 32, [1, 1], scope='conv1')
                    x = slim.conv2d(x, 2,  [1, 1], scope='conv2')
            return x

        def pass_lstm2(x, h_prev, c_prev, name):
            ''' ConvLSTM
            h and c have the same dimension as x (padding can be used)
            '''
            # TODO: multiple layers
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

        def pass_deconvolution(x, name):
            '''
            There are multiple ways to cast deconvolution. 
            Here, I use `(bilinear) image resize`+ `(regular) convolution` approach. 
            Another alternative can be transposed convolution (initiallized with bilinear kernel).
            Currently, # of deconv = # size changes in cnn1 = 4.
            '''
            # TODO: add stride 1 deconvolution at the last layer
            # TODO: Add skip connections to combine multi-scale outputs.
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        num_outputs=2,
                        kernel_size=[3,3], # TODO: Is it okay?
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = tf.image.resize_images(x, [38, 38]) 
                    x = slim.conv2d(x, scope='conv1')
                    x = tf.image.resize_images(x, [79, 79]) 
                    x = slim.conv2d(x, scope='conv2')
                    x = tf.image.resize_images(x, [241, 241]) 
                    x = slim.conv2d(x, scope='conv3')
                    #x = slim.conv2d_transpose(x, kernel_size=[3,3], stride=1, scope='deconv1')
                    #x = slim.conv2d_transpose(x, kernel_size=[5,5], stride=2, padding='SAME', scope='deconv2')
                    #x = slim.conv2d_transpose(x, kernel_size=[5,5], stride=2, scope='deconv3')
                    #x = slim.conv2d_transpose(x, kernel_size=[7,7], stride=3, scope='deconv4')
            return x

        def pass_cnn_out_hmap(x, name):
            ''' Two 1x1 convolutions for output hmap
            '''
            # TODO: skip connections for multi-scale output.
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        num_outputs=2,
                        kernel_size=[1,1],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, scope='conv1')
                    x = slim.conv2d(x, activation_fn=None, scope='conv2')
            return x

        def pass_cnn_out_rec(x, name):
            ''' Another cnn for output rectangle
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.max_pool2d(x, 4, stride=4, scope='pool1')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1')
                    x = slim.fully_connected(x, 512, scope='fc2')
                    x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
                '''
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    with slim.arg_scope([slim.conv2d],
                            num_outputs=2):
                        x = slim.conv2d(x, kernel_size=[7,7], stride=3, scope='conv1')
                        x = slim.conv2d(x, kernel_size=[5,5], stride=2, scope='conv2')
                        x = slim.conv2d(x, kernel_size=[3,3], stride=1, scope='conv3')
                        x = slim.flatten(x)
                        x = slim.fully_connected(x, 256, scope='fc1')
                        x = slim.fully_connected(x, 4, activation_fn=None, scope='fc2')
                '''
            return x


        x       = inputs['x']
        x0      = inputs['x0']
        y0      = inputs['y0']
        y       = inputs['y']
        use_gt  = inputs['use_gt']

        with tf.name_scope('lstm_initial'):
            with slim.arg_scope([slim.model_variable],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    regularizer=slim.l2_regularizer(o.wd)):
                h_init1_single = slim.model_variable('lstm1_h_init', shape=[o.nunits])
                c_init1_single = slim.model_variable('lstm1_c_init', shape=[o.nunits])
                h_init2_single = slim.model_variable('lstm2_h_init', shape=[17, 17, 2]) # TODO: adaptive
                c_init2_single = slim.model_variable('lstm2_c_init', shape=[17, 17, 2])
                h_init1 = tf.stack([h_init1_single] * o.batchsz)
                c_init1 = tf.stack([c_init1_single] * o.batchsz)
                h_init2 = tf.stack([h_init2_single] * o.batchsz)
                c_init2 = tf.stack([c_init2_single] * o.batchsz)

        y_pred = []
        hmap_pred = []

        deconvouts = []
        h2 = []

        ht1, ct1 = h_init1, c_init1
        ht2, ct2 = h_init2, c_init2
        hmap_init = get_masks_from_rectangles(y0, o)
        for t in range(o.ntimesteps):
            xt = x[:, t]
            yt = y[:, t]

            with tf.name_scope('cnn1_{}'.format(t)) as scope:
                with tf.variable_scope('cnn1', reuse=(t > 0)):
                    cnn1out = pass_cnn1(xt, scope)

            with tf.name_scope('cnn2_{}'.format(t)) as scope:
                with tf.variable_scope('cnn2', reuse=(t > 0)):
                    if t==0:
                        hmap = hmap_init
                    else:
                        hmap = tf.cond(use_gt,
                            lambda: get_masks_from_rectangles(yt, o),
                            lambda: tf.expand_dims(tf.nn.softmax(hmapt_pred)[:,:,:,0], 3))
                    xy = tf.concat([xt, hmap], axis=3)
                    cnn2out = pass_cnn2(xy, scope)

            with tf.name_scope('lstm1_{}'.format(t)) as scope:
                with tf.variable_scope('lstm1', reuse=(t > 0)):
                    ht1, ct1 = pass_lstm1(cnn2out, ht1, ct1, scope)

            with tf.name_scope('project_for_cross_correlation_{}'.format(t)) as scope:
                with tf.variable_scope('project_for_cross_correlation', reuse=(t > 0)):
                    ht1_proj = pass_project_for_cross_correlation(ht1, scope)

            with tf.name_scope('cross_correlation_{}'.format(t)) as scope:
                with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                    scoremap = pass_cross_correlation(cnn1out, ht1_proj, scope)

            with tf.name_scope('project_for_lstm2_{}'.format(t)) as scope:
                with tf.variable_scope('project_for_lstm2', reuse=(t > 0)):
                    scoremap = pass_project_for_lstm2(scoremap, scope)

            with tf.name_scope('lstm2_{}'.format(t)) as scope:
                with tf.variable_scope('lstm2', reuse=(t > 0)):
                    ht2, ct2 = pass_lstm2(scoremap, ht2, ct2, scope)

            with tf.name_scope('deconvolution_{}'.format(t)) as scope:
                with tf.variable_scope('deconvolution', reuse=(t > 0)):
                    deconvout = pass_deconvolution(ht2, scope)

            # NOTE: `pass_cnn_out_hmap` takes really long time.. why?
            with tf.name_scope('cnn_out_hmap_{}'.format(t)) as scope:
                with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                    hmapt_pred = pass_cnn_out_hmap(deconvout, scope)

            with tf.name_scope('cnn_out_rec_{}'.format(t)) as scope:
                with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                    yt_pred = pass_cnn_out_rec(deconvout, scope)

            #decovnout = tf.Print(deconvout, [deconvout[:2, :10, :10, :]], 'deconvout: ')
            #yt_pred = tf.Print(yt_pred, [yt_pred[0]], 'yt_pred[0]: ')
            #use_gt = tf.Print(use_gt, [use_gt], 'use_gt: ')

            h2.append(ht2)
            deconvouts.append(deconvout)

            hmap_pred.append(hmapt_pred)
            y_pred.append(yt_pred)

        hmap_pred = tf.stack(hmap_pred, axis=1)
        y_pred = tf.stack(y_pred, axis=1) # list to tensor

        outputs = {'y': y_pred, 'hmap': hmap_pred}
        state = {'h1': (h_init1, ht1), 'c1': (c_init1, ct1),
                 'h2': (h_init2, ht2), 'c2': (c_init2, ct2),
                 'hmap': (hmap_init, hmap),
                 'x': (x0, x[:, -1])}
        dbg = {'y_pred': y_pred, 'deconvout': tf.stack(deconvouts, axis=1), 'h2': tf.stack(h2, axis=1)}
        return outputs, state, dbg


class RNN_dual_rec(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 is_training=True,
                 summaries_collections=None):
        # Ignore is_training  - model does not change in training vs testing.
        # Ignore sumaries_collections - model does not generate any summaries.
        self.is_train = True if o.mode == 'train' else False
        self.outputs, self.state, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_cnn1(x, name):
            # NOTE: whether severe intermediate downsampling was truly necessary?
            # -> Maybe not for cnn1 -> Dilated convolution.
            # Dilated convolution will remove the need of pooling or subsampling.
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        padding='VALID',
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
                    x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
                    x = slim.max_pool2d(x, 2, stride=2, scope='pool1')
                    #x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
                    x = slim.conv2d(x, 64, [3, 3], stride=1, rate=2, scope='conv3')
                    x = slim.conv2d(x, 64, [3, 3], stride=1, rate=2, scope='conv4')
            return x

        def pass_cnn2(x, name):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    with slim.arg_scope([slim.conv2d],
                            padding='VALID'):
                        x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
                        x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
                        x = slim.max_pool2d(x, 2, scope='pool1')
                        x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
                        x = slim.max_pool2d(x, 2, scope='pool2')
                        x = slim.flatten(x)
                        x = slim.fully_connected(x, 256, scope='fc1')
            return x

        def pass_lstm1(x, h_prev, c_prev, name):
            # TODO: multiple layers
            with tf.name_scope(name):
                ft, it, ct_tilda, ot = tf.split(
                    slim.fully_connected(
                        tf.concat_v2((h_prev, x), 1),
                        o.nunits*4,
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(o.wd)),
                    4, 1)
                ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
                ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
            return ht, ct

        def pass_project_for_cross_correlation(h, name):
            # TODO: maybe do not reduce the feature vector size to much.
            with tf.name_scope(name):
                out = slim.fully_connected(h, 64)
                out = tf.expand_dims(out, 1)
                out = tf.expand_dims(out, 1)
            return out

        def pass_cross_correlation(search, filt, name):
            # TODO: mutli-scale cross-correlation.
            # TODO: How can I achieve this without using o.batchsz?
            with tf.name_scope(name):
                scoremap = []
                for i in range(o.batchsz):
                    searchimg = tf.expand_dims(search[i],0)
                    filtimg = tf.expand_dims(filt[i],3)
                    scoremap.append(
                        tf.nn.depthwise_conv2d(searchimg, filtimg, [1,1,1,1], 'SAME'))
                scoremap = tf.squeeze(tf.stack(scoremap, axis=0), squeeze_dims=1)
                scoremap.set_shape([None, None, None, 64]) # TODO: is this working?
            return scoremap

        def pass_project_for_lstm2(x, name):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 32, [1, 1], scope='conv1')
                    x = slim.conv2d(x, 2,  [1, 1], scope='conv2')
            return x

        def pass_lstm2(x, h_prev, c_prev, name):
            ''' ConvLSTM
            h and c have the same dimension as x (padding can be used)
            '''
            # TODO: multiple layers
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

        def pass_cnn_out_rec(x, name):
            ''' Another cnn for output rectangle
            '''
            with tf.name_scope(name):
                with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 512, scope='fc1')
                    x = slim.fully_connected(x, 256, scope='fc2')
                    x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x


        x       = inputs['x']  # shape [b, ntimesteps, h, w, 3]
        x0      = inputs['x0'] # shape [b, h, w, 3]
        y0      = inputs['y0'] # shape [b, 4]
        y       = inputs['y']  # shape [b, ntimesteps, 4]

        with tf.name_scope('lstm_initial'):
            with slim.arg_scope([slim.model_variable],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    regularizer=slim.l2_regularizer(o.wd)):
                h1_init_single = slim.model_variable('lstm1_h_init', shape=[o.nunits])
                c1_init_single = slim.model_variable('lstm1_c_init', shape=[o.nunits])
                h2_init_single = slim.model_variable('lstm2_h_init', shape=[11, 11, 2]) # TODO: adaptive
                c2_init_single = slim.model_variable('lstm2_c_init', shape=[11, 11, 2])
                h1_init = tf.stack([h1_init_single] * o.batchsz)
                c1_init = tf.stack([c1_init_single] * o.batchsz)
                h2_init = tf.stack([h2_init_single] * o.batchsz)
                c2_init = tf.stack([c2_init_single] * o.batchsz)

        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0)
        x_prev = x_init
        y_prev = y_init
        h1_prev, c1_prev = h1_init, c1_init
        h2_prev, c2_prev = h2_init, c2_init
        y_pred = []
        h2 = []
        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]
            with tf.name_scope('cnn1_{}'.format(t)) as scope:
                with tf.variable_scope('cnn1', reuse=(t > 0)):
                    cnn1out = pass_cnn1(x_curr, scope)

            with tf.name_scope('cnn2_{}'.format(t)) as scope:
                with tf.variable_scope('cnn2', reuse=(t > 0)):
                    hmap = get_masks_from_rectangles(y_prev, o)
                    xy = tf.concat([x_prev, hmap], axis=3)
                    cnn2out = pass_cnn2(xy, scope)

            with tf.name_scope('lstm1_{}'.format(t)) as scope:
                with tf.variable_scope('lstm1', reuse=(t > 0)):
                    h1_curr, c1_curr = pass_lstm1(cnn2out, h1_prev, c1_prev, scope)

            with tf.name_scope('project_for_cross_correlation_{}'.format(t)) as scope:
                with tf.variable_scope('project_for_cross_correlation', reuse=(t > 0)):
                    h1_curr_proj = pass_project_for_cross_correlation(h1_curr, scope)

            with tf.name_scope('cross_correlation_{}'.format(t)) as scope:
                with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                    scoremap = pass_cross_correlation(cnn1out, h1_curr_proj, scope)

            with tf.name_scope('project_for_lstm2_{}'.format(t)) as scope:
                with tf.variable_scope('project_for_lstm2', reuse=(t > 0)):
                    scoremap = pass_project_for_lstm2(scoremap, scope)

            with tf.name_scope('lstm2_{}'.format(t)) as scope:
                with tf.variable_scope('lstm2', reuse=(t > 0)):
                    h2_curr, c2_curr = pass_lstm2(scoremap, h2_prev, c2_prev, scope)

            with tf.name_scope('cnn_out_rec_{}'.format(t)) as scope:
                with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                    y_curr_pred = pass_cnn_out_rec(h2_curr, scope)

            x_prev = x_curr
            y_prev = y_curr # TODO: 'use_gt'
            h1_prev, c1_prev = h1_curr, c1_curr
            h2_prev, c2_prev = h2_curr, c2_curr
            y_pred.append(y_curr_pred)
            h2.append(h2_curr)

        y_pred = tf.stack(y_pred, axis=1) # list to tensor

        outputs = {'y': y_pred}
        state = {'h1': (h1_init, h1_curr), 'c1': (c1_init, c1_curr),
                 'h2': (h2_init, h2_curr), 'c2': (c2_init, c2_curr),
                 'x': (x_init, x_curr), 'y': (y_init, y_curr)} # TODO: 'use_gt'
        dbg = {'y_pred': y_pred, 'h2': tf.stack(h2, axis=1)}
        return outputs, state, dbg


def rnn_conv_asymm(example, o,
                   is_training=True,
                   summaries_collections=None,
                   # Model parameters:
                   input_num_layers=3,
                   input_kernel_size=[7, 5, 3],
                   input_num_channels=[16, 32, 64],
                   input_stride=[2, 1, 1],
                   input_pool=[True, True, True],
                   input_pool_stride=[2, 2, 2],
                   input_pool_kernel_size=[3, 3, 3],
                   input_batch_norm=False):
                   # lstm_num_layers=1,
                   # lstm_kernel_size=[3]):

    images = example['x']
    x0     = example['x0']
    y0     = example['y0']
    masks = get_masks_from_rectangles(y0, o)
    if o.debugmode:
        with tf.name_scope('input_preview'):
            tf.summary.image('x', images[0], collections=summaries_collections)
            target = tf.concat([images[0, 0], masks[0]], axis=2)
            tf.summary.image('target', tf.expand_dims(target, axis=0),
                             collections=summaries_collections)
    if o.activ_histogram:
        with tf.name_scope('input_histogram'):
            tf.summary.histogram('x', images, collections=summaries_collections)
    init_input = tf.concat([x0, masks], axis=3)

    assert(len(input_kernel_size)      == input_num_layers)
    assert(len(input_num_channels)     == input_num_layers)
    assert(len(input_stride)           == input_num_layers)
    assert(len(input_pool)             == input_num_layers)
    assert(len(input_pool_stride)      == input_num_layers)
    assert(len(input_pool_kernel_size) == input_num_layers)
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
                layers = {'i': i, 'f': f, 'o': y, 'c_tilde': c, 'c': c, 'h': h}
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

    lstm_dim = 64
    # At start of sequence, compute hidden state from first example.
    # Feed (h_init, c_init) to resume tracking from previous state.
    # Do NOT feed (h_init, c_init) when starting new sequence.
    with tf.name_scope('h_init') as scope:
        with tf.variable_scope('h_init'):
            h_init = input_cnn(init_input, num_outputs=lstm_dim, name=scope)
    with tf.name_scope('c_init') as scope:
        with tf.variable_scope('c_init'):
            c_init = input_cnn(init_input, num_outputs=lstm_dim, name=scope)

    # # TODO: Process all frames together in training (when sequences are equal length)
    # # (because it enables batch-norm to operate on whole sequence)
    # # but not during testing (when sequences are different lengths)
    # x, unmerge = merge_dims(images, 0, 2)
    # with tf.name_scope('frame_cnn') as scope:
    #     with tf.variable_scope('frame_cnn'):
    #         # Pass name scope from above, otherwise makes new name scope
    #         # within name scope created by variable scope.
    #         r = input_cnn(x, num_outputs=lstm_dim, name=scope)
    # r = unmerge(r, 0)

    y = []
    ht, ct = h_init, c_init
    for t in range(o.ntimesteps):
        xt = images[:, t]
        with tf.name_scope('frame_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('frame_cnn', reuse=(t > 0)):
                # Pass name scope from above, otherwise makes new name scope
                # within name scope created by variable scope.
                rt = input_cnn(xt, num_outputs=lstm_dim, name=scope)
        with tf.name_scope('conv_lstm_{}'.format(t)) as scope:
            with tf.variable_scope('conv_lstm', reuse=(t > 0)):
                ht, ct = conv_lstm(rt, ht, ct, state_dim=lstm_dim, name=scope)
        with tf.name_scope('out_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('out_cnn', reuse=(t > 0)):
                yt = output_cnn(ht, name=scope)
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
    state = {'h': (h_init, h_last), 'c': (c_init, c_last)}

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


def load_model(o, model_params=None):
    '''
    example is a dictionary that maps strings to Tensors.
    Its keys should include 'inputs', 'labels', 'x0', 'y0'.
    '''
    model_params = model_params or {}
    assert('is_training' not in model_params)
    assert('summaries_collections' not in model_params)
    if o.model == 'RNN_dual':
        model = functools.partial(RNN_dual, o=o, **model_params)
    elif o.model == 'RNN_dual_rec':
        model = functools.partial(RNN_dual_rec, o=o, **model_params)
    elif o.model == 'RNN_conv_asymm':
        model = functools.partial(rnn_conv_asymm, o=o, **model_params)
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

