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
            masks = concat(
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

def process_image_with_hmap(x, hmap, o, mode, margin_ratio=3.0):
    '''
    Process input image x with hmap depending on given mode {crop, mask}.
        'crop': Crop image based on hmap. Used for target appearance.
        'mask': Mask image based on hmap. Used for search space.

    input hmap ranges [0,1] due to softmax beforehand.

    hyperparameters.
        filt, strides, rates: dilation filter kernel.
        threshold: the value [0,1) to binarize hmap.
        crop_size: the output size of crop after resizing (default o.frmsz/2 = 120).
        margin_ratio: Sets the search space size with respect to object's size
            (e.g., if set 2.0 the search area will be twice bigger than object).
    '''
    # image dilation
    #filt = tf.ones(shape=[3, 3, 1]) # conservative for now to reduce the effect.
    #strides = [1, 1, 1, 1]
    #rates = [1, 1, 1, 1]
    #hmap = tf.nn.dilation2d(hmap, filt, strides, rates, padding='SAME')

    # normalize to have max of 1 (so that I can apply fixed threshold).
    hmap = hmap / tf.reduce_max(hmap, axis=(1,2), keep_dims=True)

    # find indices. Then either crop (crop-and-resize) or mask.
    threshold = 0.9 # TODO: try different threshold. Should be better with some bound after thresholding.
    x_out = []
    area = [] # visualize the area being affected. For debugging purpose.
    for b in range(o.batchsz):
        indices = tf.where(hmap[b] > threshold)
        x1 = tf.cast(tf.reduce_min(indices[:,1]), o.dtype) / o.frmsz
        y1 = tf.cast(tf.reduce_min(indices[:,0]), o.dtype) / o.frmsz
        x2 = tf.cast(tf.reduce_max(indices[:,1]), o.dtype) / o.frmsz
        y2 = tf.cast(tf.reduce_max(indices[:,0]), o.dtype) / o.frmsz
        if mode == 'crop':
            box = tf.expand_dims(tf.cast(tf.stack((y1, x1, y2, x2), 0), o.dtype), 0)
            crop = tf.image.crop_and_resize(tf.expand_dims(x[b],0),
                                        boxes=box,
                                        box_ind=[0],
                                        crop_size=[o.frmsz/2, o.frmsz/2])
            x_out.append(crop)
            area.append(get_masks_from_rectangles(box, o))
        elif mode == 'mask':
            w_margin = (x2 - x1) * 0.5 * (margin_ratio - 1.0)
            h_margin = (y2 - y1) * 0.5 * (margin_ratio - 1.0)
            x1 = tf.maximum(x1 - w_margin, 0.0)
            y1 = tf.maximum(y1 - h_margin, 0.0)
            x2 = tf.minimum(x2 + w_margin, 1.0)
            y2 = tf.minimum(y2 + h_margin, 1.0)
            box = tf.expand_dims(tf.cast(tf.stack((y1, x1, y2, x2), 0), o.dtype), 0)
            mask = get_masks_from_rectangles(box, o)
            search = mask[0] * x[b]
            x_out.append(tf.expand_dims(search, 0))
            area.append(mask)

    return tf.concat(x_out, 0), tf.concat(area, 0)


class Nornn(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 ):
        # model parameters
        # Ignore sumaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_cnn(x):
            ''' Fully convolutional cnn.
            '''
            with slim.arg_scope([slim.conv2d],
                    padding='VALID',
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(x, 96, 11, stride=2, scope='conv1')
                x = slim.max_pool2d(x, 3, scope='pool1')
                x = slim.conv2d(x, 256, 5, stride=1, scope='conv2')
                x = slim.max_pool2d(x, 3, scope='pool2')
                x = slim.conv2d(x, 384, 3, stride=1, scope='conv3')
                x = slim.conv2d(x, 384, 3, stride=1, scope='conv4')
                x = slim.conv2d(x, 256, 3, stride=1, scope='conv5')
            return x

        def pass_cross_correlation(search, filt, o):
            # TODO: ICCV paper performed normal conv rather than depthwise_conv2d.
            scoremap = []
            for i in range(o.batchsz):
                scoremap.append(
                        tf.nn.depthwise_conv2d(tf.expand_dims(search[i],0),
                                               tf.expand_dims(filt[i], 3),
                                               strides=[1,1,1,1],
                                               padding='SAME'))
            return tf.concat(scoremap, 0)

        def pass_deconvolution(x):
            shape_to = [53, 116] # magic number picked from CNN.
            numout_to = [256, 96]
            with slim.arg_scope([slim.conv2d],
                    kernel_size=[3,3],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                for i in range(len(shape_to)):
                    x = slim.conv2d(tf.image.resize_images(x, [shape_to[i]]*2),
                                    num_outputs=numout_to[i],
                                    scope='deconv{}'.format(i+1))
            return x

        def pass_out_rectangle(x):
            ''' Regress output rectangle.
            '''
            with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                # add size reduction operations.
                x = slim.conv2d(x, 16, 1, scope='conv1')
                x = slim.conv2d(x, 2, 1, scope='conv2')
                x = slim.max_pool2d(x, 2, scope='pool1')
                x = slim.flatten(x)
                x = slim.fully_connected(x, 1024, scope='fc1')
                x = slim.fully_connected(x, 1024, scope='fc2')
                x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        def pass_out_heatmap(x):
            ''' Upsample and generate spatial heatmap.
            '''
            with slim.arg_scope([slim.conv2d],
                    #num_outputs=x.shape.as_list()[-1],
                    num_outputs=2,
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(tf.image.resize_images(x, [241, 241]), kernel_size=[3, 3], scope='deconv')
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
        y_init = tf.identity(y0) # for `delta` regression type output.
        hmap_init = tf.identity(get_masks_from_rectangles(y0, o))

        x_prev = x_init
        hmap_prev = hmap_init

        y_pred = []
        hmap_pred = []

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]

            with tf.variable_scope('cnn1', reuse=(t > 0)):
                search, _ = process_image_with_hmap(x_curr, hmap_prev, o, mode='mask')
                feat_search = pass_cnn(search)

            with tf.variable_scope('cnn2', reuse=(t > 0)):
                target, _ = process_image_with_hmap(x_prev, hmap_prev, o, mode='crop')
                feat_target = pass_cnn(target)

            with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                scoremap = pass_cross_correlation(feat_search, feat_target, o)

            with tf.variable_scope('deconvolution', reuse=(t > 0)):
                scoremap = pass_deconvolution(scoremap)

            with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                hmap_curr_pred = pass_out_heatmap(scoremap)

            with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                y_curr_pred = pass_out_rectangle(scoremap)

            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            hmap_curr_gt = tf.identity(get_masks_from_rectangles(y_curr, o))
            hmap_prev = tf.cond(gt_condition, lambda: hmap_curr_gt,
                                              lambda: tf.expand_dims(tf.nn.softmax(hmap_curr_pred)[:,:,:,0], 3))

            x_prev = x_curr

            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr_pred)

        y_pred = tf.stack(y_pred, axis=1) # list to tensor
        hmap_pred = tf.stack(hmap_pred, axis=1)
        y_prev = y_pred[:,-1,:] # for `delta` regression type output.

        outputs = {'y': y_pred, 'hmap': hmap_pred, 'hmap_softmax': tf.nn.softmax(hmap_pred)}
        state = {}
        state.update({'x': (x_init, x_prev)})
        state.update({'hmap': (hmap_init, hmap_prev)})
        state.update({'y': (y_init, y_prev)})

        #dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        dbg = {}
        return outputs, state, dbg


class RNN_dual_mix(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 crop_target=False,
                 mask_search=False,
                 lstm1_nlayers=1,
                 lstm2_nlayers=1,
                 layer_norm=False,
                 residual_lstm=False,
                 feed_examplar=False,
                 dropout_rnn=False,
                 keep_prob=0.2, # following `Recurrent Neural Network Regularization, Zaremba et al.
                 ):
        # model parameters
        self.crop_target   = crop_target
        self.mask_search   = mask_search
        self.lstm1_nlayers = lstm1_nlayers
        self.lstm2_nlayers = lstm2_nlayers
        self.layer_norm    = layer_norm
        self.residual_lstm = residual_lstm
        self.feed_examplar = feed_examplar
        self.dropout_rnn   = dropout_rnn
        self.keep_prob     = keep_prob
        # Ignore sumaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_cnn(x, fully_connected):
            out = []
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
                if fully_connected:
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1'); out.append(x)
                    x = slim.fully_connected(x, 1024, scope='fc2'); out.append(x)
            return out

        #def pass_lstm1(x, h_prev, c_prev):
        #    with slim.arg_scope([slim.fully_connected],
        #            num_outputs=o.nunits,
        #            activation_fn=None,
        #            weights_regularizer=slim.l2_regularizer(o.wd)):
        #        # NOTE: `An Empirical Exploration of Recurrent Neural Network Architecture`.
        #        # Initialize forget bias to be 1.
        #        # They also use `tanh` instead of `sigmoid` for input gate. (yet not employed here)
        #        ft = slim.fully_connected(concat((h_prev, x), 1), biases_initializer=tf.ones_initializer(), scope='hf')
        #        it = slim.fully_connected(concat((h_prev, x), 1), scope='hi')
        #        ct_tilda = slim.fully_connected(concat((h_prev, x), 1), scope='hc')
        #        ot = slim.fully_connected(concat((h_prev, x), 1), scope='ho')
        #        ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
        #        ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
        #    return ht, ct

        def pass_lstm1(x, h_prev, c_prev):
            '''
            `forget` bias is initialized to be 1 as in
            `An Empirical Exploration of Recurrent Neural Network Architecture`.
            (with zeros initialization for all gates, training fails!!!)
            As moving to layer normalization, I compute linear functions of
            input and hidden separately (all at once for 4 gates as before).
            '''
            def ln(inputs, epsilon = 1e-5, scope = None):
                mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
                with tf.variable_scope(scope + 'LN'):
                    scale = tf.get_variable('alpha', shape=[inputs.get_shape()[1]],
                                            initializer=tf.constant_initializer(1))
                    shift = tf.get_variable('beta', shape=[inputs.get_shape()[1]],
                                            initializer=tf.constant_initializer(0))
                LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
                return LN

            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                biases_initializer=None,
                                weights_regularizer=slim.l2_regularizer(o.wd)):
                x_linear = slim.fully_connected(x, 4*o.nunits, scope='x_linear')
                h_linear = slim.fully_connected(h_prev, 4*o.nunits, scope='h_linear')

            if self.layer_norm:
                x_linear = ln(x_linear, scope='x/')
                h_linear = ln(h_linear, scope='h/')

            ft, it, ot, ct_tilda = tf.split(x_linear + h_linear, 4, axis=1)

            with tf.variable_scope('bias'):
                bf = tf.get_variable('bf', shape=[o.nunits], initializer=tf.ones_initializer())
                bi = tf.get_variable('bi', shape=[o.nunits], initializer=tf.zeros_initializer())
                bo = tf.get_variable('bo', shape=[o.nunits], initializer=tf.zeros_initializer())
                bc = tf.get_variable('bc', shape=[o.nunits], initializer=tf.zeros_initializer())

            ft = ft + bf
            it = it + bi
            ot = ot + bo
            ct_tilda = ct_tilda + bc

            ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
            ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
            return ht, ct

        def pass_multi_level_cross_correlation(search, filt):
            ''' Multi-level cross-correlation function producing scoremaps.
            Option 1: depth-wise convolution
            Option 2: similarity score (-> doesn't work well)
            Note that depth-wise convolution with 1x1 filter is actually same as
            channel-wise (and element-wise) multiplication.
            '''
            # TODO: sigmoid or softmax over scoremap?
            # channel-wise l2 normalization as in Universal Correspondence Network?
            scoremap = []
            with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                for i in range(len(search)):
                    depth = search[i].shape.as_list()[-1]
                    scoremap.append(search[i] *
                            tf.expand_dims(tf.expand_dims(slim.fully_connected(filt, depth), 1), 1))
            return scoremap

        def pass_multi_level_deconvolution(x):
            ''' Multi-level deconvolutions.
            This is in a way similar to HourglassNet.
            Using sum.
            '''
            deconv = x[-1]
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

        def pass_lstm2(x, h_prev, c_prev):
            ''' ConvLSTM
            h and c have the same spatial dimension as x.
            '''
            # TODO: increase size of hidden
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

        def pass_out_rectangle(x):
            ''' Regress output rectangle.
            '''
            with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.flatten(x)
                x = slim.fully_connected(x, 1024, scope='fc1')
                x = slim.fully_connected(x, 1024, scope='fc2')
                x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        def pass_out_heatmap(x):
            ''' Upsample and generate spatial heatmap.
            '''
            with slim.arg_scope([slim.conv2d],
                    #num_outputs=x.shape.as_list()[-1],
                    num_outputs=2,
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(tf.image.resize_images(x, [241, 241]), kernel_size=[3, 3], scope='deconv')
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

        if self.feed_examplar:
            examplar = pass_cnn(concat([x0, get_masks_from_rectangles(y0, o)], axis=3), True)[-1]

        h1_init = [None] * self.lstm1_nlayers
        c1_init = [None] * self.lstm1_nlayers
        h2_init = [None] * self.lstm2_nlayers
        c2_init = [None] * self.lstm2_nlayers
        with tf.variable_scope('lstm_init'):
            with slim.arg_scope([slim.model_variable],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    regularizer=slim.l2_regularizer(o.wd)):
                for i in range(self.lstm1_nlayers):
                    h1_init_single = slim.model_variable('h1_{}'.format(i+1), shape=[o.nunits])
                    c1_init_single = slim.model_variable('c1_{}'.format(i+1), shape=[o.nunits])
                    h1_init[i] = tf.stack([h1_init_single] * o.batchsz)
                    c1_init[i] = tf.stack([c1_init_single] * o.batchsz)
                for i in range(self.lstm2_nlayers):
                    h2_init_single = slim.model_variable('h2_{}'.format(i+1), shape=[81, 81, 2])
                    c2_init_single = slim.model_variable('c2_{}'.format(i+1), shape=[81, 81, 2])
                    h2_init[i] = tf.stack([h2_init_single] * o.batchsz)
                    c2_init[i] = tf.stack([c2_init_single] * o.batchsz)


        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0) # for `delta` regression type output.
        hmap_init = tf.identity(get_masks_from_rectangles(y0, o))

        x_prev = x_init
        hmap_prev = hmap_init
        h1_prev, c1_prev = h1_init, c1_init
        h2_prev, c2_prev = h2_init, c2_init

        y_pred = []
        hmap_pred = []

        cnn1in, cnn1area = [], []
        cnn2in, cnn2area = [], []

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]

            with tf.variable_scope('cnn1', reuse=(t > 0)):
                if self.mask_search:
                    x_search, area = process_image_with_hmap(x_curr, hmap_prev, o, mode='mask')
                cnn1out = pass_cnn(x_search, False)
                cnn1in.append(x_search)
                cnn1area.append(area)

            with tf.variable_scope('cnn2', reuse=(t > 0)):
                #xy = tf.stop_gradient(concat([x_prev, hmap_prev], axis=3))
                if not self.crop_target:
                    xy = concat([x_prev, hmap_prev], axis=3)
                else:
                    xy, area = process_image_with_hmap(x_prev, hmap_prev, o, mode='crop')
                cnn2out = pass_cnn(xy, True)
                cnn2in.append(xy)
                cnn2area.append(area)

            h1_curr = [None] * self.lstm1_nlayers
            c1_curr = [None] * self.lstm1_nlayers
            with tf.variable_scope('lstm1', reuse=(t > 0)):
                xin = tf.identity(cnn2out[-1])
                for i in range(self.lstm1_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h1_curr[i], c1_curr[i] = pass_lstm1(xin, h1_prev[i], c1_prev[i])
                        if self.residual_lstm:
                            xin = h1_curr[i] + slim.fully_connected(xin, o.nunits, scope='proj')
                        else:
                            xin = h1_curr[i]
                    if self.dropout_rnn:
                        xin = slim.dropout(xin, keep_prob=self.keep_prob,
                                           is_training=is_training, scope='dropout')

            if self.feed_examplar:
                with tf.variable_scope('combine_examplar', reuse=(t > 0)):
                    nch_xin = xin.shape.as_list()[-1]
                    nch_examplar = examplar.shape.as_list()[-1]
                    if nch_xin != nch_examplar:
                        xin = xin + slim.fully_connected(examplar, nch_xin, scope='proj')
                    else:
                        xin = xin + examplar

            with tf.variable_scope('multi_level_cross_correlation', reuse=(t > 0)):
                #scoremap = pass_multi_level_cross_correlation(cnn1out, h1_curr[-1]) # multi-layer lstm1
                scoremap = pass_multi_level_cross_correlation(cnn1out, xin)

            with tf.variable_scope('multi_level_deconvolution', reuse=(t > 0)):
                scoremap = pass_multi_level_deconvolution(scoremap)

            with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                hmap_curr_pred = pass_out_heatmap(scoremap)

            h2_curr = [None] * self.lstm2_nlayers
            c2_curr = [None] * self.lstm2_nlayers
            with tf.variable_scope('lstm2', reuse=(t > 0)):
                xin = tf.identity(scoremap)
                for i in range(self.lstm2_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h2_curr[i], c2_curr[i] = pass_lstm2(xin, h2_prev[i], c2_prev[i])
                        if self.residual_lstm:
                            xin = h2_curr[i] + slim.conv2d(xin, 2, 1, scope='proj')
                        else:
                            xin = h2_curr[i]
                    if self.dropout_rnn:
                        xin = slim.dropout(h2_curr[i], keep_prob=self.keep_prob,
                                           is_training=is_training, scope='dropout')

            with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                #y_curr_pred = pass_out_rectangle(h2_curr[-1]) # multi-layer lstm2
                y_curr_pred = pass_out_rectangle(xin)

            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            hmap_curr_gt = tf.identity(get_masks_from_rectangles(y_curr, o))
            hmap_prev = tf.cond(gt_condition, lambda: hmap_curr_gt,
                                              lambda: tf.expand_dims(tf.nn.softmax(hmap_curr_pred)[:,:,:,0], 3))

            x_prev = x_curr
            h1_prev, c1_prev = h1_curr, c1_curr
            h2_prev, c2_prev = h2_curr, c2_curr

            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr_pred)

        y_pred = tf.stack(y_pred, axis=1) # list to tensor
        hmap_pred = tf.stack(hmap_pred, axis=1)
        y_prev = y_pred[:,-1,:] # for `delta` regression type output.

        outputs = {'y': y_pred, 'hmap': hmap_pred, 'hmap_softmax': tf.nn.softmax(hmap_pred)}
        state = {}
        state.update({'h1_{}'.format(i+1): (h1_init[i], h1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'c1_{}'.format(i+1): (c1_init[i], c1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'h2_{}'.format(i+1): (h2_init[i], h2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'c2_{}'.format(i+1): (c2_init[i], c2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'x': (x_init, x_prev)})
        state.update({'hmap': (hmap_init, hmap_prev)})
        state.update({'y': (y_init, y_prev)})

        #dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        #dbg = {}
        dbg = {'cnn1in': tf.stack(cnn1in, axis=1),
               'cnn2in': tf.stack(cnn2in, axis=1),
               'cnn1area': tf.stack(cnn1area, axis=1),
               'cnn2area': tf.stack(cnn2area, axis=1)}
        return outputs, state, dbg


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


def rnn_multi_res(example, o,
                  summaries_collections=None,
                  # Model options:
                  kind='vgg',
                  use_heatmap=False,
                  **model_params):

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

    # net_fn(x, None, init=True, ...) returns None, h_init.
    # net_fn(x, h_prev, init=False, ...) returns y, h.
    if kind == 'vgg':
        net_fn = multi_res_vgg
    elif kind == 'resnet':
        raise Exception('not implemented')
    else:
        raise ValueError('unknown net type: {}'.format(kind))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            # TODO: Share some layers?
            with tf.variable_scope('rnn_init'):
                _, s_init = net_fn(init_input, None, init=True,
                    use_heatmap=use_heatmap, heatmap_stride=o.heatmap_stride,
                    **model_params)

            y, heatmap = [], []
            s_prev = s_init
            for t in range(o.ntimesteps):
                with tf.name_scope('t{}'.format(t)):
                    with tf.variable_scope('frame', reuse=(t > 0)):
                        outputs_t, s = net_fn(images[:, t], s_prev, init=False,
                            use_heatmap=use_heatmap, heatmap_stride=o.heatmap_stride,
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
    model.image_size   = (o.frmsz, o.frmsz)
    model.sequence_len = o.ntimesteps # Static length of unrolled RNN.
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


def load_model(o, model_params=None):
    '''
    example is a dictionary that maps strings to Tensors.
    Its keys should include 'inputs', 'labels', 'x0', 'y0'.
    '''
    model_params = model_params or {}
    assert('summaries_collections' not in model_params)
    if o.model == 'RNN_dual_mix':
        model = functools.partial(RNN_dual_mix, o=o, **model_params)
    elif o.model == 'Nornn':
        model = functools.partial(Nornn, o=o, **model_params)
    elif o.model == 'RNN_conv_asymm':
        model = functools.partial(rnn_conv_asymm, o=o, **model_params)
    elif o.model == 'RNN_multi_res':
        model = functools.partial(rnn_multi_res, o=o, **model_params)
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

