import pdb
import tensorflow as tf
from tensorflow.contrib import slim
import os

import numpy as np

import cnnutil

# Model interface:
#
# Inputs:
# inputs_raw -- Images for frames {t .. t+T}.
# inputs_valid -- Boolean for frames {t .. t+T}.
# inputs_HW -- Dimension of video.
# labels -- Location of object in frames {t .. t+T}.
# x0_raw -- Image for frame 0 (perhaps t != 0).
# y0 -- Location of object in frame 0.
# target_raw -- Image of object in frame 0, centered and masked.
#
# A model may also define a collection of state variables.
# The state variables will be fed only when the segment is being continued
# from a previous segment.

def make_input_placeholders(o, stat=None):
    '''
    Feed images to 'inputs_raw', 'x0_raw', etc.
    Compute functions of 'inputs', 'x0', etc.
    '''

    # placeholders for inputs
    inputs_raw = tf.placeholder(o.dtype,
            shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel],
            name='inputs_raw')
    inputs_valid = tf.placeholder(tf.bool,
            shape=[o.batchsz, o.ntimesteps+1],
            name='inputs_valid')
    inputs_HW = tf.placeholder(o.dtype,
            shape=[o.batchsz, 2],
            name='inputs_HW')
    labels = tf.placeholder(o.dtype,
            shape=[o.batchsz, o.ntimesteps+1, o.outdim],
            name='labels')

    # placeholders for initializations of full-length sequences
    # y_init = tf.placeholder_with_default(
    #         labels[:,0],
    #         shape=[o.batchsz, o.outdim], name='y_init')
    y_init = tf.placeholder(o.dtype, shape=[o.batchsz, o.outdim], name='y_init')

    # placeholders for x0 and y0. This is used for full-length sequences
    # NOTE: when it's not first segment, you should always feed x0, y0
    # otherwise it will use GT as default. It will be wrong then.
    # x0_raw = tf.placeholder_with_default(
    #         inputs_raw[:,0],
    #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], name='x0_raw')
    # y0 = tf.placeholder_with_default(o.dtype
    #         labels[:,0],
    #         shape=[o.batchsz, o.outdim], name='y0')
    x0_raw = tf.placeholder(o.dtype,
            shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel],
            name='x0_raw')
    y0 = tf.placeholder(o.dtype, shape=[o.batchsz, o.outdim], name='y0')
    # NOTE: when it's not first segment, be careful what is passed to target.
    # Make sure to pass x0, not the first frame of every segments.
    target_raw = tf.placeholder(o.dtype,
            shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel],
            name='target_raw')

    assert(stat is not None)
    with tf.name_scope('image_stats') as scope:
        if stat:
            mean = tf.constant(stat['mean'], o.dtype, name='mean')
            std  = tf.constant(stat['std'],  o.dtype, name='std')
        else:
            mean = tf.constant(0.0, o.dtype, name='mean')
            std  = tf.constant(1.0, o.dtype, name='std')
    inputs = whiten(inputs_raw, mean, std, name='inputs')
    x0     = whiten(x0_raw,     mean, std, name='x0')
    target = whiten(target_raw, mean, std, name='target')

    return {
            'inputs_raw':   inputs_raw,
            'inputs':       inputs,
            'inputs_valid': inputs_valid,
            'inputs_HW':    inputs_HW,
            'labels':       labels,
           #'outputs':      outputs,
           #'loss':         loss_total,
            'y_init':       y_init,
           #'y_last':       y_curr,
            'x0_raw':       x0_raw,
            'x0':           x0,
            'y0':           y0,
            'target_raw':   target_raw,
            'target':       target,
            }

def whiten(x, mean, std, name='whiten'):
    with tf.name_scope(name) as scope:
        return tf.divide(x - mean, std, name=scope)

class RNN_attention_s(object):
    def __init__(self, o, is_train):
        self._is_train = is_train
        self.net = None
        self.cnnout = {}
        self.params = {}
        self.net = self._create_network(o)

    def _cnnout_update(self, cnn, inputs):
        self.cnnout['feat'] = cnn.create_network(inputs)
        self.cnnout['shape'] = self.cnnout['feat'].get_shape().as_list()
        self.cnnout['h'] = self.cnnout['shape'][2] #TODO: double check w and h
        self.cnnout['w'] = self.cnnout['shape'][3]
        self.cnnout['c'] = self.cnnout['shape'][4]
        self.featdim = self.cnnout['h']*self.cnnout['w']*self.cnnout['c']
        assert(len(self.cnnout['shape'])==5)
        assert(self.cnnout['h']==self.cnnout['w']) # NOTE: be careful if change

    def _create_network(self, o):
        inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel]
        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(tf.int32, shape=[o.batchsz])
        inputs_HW = tf.placeholder(o.dtype, shape=[o.batchsz, 2])
        labels = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, o.outdim])

        # CNN 
        if o.cnn_pretrain: # use pre-trained model
            raise ValueError('not implemented yet') # TODO: try vgg
        else: # train from scratch
            cnn = CNN(o)
            self._cnnout_update(cnn, inputs)

        # RNN
        outputs = self._rnn_pass(labels, o)
 
        loss = get_loss(outputs, labels, inputs_length, inputs_HW, o, 'rectangle')
        tf.add_to_collection('losses', loss)
        loss_total = tf.reduce_sum(tf.get_collection('losses'), name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
                'inputs_HW': inputs_HW,
                'labels': labels,
                'outputs': outputs,
                'loss': loss_total}
        return net

    def _rnn_pass(self, labels, o):

        def _get_lstm_params(o):
            shape_W = [o.nunits+self.featdim, o.nunits*4] 
            shape_b = [o.nunits*4]
            with tf.variable_scope('LSTMcell'):
                self.params['W_lstm'] = tf.get_variable(
                    'W_lstm', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer(stddev=0.001)) 
                self.params['b_lstm'] = tf.get_variable(
                    'b_lstm', shape=shape_b, dtype=o.dtype, 
                    initializer=tf.constant_initializer())

        def _get_attention_params(o):
            annotationdim = self.cnnout['h']*self.cnnout['w']
            if o.yprev_mode == 'nouse':
                shape_W_mlp1 = [o.nunits+self.featdim, o.nunits] # NOTE: 1st arg can be diff size
            elif o.yprev_mode == 'concat_abs':
                shape_W_mlp1 = [o.nunits+self.featdim+o.outdim, o.nunits] # NOTE: 1st arg can be diff size
            elif o.yprev_mode == 'weight':
                shape_W_mlp1 = [o.nunits+self.featdim, o.nunits] # NOTE: 1st arg can be diff size
            else:
                raise ValueError('not implemented yet')
            shape_b_mlp1 = [o.nunits] # NOTE: 1st arg can be different mlp size
            shape_W_mlp2 = [o.nunits, annotationdim] # NOTE: 1st arg can be different mlp size 
            shape_b_mlp2 = [annotationdim]
            with tf.variable_scope('mlp'):
                self.params['W_mlp1'] = tf.get_variable(
                    'W_mlp1', shape=shape_W_mlp1, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp1'] = tf.get_variable(
                    'b_mlp1', shape=shape_b_mlp1, dtype=o.dtype,
                    initializer=tf.constant_initializer())
                self.params['W_mlp2'] = tf.get_variable(
                    'W_mlp2', shape=shape_W_mlp2, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp2'] = tf.get_variable(
                    'b_mlp2', shape=shape_b_mlp2, dtype=o.dtype,
                    initializer=tf.constant_initializer())

        def _get_rnnout_params(o):
            with tf.variable_scope('rnnout'):
                self.params['W_out'] = tf.get_variable(
                    'W_out', shape=[o.nunits, o.outdim], dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer()) # NOTE: stddev
                self.params['b_out'] = tf.get_variable(
                    'b_out', shape=[o.outdim], dtype=o.dtype, 
                    initializer=tf.constant_initializer())

        def _activate_rnncell(x_curr, h_prev, C_prev, y_prev, o):
            W_lstm = self.params['W_lstm']
            b_lstm = self.params['b_lstm']
            W_mlp1 = self.params['W_mlp1']
            b_mlp1 = self.params['b_mlp1']
            W_mlp2 = self.params['W_mlp2']
            b_mlp2 = self.params['b_mlp2']

            # mlp network # TODO: try diff nlayers, mlp size, activation
            if o.yprev_mode == 'nouse':
                input_to_mlp = tf.concat_v2(
                    (tf.reshape(x_curr, [o.batchsz, -1]), h_prev), 1) 
            elif o.yprev_mode == 'concat_abs':
                input_to_mlp = tf.concat_v2(
                    (tf.reshape(x_curr, [o.batchsz, -1]), h_prev, y_prev), 1)
            elif o.yprev_mode == 'weight':
                # Put higher weights on the ROI of x_curr based on y_prev.
                # Note that the sum of x_curr is preserved after reweighting.
                # (eq) beta*S_all + beta*gamma*S_roi = S_all 
                # <-> gamma = ((1-beta)/beta) * (S_all/S_roi)
                # beta and gamma: decreasing and increasing factor respectively.
                # TODO: optimum beta?
                s_all = tf.constant(o.frmsz ** 2, dtype=o.dtype) 
                s_roi = (y_prev[:,2]-y_prev[:,0]) * (y_prev[:,3]-y_prev[:,1])
                tf.assert_positive(s_roi)
                tf.assert_greater_equal(s_all, s_roi)
                beta = 0.95 # TODO: make it optionable
                gamma = ((1-beta)/beta) * s_all / s_roi
                y_prev_scale = self.cnnout['w'] / o.frmsz

                x_curr_weighted = []
                for b in range(o.batchsz):
                    x_start = y_prev[b,0] * y_prev_scale
                    x_end   = y_prev[b,2] * y_prev_scale
                    y_start = y_prev[b,1] * y_prev_scale
                    y_end   = y_prev[b,3] * y_prev_scale
                    grid_x, grid_y = tf.meshgrid(
                        tf.range(x_start, x_end),
                        tf.range(y_start, y_end))
                    grid_x_flat = tf.reshape(grid_x, [-1])
                    grid_y_flat = tf.reshape(grid_y, [-1])
                    # NOTE: Can't check if indices is empty or not.. problem?
                    #tf.assert_positive(grid_x_flat.get_shape().num_elements())
                    #tf.assert_positive(grid_y_flat.get_shape().num_elements())
                    grids = tf.stack([grid_y_flat, grid_x_flat]) # NOTE: x,y order
                    grids = tf.reshape(tf.transpose(grids), [-1, 2])
                    grids = tf.cast(grids, tf.int32)
                    initial = tf.constant(
                        beta, shape=[self.cnnout['h'], self.cnnout['w']])
                    mask_beta = tf.Variable(initial)
                    mask_subset = tf.gather_nd(mask_beta, grids)
                    mask_subset_reweight = mask_subset*gamma[b]
                    mask_gamma = tf.scatter_nd_update(
                        mask_beta, grids, mask_subset_reweight)
                    x_curr_weighted.append(
                        x_curr[b] * tf.expand_dims(mask_gamma, 2))
                x_curr_weighted = tf.stack(x_curr_weighted, axis=0)
                input_to_mlp = tf.concat_v2(
                    (tf.reshape(x_curr_weighted, [o.batchsz, -1]), h_prev), 1)

                ''' DEPRECATED
                x_shape = x_curr.get_shape().as_list() 
                w_mask = tf.truncated_normal(x_shape) # default mean, stddev

                # NOTE: y_prev
                # - flooring might affect the box to be slightly leftwards
                # - this can have large effect if scaling a lot.
                assert(x_shape[1] == x_shape[2])
                #y_prev = tf.cast(tf.floor(y_prev * (x_shape[1] / o.frmsz)), 
                        #dtype=tf.int32)
                y_prev = tf.floor(y_prev * (x_shape[1] / o.frmsz))

                # create w_mask (using update_roi op)
                update_roi_libpath = os.path.join(o.path_customlib, 'update_roi.so')
                update_roi_module = tf.load_op_library(update_roi_libpath)
                w_mask_updated = update_roi_module.update_roi(
                        w_mask, y_prev, fillval=1.0) # TODO: higher fillval

                # weighting/masking/filtering of input feature 
                x_curr = x_curr * w_mask
                x_curr = tf.reshape(x_curr, [o.batchsz, -1])
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr), 1), W_lstm) + b_lstm, 4, 1)
                '''
            else:
                raise ValueError('Unavailable yprev_mode.')
            mlp1 = tf.tanh(tf.matmul(input_to_mlp, W_mlp1) + b_mlp1)
            e = tf.tanh(tf.matmul(mlp1, W_mlp2) + b_mlp2) # TODO: try diff act.

            # attention weight alpha
            #e_exp = tf.exp(e)
            #alpha = e_exp/tf.expand_dims(tf.reduce_sum(e_exp,axis=1),1)
            alpha = tf.nn.softmax(e)

            # compute (expected) attention overlayed context vector z
            z = tf.reshape(alpha, [o.batchsz, self.cnnout['h'], self.cnnout['w'], 1]) * x_curr
            z = tf.reshape(z, [o.batchsz, -1])

            # LSTM (standard; no peep hole or coupled input/forget gate version)
            f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                tf.matmul(tf.concat_v2((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

            if o.lstmforgetbias:
                C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            else:
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            return h_curr, C_curr

        # get params (NOTE: variables shared across timestep)
        _get_lstm_params(o)
        _get_attention_params(o)
        _get_rnnout_params(o)

        # initial states 
        #TODO: this changes whether t<n or t>=n
        h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
        C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
        y_prev = labels[:,0]

        # rnn unroll
        outputs = []
        for t in range(o.ntimesteps): # TODO: change if variable length input/out
            h_prev, C_prev = _activate_rnncell(
                self.cnnout['feat'][:,t], h_prev, C_prev, y_prev, o) 
            # TODO: Need to pass GT labels during training like teacher forcing
            if t == 0: # NOTE: pass ground-truth at t=2 (if t<n)
                y_prev = labels[:,0]
            else:
                y_prev = tf.matmul(h_prev, self.params['W_out']) \
                    + self.params['b_out']
            outputs.append(y_prev)
            
        # list to tensor
        outputs = tf.stack(outputs, axis=1)
        return outputs 


#------------------------------------------------------------------------------
# NOTE: rnn_attention_st
# - concatenation was used instead of dot product which measures similarity.
# - y_prev is used only implicitly through hidden states (not used explicitly). 
#------------------------------------------------------------------------------
class RNN_attention_st(object):
    def __init__(self, o, is_train):
        self._is_train = is_train
        self.net = None
        self.cnnout = {}
        self.params = {}
        self.net = self._create_network(o)

    def _cnnout_update(self, cnn, inputs):
        self.cnnout['feat'] = cnn.create_network(inputs)
        self.cnnout['shape'] = self.cnnout['feat'].get_shape().as_list()
        self.cnnout['h'] = self.cnnout['shape'][2] #TODO: double check w and h
        self.cnnout['w'] = self.cnnout['shape'][3]
        self.cnnout['c'] = self.cnnout['shape'][4]
        self.featdim = self.cnnout['h']*self.cnnout['w']*self.cnnout['c']
        assert(len(self.cnnout['shape'])==5)
        assert(self.cnnout['h']==self.cnnout['w']) # NOTE: be careful if change

    def _create_network(self, o):
        inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel]
        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(tf.int32, shape=[o.batchsz])
        inputs_HW = tf.placeholder(o.dtype, shape=[o.batchsz, 2])
        labels = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, o.outdim])

        # CNN 
        if o.cnn_pretrain: # use pre-trained model
            raise ValueError('not implemented yet') # TODO: try vgg
        else: # train from scratch
            cnn = CNN(o)
            self._cnnout_update(cnn, inputs)

        # RNN
        outputs = self._rnn_pass(labels, o)

        loss = get_loss(outputs, labels, inputs_length, inputs_HW, o, 'rectangle')
        tf.add_to_collection('losses', loss)
        loss_total = tf.reduce_sum(tf.get_collection('losses'), name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
                'inputs_HW': inputs_HW,
                'labels': labels,
                'outputs': outputs,
                'loss': loss_total}
        return net

    def _rnn_pass(self, labels, o):

        def _get_lstm_params(o):
            shape_W_s = [o.nunits+self.featdim, o.nunits*4] 
            shape_b_s = [o.nunits*4]
            with tf.variable_scope('LSTMcell_s'):
                self.params['W_lstm_s'] = tf.get_variable(
                    'W_lstm_s', shape=shape_W_s, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer(stddev=0.001)) 
                self.params['b_lstm_s'] = tf.get_variable(
                    'b_lstm_s', shape=shape_b_s, dtype=o.dtype, 
                    initializer=tf.constant_initializer())
            #shape_W_t = [o.nunits+(o.nunits*o.h_concat_ratio*o.ntimesteps), 
                #o.nunits*4] 
            shape_W_t = [o.nunits+o.nunits, o.nunits*4] # because sum will be used.
            shape_b_t = [o.nunits*4]
            with tf.variable_scope('LSTMcell_t'):
                self.params['W_lstm_t'] = tf.get_variable(
                    'W_lstm_t', shape=shape_W_t, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer(stddev=0.001)) 
                self.params['b_lstm_t'] = tf.get_variable(
                    'b_lstm_t', shape=shape_b_t, dtype=o.dtype, 
                    initializer=tf.constant_initializer())

        def _get_attention_params(o):
            # NOTE: For rnn_attention_st, we don't use y_prev explicitly, ie.,
            # it's always "nouse" because it's impossible to use y_prev.  
            # However, it's actually using it via h_prev so shouldn't be bad.

            # 1. attention_s
            annotationdim_s = self.cnnout['h']*self.cnnout['w']
            shape_W_mlp1_s = [o.nunits+self.featdim, o.nunits] # NOTE: 1st arg can be diff size
            shape_b_mlp1_s = [o.nunits] # NOTE: 1st arg can be different mlp size
            shape_W_mlp2_s = [o.nunits, annotationdim_s] # NOTE: 1st arg can be different mlp size 
            shape_b_mlp2_s = [annotationdim_s]
            with tf.variable_scope('mlp_s'):
                self.params['W_mlp1_s'] = tf.get_variable(
                    'W_mlp1_s', shape=shape_W_mlp1_s, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp1_s'] = tf.get_variable(
                    'b_mlp1_s', shape=shape_b_mlp1_s, dtype=o.dtype,
                    initializer=tf.constant_initializer())
                self.params['W_mlp2_s'] = tf.get_variable(
                    'W_mlp2_s', shape=shape_W_mlp2_s, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp2_s'] = tf.get_variable(
                    'b_mlp2_s', shape=shape_b_mlp2_s, dtype=o.dtype,
                    initializer=tf.constant_initializer())

            # 2. attention_t (Note that y_prev is only used in the bottom RNN)
            attentiondim_t = o.ntimesteps
            shape_W_mlp1_t = [o.nunits+
                (o.nunits*o.h_concat_ratio*attentiondim_t), o.nunits] 
            shape_b_mlp1_t = [o.nunits]
            shape_W_mlp2_t = [o.nunits, attentiondim_t] 
            shape_b_mlp2_t = [attentiondim_t]
            with tf.variable_scope('mlp_t'):
                self.params['W_mlp1_t'] = tf.get_variable(
                    'W_mlp1_t', shape=shape_W_mlp1_t, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp1_t'] = tf.get_variable(
                    'b_mlp1_t', shape=shape_b_mlp1_t, dtype=o.dtype,
                    initializer=tf.constant_initializer())
                self.params['W_mlp2_t'] = tf.get_variable(
                    'W_mlp2_t', shape=shape_W_mlp2_t, dtype=o.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                self.params['b_mlp2_t'] = tf.get_variable(
                    'b_mlp2_t', shape=shape_b_mlp2_t, dtype=o.dtype,
                    initializer=tf.constant_initializer())
        
        def _get_rnnout_params(o):
            with tf.variable_scope('rnnout'):
                self.params['W_out'] = tf.get_variable(
                    'W_out', shape=[o.nunits, o.outdim], dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer()) # NOTE: stddev
                self.params['b_out'] = tf.get_variable(
                    'b_out', shape=[o.outdim], dtype=o.dtype, 
                    initializer=tf.constant_initializer())

        def _activate_rnncell_s(x_curr, h_prev, C_prev, o):
            W_lstm = self.params['W_lstm_s']
            b_lstm = self.params['b_lstm_s']
            W_mlp1 = self.params['W_mlp1_s']
            b_mlp1 = self.params['b_mlp1_s']
            W_mlp2 = self.params['W_mlp2_s']
            b_mlp2 = self.params['b_mlp2_s']

            # NOTE: I am not doing similarity based attention here. 
            # Instead, I am doing mlp + softmax
            # However, even show,attend,tell paper didn't do cosine similarity.
            # They combined hidden and features merely by addition.
            # They used relu instead of tanh.
            # Also consider using batch norm.
            input_to_mlp = tf.concat_v2(
                (tf.reshape(x_curr, [o.batchsz, -1]), h_prev), 1) 

            mlp1 = tf.tanh(tf.matmul(input_to_mlp, W_mlp1) + b_mlp1)
            e = tf.tanh(tf.matmul(mlp1, W_mlp2) + b_mlp2) # TODO: try diff act.

            # attention weight alpha
            #e_exp = tf.exp(e)
            #alpha = e_exp/tf.expand_dims(tf.reduce_sum(e_exp,axis=1),1)
            alpha = tf.nn.softmax(e)

            # compute (expected) attention overlayed context vector z
            z = tf.reshape(alpha, [o.batchsz, self.cnnout['h'], self.cnnout['w'], 1]) * x_curr
            z = tf.reshape(z, [o.batchsz, -1])

            # LSTM (standard; no peep hole or coupled input/forget gate version)
            f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                tf.matmul(tf.concat_v2((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

            if o.lstmforgetbias:
                C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            else:
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            return h_curr, C_curr

        def _activate_rnncell_t(x_curr, h_prev, C_prev, t, o):
            W_lstm = self.params['W_lstm_t']
            b_lstm = self.params['b_lstm_t']

            # NOTE: from the original attention paper by Cho or Chris Olah's 
            # blog, it says that they measure the similarity between h_{t-1} 
            # and others using dot product. I am instead doing concatenation.

            # method1: using MLP same as attention_s
            '''
            W_mlp1 = self.params['W_mlp1_t']
            b_mlp1 = self.params['b_mlp1_t']
            W_mlp2 = self.params['W_mlp2_t']
            b_mlp2 = self.params['b_mlp2_t']
            input_to_mlp = tf.concat_v2(
                (tf.reshape(x_curr, [o.batchsz, -1]), h_prev), 1) 

            mlp1 = tf.tanh(tf.matmul(input_to_mlp, W_mlp1) + b_mlp1)
            e = tf.tanh(tf.matmul(mlp1, W_mlp2) + b_mlp2) # TODO: try diff act.
            '''
            # method2: using cosine similarity like standard attention
            # e: 'similarities'
            # NOTE: this is way slower than method1
            #x_curr = x_curr[:, 0:t+1, :]
            # attention range should be up until current time step
            h_prev_norm = tf.sqrt(tf.reduce_sum(tf.pow(h_prev, 2), 1))
            x_curr_norms = tf.sqrt(tf.reduce_sum(tf.pow(x_curr, 2), 2))
            h_x_dotprod = tf.reduce_sum(
                tf.expand_dims(h_prev, axis=1) * x_curr, axis=2)
            # TODO: CHECK tf.div or tf.divide
            #e = tf.div(h_x_dotprod, 
                #tf.expand_dims(h_prev_norm, axis=1) * x_curr_norms + 1e-4)
            e = h_x_dotprod / (tf.expand_dims(h_prev_norm, axis=1) * 
                    x_curr_norms + 1e-4)

            # attention weight alpha
            #e_exp = tf.exp(e)
            #alpha = e_exp/tf.expand_dims(tf.reduce_sum(e_exp,axis=1),1)
            alpha = tf.nn.softmax(e)

            # compute (expected) attention overlayed context vector z
            z = tf.expand_dims(alpha, 2) * x_curr
            z = tf.reduce_sum(z, axis=1)  
            #z = tf.reshape(z, [o.batchsz, -1])
            # NOTE: used summation instead of reshape to make the dimension 
            # consistent over different time step (same as S.A.T implentation).
            # I am not sure if sum is really a good thing to do though. Also,
            # the training progress seems worse.

            # LSTM (standard; no peep hole or coupled input/forget gate version)
            f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                tf.matmul(tf.concat_v2((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

            if o.lstmforgetbias:
                C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            else:
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            return h_curr, C_curr

        # get params (NOTE: variables shared across timestep)
        _get_lstm_params(o)
        _get_attention_params(o)
        _get_rnnout_params(o)

        # initial states 
        #TODO: this changes whether t<n or t>=n
        h_prev_s = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
        C_prev_s = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
        h_prev_t = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
        C_prev_t = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init

        # rnn_s unroll
        hs = []
        for ts in range(o.ntimesteps): # TODO: change if variable length input/out
            h_prev_s, C_prev_s = _activate_rnncell_s(
                self.cnnout['feat'][:,ts], h_prev_s, C_prev_s, o)
            # NOTE: can reduce the size of h_prev_s using h_concat_ratio
            hs.append(h_prev_s)
        hs = tf.stack(hs, axis=1) # this becomes x_curr for rnn_t

        # rnn_t unroll
        outputs = []
        for tt in range(o.ntimesteps): # TODO: change if variable length input/out
            h_prev_t, C_prev_t = _activate_rnncell_t(
                hs, h_prev_t, C_prev_t, tt, o)
            y_prev = tf.matmul(h_prev_t, self.params['W_out']) \
                + self.params['b_out']
            outputs.append(y_prev)

        # list to tensor
        outputs = tf.stack(outputs, axis=1)
        return outputs 


def get_loss(outputs, labels, inputs_valid, inputs_HW, o, outtype, name='loss'):
    with tf.name_scope(name) as scope:
        # NOTE: Be careful about length of labels and outputs. 
        # labels and inputs_valid will be of T+1 length, and y0 shouldn't be used.
        assert(outputs.get_shape().as_list()[1] == o.ntimesteps)
        assert(labels.get_shape().as_list()[1] == o.ntimesteps+1)

        losses = dict()
        
        if outtype == 'rectangle':
            # loss1: sum of two l1 distances for left-top and right-bottom
            if 'l1' in o.losses: # TODO: double check
                labels_valid = tf.boolean_mask(labels[:,1:], inputs_valid[:,1:])
                outputs_valid = tf.boolean_mask(outputs, inputs_valid[:,1:])
                loss_l1 = tf.reduce_mean(tf.abs(labels_valid - outputs_valid))
                losses['l1'] = loss_l1

            # loss2: IoU
            if 'iou' in o.losses:
                assert(False) # TODO: change from inputs_length to inputs_valid
                scalar = tf.stack((inputs_HW[:,1], inputs_HW[:,0], 
                    inputs_HW[:,1], inputs_HW[:,0]), axis=1)
                boxA = outputs * tf.expand_dims(scalar, 1)
                boxB = labels[:,1:,:] * tf.expand_dims(scalar, 1)
                xA = tf.maximum(boxA[:,:,0], boxB[:,:,0])
                yA = tf.maximum(boxA[:,:,1], boxB[:,:,1])
                xB = tf.minimum(boxA[:,:,2], boxB[:,:,2])
                yB = tf.minimum(boxA[:,:,3], boxB[:,:,3])
                interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)
                boxAArea = (boxA[:,:,2] - boxA[:,:,0]) * (boxA[:,:,3] - boxA[:,:,1]) 
                boxBArea = (boxB[:,:,2] - boxB[:,:,0]) * (boxB[:,:,3] - boxB[:,:,1]) 
                # TODO: CHECK tf.div or tf.divide
                #iou = tf.div(interArea, (boxAArea + boxBArea - interArea) + 1e-4)
                iou = interArea / (boxAArea + boxBArea - interArea + 1e-4) 
                iou_valid = []
                for i in range(o.batchsz):
                    iou_valid.append(iou[i, :inputs_length[i]-1])
                iou_mean = tf.reduce_mean(iou_valid)
                loss_iou = 1 - iou_mean # NOTE: Any normalization?
                losses['iou'] = loss_iou

        elif outtype == 'heatmap':
            # First of all, need to convert labels into heat maps
            labels_heatmap = convert_rec_to_heatmap(labels, o)

            # valid labels and outputs
            labels_valid = tf.boolean_mask(labels_heatmap[:,1:], inputs_valid[:,1:])
            outputs_valid = tf.boolean_mask(outputs, inputs_valid[:,1:])

            # loss1: cross-entropy between probabilty maps (need to change label) 
            if 'ce' in o.losses: 
                labels_flat = tf.reshape(labels_valid, [-1, o.frmsz**2])
                outputs_flat = tf.reshape(outputs_valid, [-1, o.frmsz**2])
                assert_finite = lambda x: tf.Assert(tf.reduce_all(tf.is_finite(x)), [x])
                with tf.control_dependencies([assert_finite(outputs_valid)]):
                    outputs_valid = tf.identity(outputs_valid)
                loss_ce = tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat, logits=outputs_flat)
                # Wrap with assertion that loss is finite.
                with tf.control_dependencies([assert_finite(loss_ce)]):
                    loss_ce = tf.identity(loss_ce)
                loss_ce = tf.reduce_mean(loss_ce)
                losses['ce'] = loss_ce

            # loss2: tf's l2 (without sqrt)
            if 'l2' in o.losses:
                labels_flat = tf.reshape(labels_valid, [-1, o.frmsz**2])
                outputs_flat = tf.reshape(outputs_valid, [-1, o.frmsz**2])
                outputs_softmax = tf.nn.softmax(outputs_flat)
                loss_l2 = tf.nn.l2_loss(labels_flat - outputs_softmax)
                losses['l2'] = loss_l2

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss)

        return tf.reduce_sum(losses.values(), name=scope)

def convert_rec_to_heatmap(rec, o):
    '''Create heatmap from rectangle
    Args:
        rec: [batchsz x ntimesteps+1] ground-truth rectangle labels
    Return:
        heatmap: [batchsz x ntimesteps+1 x o.frmsz x o.frmsz x 1]
    '''
    masks = []
    for t in range(o.ntimesteps+1):
        masks.append(get_masks_from_rectangles(rec[:,t], o))
    masks = tf.stack(masks, axis=1)
    # normalize
    masks_sum = tf.reshape(tf.reduce_sum(masks, axis=[2,3,4]),
            [o.batchsz, o.ntimesteps+1, 1, 1, 1])
    heatmap = tf.divide(masks, masks_sum)
    return heatmap

# weight variable; created variables are new and will not be shared.
def get_weight_variable(
        shape_, mean_=0.0, stddev_=0.1, trainable_=True, name_=None, wd=0.0):
    initial = tf.truncated_normal(shape=shape_, mean=mean_, stddev=stddev_)
    var = tf.Variable(initial_value=initial, trainable=trainable_, name=name_)
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
    return var

# bias variable; created variables are new and will not be shared.
def get_bias_variable(shape_, value=0.1, trainable_=True, name_=None, wd=0.0):
    initial = tf.constant(value, shape=shape_)
    var = tf.Variable(initial_value=initial, trainable=trainable_, name=name_)
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
    return var

def get_variable_with_initial(initial_, trainable_=True, name_=None, wd=0.0):
    var = tf.Variable(initial_value=initial_, trainable=trainable_, name=name_) 
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(x, w, b, stride=1, padding='SAME'):
    x = tf.nn.conv2d(x, w, strides=(1, stride, stride, 1), padding=padding)
    x = tf.nn.bias_add(x, b)
    return x

def max_pool(x, support=3, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=(1, support, support, 1),
                             strides=(1, stride, stride, 1),
                             padding=padding)

def activate(input_, activation_='relu'):
    if activation_ == 'relu':
        return tf.nn.relu(input_)
    elif activation_ == 'tanh':
        return tf.nn.tanh(input_)
    elif activation_ == 'sigmoid':
        return tf.nn.sigmoid(input_)
    elif activation_ == 'linear': # no activation!
        return input_
    else:
        raise ValueError('no available activation type!')

def get_masks_from_rectangles(rec, o):
    # create mask using rec; typically rec=y_prev
    x1 = rec[:,0] * o.frmsz
    y1 = rec[:,1] * o.frmsz
    x2 = rec[:,2] * o.frmsz
    y2 = rec[:,3] * o.frmsz
    x1, y1, x2, y2 = enforce_min_size(x1, y1, x2, y2, 1.0)
    grid_x, grid_y = tf.meshgrid(
            tf.range(o.frmsz, dtype=o.dtype),
            tf.range(o.frmsz, dtype=o.dtype))
    # resize tensors so that they can be compared
    x1 = tf.expand_dims(tf.expand_dims(x1,1),2)
    x2 = tf.expand_dims(tf.expand_dims(x2,1),2)
    y1 = tf.expand_dims(tf.expand_dims(y1,1),2)
    y2 = tf.expand_dims(tf.expand_dims(y2,1),2)
    grid_x = tf.tile(tf.expand_dims(grid_x,0), [o.batchsz,1,1])
    grid_y = tf.tile(tf.expand_dims(grid_y,0), [o.batchsz,1,1])
    # mask
    masks = tf.logical_and(
        tf.logical_and(tf.less_equal(x1, grid_x), 
            tf.less_equal(grid_x, x2)),
        tf.logical_and(tf.less_equal(y1, grid_y), 
            tf.less_equal(grid_y, y2)))
    # type and dim change so that it can be concated with x
    masks = tf.expand_dims(tf.cast(masks, o.dtype),3)
    return masks

def enforce_min_size(x1, y1, x2, y2, min=1.0):
    # Ensure that x2-x1 > 1
    xc, xs = 0.5*(x1 + x2), x2-x1
    yc, ys = 0.5*(y1 + y2), y2-y1
    xs = tf.maximum(min, xs)
    ys = tf.maximum(min, ys)
    x1, x2 = xc-xs/2, xc+xs/2
    y1, y2 = yc-ys/2, yc+ys/2
    return x1, y1, x2, y2


class RNN_basic(object):
    def __init__(self, o, stat=None):
        self.is_train = True if o.mode == 'train' else False

        self.params = {}
        self._update_params_cnn(o)
        self.net = self._load_model(o, stat=stat) 

    def _update_params_cnn(self, o):
        # non-learnable params; dataset dependent
        if o.dataset in ['moving_mnist', 'bouncing_mnist']: # easy datasets
            self.nlayers = 2
            self.nchannels = [16, 16]
            self.filtsz = [3, 3]
            self.strides = [3, 3]
        else: # ILSVRC or other data set of natural images 
            # TODO: potentially a lot of room for improvement here
            self.nlayers = 3
            self.nchannels = [16, 32, 64]
            self.filtsz = [7, 5, 3]
            self.strides = [3, 2, 1]
        assert(self.nlayers == len(self.nchannels))
        assert(self.nlayers == len(self.filtsz))
        assert(self.nlayers == len(self.strides))

        # learnable parameters; CNN params shared across all time steps
        w_conv = []
        b_conv = []
        for i in range(self.nlayers):
            with tf.name_scope('layer_{}'.format(i)):
                if not o.pass_yinit:
                    # two input images + 1 mask channel
                    shape_w = [self.filtsz[i], self.filtsz[i], 
                            o.ninchannel*2+1 if i==0 else self.nchannels[i-1], 
                            self.nchannels[i]]
                else:
                    # three input images + 2 mask channel
                    shape_w = [self.filtsz[i], self.filtsz[i], 
                            o.ninchannel*3+2 if i==0 else self.nchannels[i-1], 
                            self.nchannels[i]]
                shape_b = [self.nchannels[i]]
                w_conv.append(get_weight_variable(shape_w,name_='w',wd=o.wd))
                b_conv.append(get_bias_variable(shape_b,name_='b',wd=o.wd))
        self.params['w_conv'] = w_conv
        self.params['b_conv'] = b_conv

    def _update_params_rnn(self, cnnout, o):
        # cnn params
        cnnout_shape = cnnout.get_shape().as_list()
        self.params['cnn_h'] = cnnout_shape[1]
        self.params['cnn_w'] = cnnout_shape[2]
        self.params['cnn_c'] = cnnout_shape[3]
        self.params['cnn_featdim'] = cnnout_shape[1]*cnnout_shape[2]*cnnout_shape[3]
        # lstm params
        with tf.variable_scope('LSTMcell'):
            shape_w = [o.nunits+self.params['cnn_featdim'], o.nunits*4] 
            shape_b = [o.nunits*4]
            self.params['w_lstm'] = tf.get_variable(
                'w_lstm', shape=shape_w, dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1)) 
            self.params['b_lstm'] = tf.get_variable(
                'b_lstm', shape=shape_b, dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))
        # rnnout params 
        with tf.variable_scope('rnnout'):
            self.params['w_out1'] = tf.get_variable(
                'w_out1', shape=[o.nunits, o.nunits/2], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.params['b_out1'] = tf.get_variable(
                'b_out1', shape=[o.nunits/2], dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))
            self.params['w_out2'] = tf.get_variable(
                'w_out2', shape=[o.nunits/2, o.outdim], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.params['b_out2'] = tf.get_variable(
                'b_out2', shape=[o.outdim], dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))

    def _pass_cnn(self, x_curr, x_prev, y_prev, o, x0=None, y0=None):
        if not o.pass_yinit:
            # create masks from y_prev
            masks = get_masks_from_rectangles(y_prev, o)

            # input concat
            cnnin = tf.concat_v2((x_prev, x_curr, masks), 3)
        else:
            masks_yprev = get_masks_from_rectangles(y_prev, o)
            masks_yinit = get_masks_from_rectangles(y0, o)
            cnnin = tf.concat_v2(
                    (x0, masks_yinit, x_prev, x_curr, masks_yprev), 3)

        # convolutions; feed-forward
        for i in range(self.nlayers):
            x = cnnin if i==0 else act
            conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], stride=self.strides[i])
            act = activate(conv, 'relu')
            if self.is_train and o.dropout_cnn and i==1: # NOTE: maybe only at 2nd layer
                act = tf.nn.dropout(act, o.keep_ratio_cnn)
        return act

    def _pass_rnn(self, cnnout, h_prev, C_prev, o):
        # directly flatten cnnout and concat with h_prev 
        # NOTE: possibly other options such as projection of cnnout before 
        # concatenating with h_prev; C. Olah -> no need. 
        # TODO: try Conv LSTM
        xy_in = tf.reshape(cnnout, [o.batchsz, -1])

        f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
            tf.concat_v2((h_prev, xy_in), 1), self.params['w_lstm']) + 
            self.params['b_lstm'], 4, 1)

        if o.lstmforgetbias:
            C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        else:
            C_curr = tf.sigmoid(f_curr) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
        return h_curr, C_curr

    def _load_model(self, o, stat=None):
        net = make_input_placeholders(o, stat)
        inputs       = net['inputs']
        inputs_valid = net['inputs_valid']
        inputs_HW    = net['inputs_HW']
        labels       = net['labels']
        y_init       = net['y_init']
        x0           = net['x0']
        y0           = net['y0']
        target       = net['target']

        # # placeholders for inputs
        # inputs = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='inputs')
        # inputs_valid = tf.placeholder(tf.bool, 
        #         shape=[o.batchsz, o.ntimesteps+1], 
        #         name='inputs_valid')
        # inputs_HW = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, 2], 
        #         name='inputs_HW')
        # labels = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.outdim], 
        #         name='labels')

        # placeholders for initializations of full-length sequences
        h_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zeor or normal
                shape=[o.batchsz, o.nunits], name='h_init')
        C_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zeor or normal
                shape=[o.batchsz, o.nunits], name='C_init')
        # y_init = tf.placeholder_with_default(
        #         labels[:,0], 
        #         shape=[o.batchsz, o.outdim], name='y_init')

        # # placeholders for x0 and y0. This is used for full-length sequences
        # # NOTE: when it's not first segment, you should always feed x0, y0
        # # otherwise it will use GT as default. It will be wrong then.
        # x0 = tf.placeholder_with_default(
        #         inputs[:,0], 
        #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], name='x0')
        # y0 = tf.placeholder_with_default(
        #         labels[:,0],
        #         shape=[o.batchsz, o.outdim], name='y0')
        # # NOTE: when it's not first segment, be careful what is passed to target.
        # # Make sure to pass x0, not the first frame of every segments.
        # target = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='target')

        # RNN unroll
        outputs = []
        rnninit = False
        for t in range(1, o.ntimesteps+1):
            if t==1:
                h_prev = h_init
                C_prev = C_init
                y_prev = y_init
            else:
                h_prev = h_curr
                C_prev = C_curr
                y_prev = y_curr
            x_prev = inputs[:,t-1]
            x_curr = inputs[:,t]

            if o.pass_yinit:
                cnnout = self._pass_cnn(x_curr, x_prev, y_prev, o, x0, y0)
            else:
                cnnout = self._pass_cnn(x_curr, x_prev, y_prev, o)

            if not rnninit: self._update_params_rnn(cnnout, o); rnninit = True
            h_curr, C_curr = self._pass_rnn(cnnout, h_prev, C_prev, o)

            h_out = tf.matmul(h_curr, self.params['w_out1']) + self.params['b_out1']
            y_curr = tf.matmul(h_out, self.params['w_out2']) + self.params['b_out2']
            outputs.append(y_curr)
        outputs = tf.stack(outputs, axis=1) # list to tensor

        loss = get_loss(outputs, labels, inputs_valid, inputs_HW, o)
        tf.add_to_collection('losses', loss)
        loss_total = tf.reduce_sum(tf.get_collection('losses'),name='loss_total')

        # net = {
        net.update({
                # 'inputs': inputs,
                # 'inputs_valid': inputs_valid,
                # 'inputs_HW': inputs_HW,
                # 'labels': labels,
                'outputs': outputs,
                'loss': loss_total,
                'h_init': h_init,
                'C_init': C_init,
                # 'y_init': y_init,
                'h_last': h_curr,
                'C_last': C_curr,
                'y_last': y_curr,
                # 'x0': x0,
                # 'y0': y0
                })
        return net


class RNN_new(object):
    def __init__(self, o, stat=None):
        self.is_train = True if o.mode == 'train' else False

        self.params = self._update_params(o)
        self.net = self._load_model(o, stat=stat) 

    def _update_params(self, o):
        # cnn params (depends kernel size and strides at each layer)
        with tf.variable_scope('cnn'):
            cnn = {}

            # TODO: potentially a lot of room for improvement here
            cnn['nlayers'] = 3
            cnn['layer'] = []
            cnn['layer'].append({'filtsz': 7, 'st': 3, 'chin': 3, 'chout': 16})
            cnn['layer'].append({'filtsz': 5, 'st': 2, 'chin': 16, 'chout': 32})
            cnn['layer'].append({'filtsz': 3, 'st': 1, 'chin': 32, 'chout': 64})

            # compute cnn output sizes
            h, w = o.frmsz, o.frmsz
            for i in range(cnn['nlayers']):
                h = int(np.ceil(h / float(cnn['layer'][i]['st'])))
                w = int(np.ceil(w / float(cnn['layer'][i]['st'])))
                cnn['layer'][i].update({'out_h': h, 'out_w': w})

            # CNN for RNN input images; shared across all time steps
            for i in range(cnn['nlayers']):
                shape_w = [cnn['layer'][i]['filtsz'], cnn['layer'][i]['filtsz'], 
                        cnn['layer'][i]['chin'], cnn['layer'][i]['chout']]
                shape_b = [cnn['layer'][i]['chout']]
                cnn['layer'][i].update({'w': get_weight_variable(
                    shape_w, name_='w{}'.format(i), wd=o.wd) })
                cnn['layer'][i].update({'b': get_bias_variable(
                    shape_b, name_='b{}'.format(i), wd=o.wd) })

        # 1x1 CNN params; used before putting cnn features to RNN cell
        with tf.variable_scope('cnn_1x1'):
            cnn_1x1 = {}
            cnn_1x1['nlayers'] = 2
            cnn_1x1['layer'] = []
            for i in range(cnn_1x1['nlayers']):
                chin = cnn['layer'][-1]['chout']/pow(2,i)
                chout = cnn['layer'][-1]['chout']/pow(2,i+1)
                cnn_1x1['layer'].append({
                    'w': get_weight_variable([1, 1, chin, chout], 
                        name_='w{}'.format(i), wd=o.wd),
                    'b': get_bias_variable([chout], 
                        name_='b{}'.format(i), wd=o.wd),
                    'chout': chout})
        # Conv lstm params
        with tf.variable_scope('lstm'):
            lstm = {}
            gates = ['i', 'f' , 'c' , 'o']
            for gate in gates:
                lstm['w_x{}'.format(gate)] = get_weight_variable(
                        [3, 3, cnn_1x1['layer'][-1]['chout'], 
                        cnn_1x1['layer'][-1]['chout']],
                        name_='w_x{}'.format(gate), wd=o.wd)
                lstm['w_h{}'.format(gate)] = get_weight_variable(
                        [3,3,cnn_1x1['layer'][-1]['chout'], 
                        cnn_1x1['layer'][-1]['chout']],
                        name_='w_h{}'.format(gate), wd=o.wd)
                lstm['b_{}'.format(gate)] = get_weight_variable(
                    [cnn_1x1['layer'][-1]['chout']], 
                    name_='b_{}'.format(gate), wd=o.wd)
                    
        # upsample filter, initialize with bilinear upsample weights
        # NOTE: #layers may need to be at least the size change occurs in cnnin
        def get_bilinear_upsample_weights(filtsz, channels):
            '''Create weights matrix for transposed convolution with 
            bilinear filter initialization.
            '''
            factor = (filtsz+ 1) / float(2)
            if filtsz % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:filtsz, :filtsz]
            upsample_kernel = (1-abs(og[0]-center)/factor) \
                    * (1-abs(og[1]-center)/factor)
            weights = np.zeros((filtsz, filtsz, channels, channels), 
                    dtype=np.float32)
            for i in xrange(channels):
                weights[:, :, i, i] = upsample_kernel
            return weights

        # NOTE: try different number of upsamplings
        with tf.variable_scope('cnn_deconv'):
            cnn_deconv = {}
            cnn_deconv['layer'] = []
            cnn_deconv['layer'].append(
                get_variable_with_initial(get_bilinear_upsample_weights(
                    cnn['layer'][1]['filtsz'], cnn_1x1['layer'][-1]['chout'])))
            cnn_deconv['layer'].append(
                get_variable_with_initial(get_bilinear_upsample_weights(
                    cnn['layer'][0]['filtsz'], cnn_1x1['layer'][-1]['chout'])))

        # regression for y; (1x1 convolutions reducing channels to 1)
        with tf.variable_scope('out'):
            out = {}
            out['w'] = get_weight_variable(
                [1, 1, cnn_1x1['layer'][-1]['chout'], 1], name_='w', wd=o.wd)
            out['b'] = get_bias_variable([1], name_='b', wd=o.wd)

        # return params
        params = {}
        params['cnn'] = cnn
        params['cnn_1x1'] = cnn_1x1
        params['lstm'] = lstm
        params['cnn_deconv'] = cnn_deconv
        params['out'] = out
        return params

    def _pass_cnn(self, cnnin, o):
        for i in range(self.params['cnn']['nlayers']):
            x = cnnin if i==0 else act
            conv = conv2d(x, 
                self.params['cnn']['layer'][i]['w'], 
                self.params['cnn']['layer'][i]['b'], 
                stride=self.params['cnn']['layer'][i]['st'])
            act = activate(conv, 'relu')
            # TODO: can add dropout here
            if self.is_train and o.dropout_cnn and i==1: # NOTE: maybe only at 2nd layer
                act = tf.nn.dropout(act, o.keep_ratio_cnn)
        return act

    def _depthwise_convolution(self, search, filt, o):
        # similar to siamese
        scoremap = []
        for i in range(o.batchsz):
            searchimg = tf.expand_dims(search[i],0)
            filtimg = tf.expand_dims(filt[i],3)
            scoremap.append(
                tf.nn.depthwise_conv2d(searchimg, filtimg, [1,1,1,1], 'SAME'))
        scoremap = tf.squeeze(tf.stack(scoremap, axis=0), squeeze_dims=1)
        return scoremap

    def _pass_cnn_1x1s(self, scoremap, o):
        for i in range(self.params['cnn_1x1']['nlayers']):
            x = scoremap if i==0 else act
            conv = conv2d(x, 
                    self.params['cnn_1x1']['layer'][i]['w'],
                    self.params['cnn_1x1']['layer'][i]['b']) 
            act = activate(conv, 'relu')
        # TODO: can add dropout here 
        return act

    def _pass_rnn(self, cellin, h_prev, c_prev, o):
        ''' ConvLSTM
        1. h and c have the same dimension as x (padding can be used)
        2. no peephole connection
        '''
        it = activate(
            conv2d(cellin, self.params['lstm']['w_xi'], self.params['lstm']['b_i']) +
            conv2d(h_prev, self.params['lstm']['w_hi'], self.params['lstm']['b_i']), 
            'sigmoid')
        ft = activate(
            conv2d(cellin, self.params['lstm']['w_xf'], self.params['lstm']['b_f']) +
            conv2d(h_prev, self.params['lstm']['w_hf'], self.params['lstm']['b_f']), 
            'sigmoid')
        ct_tilda = activate(
            conv2d(cellin, self.params['lstm']['w_xc'], self.params['lstm']['b_c']) +
            conv2d(h_prev, self.params['lstm']['w_hc'], self.params['lstm']['b_c']), 
            'tanh')
        ct = (ft * c_prev) + (it * ct_tilda)
        ot = activate(
            conv2d(cellin, self.params['lstm']['w_xo'], self.params['lstm']['b_o']) +
            conv2d(h_prev, self.params['lstm']['w_ho'], self.params['lstm']['b_o']), 
            'sigmoid')
        ht = ot * activate(ct, 'tanh')
        return ht, ct

    def _pass_deconvolution(self, h, o):
        # transposed convolution (or fractionally strided convolution). 
        # (a.k.a. deconvolution, but wrong)
        # bilinear resampling kernel is used at initialization.
        # 3 layers and activations in between.
        deconv1 = tf.nn.conv2d_transpose(
            h, self.params['cnn_deconv']['layer'][0],
            output_shape=[o.batchsz, 
                self.params['cnn']['layer'][0]['out_h'], 
                self.params['cnn']['layer'][0]['out_w'], 
                self.params['cnn_1x1']['layer'][-1]['chout']], 
            strides=[1, 
                self.params['cnn']['layer'][1]['st'], 
                self.params['cnn']['layer'][1]['st'], 1])
        act_deconv1 = activate(deconv1, 'relu')

        deconv2 = tf.nn.conv2d_transpose(
            deconv1, self.params['cnn_deconv']['layer'][1],
            output_shape=[o.batchsz, o.frmsz, o.frmsz,
                self.params['cnn_1x1']['layer'][-1]['chout']], 
            strides=[1, 
                self.params['cnn']['layer'][0]['st'], 
                self.params['cnn']['layer'][0]['st'], 1])
        act_deconv2 = activate(deconv2, 'relu')

        return act_deconv2

    def _project_output(self, h):
        output = conv2d(h, self.params['out']['w'], self.params['out']['b'])
        return output

    def _load_model(self, o, stat=None):
        net = make_input_placeholders(o, stat)
        inputs       = net['inputs']
        inputs_valid = net['inputs_valid']
        inputs_HW    = net['inputs_HW']
        labels       = net['labels']
        y_init       = net['y_init']
        x0           = net['x0']
        y0           = net['y0']
        target       = net['target']

        # # placeholders for inputs
        # inputs = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='inputs')
        # inputs_valid = tf.placeholder(tf.bool, 
        #         shape=[o.batchsz, o.ntimesteps+1], 
        #         name='inputs_valid')
        # inputs_HW = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, 2], 
        #         name='inputs_HW')
        # labels = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.outdim], 
        #         name='labels')

        # placeholders for initializations of full-length sequences
        # NOTE: currently, h and c have the same dimension as input to ConvLSTM
        cnnh = self.params['cnn']['layer'][-1]['out_h']
        cnnw = self.params['cnn']['layer'][-1]['out_w']
        cnnch = self.params['cnn_1x1']['layer'][-1]['chout']
        h_init = tf.placeholder_with_default(
            tf.truncated_normal([o.batchsz, cnnh, cnnw, cnnch], dtype=o.dtype), 
            shape=[o.batchsz, cnnh, cnnw, cnnch], name='h_init')
        c_init = tf.placeholder_with_default(
            tf.truncated_normal([o.batchsz, cnnh, cnnw, cnnch], dtype=o.dtype), 
            shape=[o.batchsz, cnnh, cnnw, cnnch], name='c_init')

        # placeholders for x0 and y0. This is used for full-length sequences
        # NOTE: when it's not first segment, you should always feed x0, y0
        # otherwise it will use GT as default. It will be wrong then.
        #x0 = tf.placeholder_with_default(
        #        inputs[:,0], 
        #        shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], name='x0')
        #y0 = tf.placeholder_with_default(
        #        labels[:,0],
        #        shape=[o.batchsz, o.outdim], name='y0')

        # # NOTE: when it's not first segment, be careful what is passed to target.
        # # Make sure to pass x0, not the first frame of every segments.
        # target = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='target')


        # RNN unroll
        outputs = []
        for t in range(1, o.ntimesteps+1):
            if t==1:
                h_prev = h_init
                c_prev = c_init
            else:
                h_prev = h_curr
                c_prev = c_curr
            x_curr = inputs[:,t]

            # siamese
            feat_x0 = self._pass_cnn(target, o)
            feat_xt = self._pass_cnn(x_curr, o)
            scoremap = self._depthwise_convolution(feat_xt, feat_x0, o)
            scoremap = self._pass_cnn_1x1s(scoremap, o)

            # ConvLSTM (try convGRU)
            h_curr, c_curr = self._pass_rnn(scoremap, h_prev, c_prev, o)

            # regress output heatmap using deconvolution layers (transpose of conv)
            h_deconv = self._pass_deconvolution(h_curr, o)
            y_curr = self._project_output(h_deconv)

            # TODO: debugging nan
            dbg = tf.reduce_sum(tf.cast(tf.is_nan(c_curr), dtype=o.dtype)) 
            #dbg = tf.Print(dbg, [dbg], 'dbg:')

            outputs.append(y_curr)
        outputs = tf.stack(outputs, axis=1) # list to tensor

        loss = get_loss(outputs, labels, inputs_valid, inputs_HW, o, 'heatmap')
        tf.add_to_collection('losses', loss)
        loss_total = tf.reduce_sum(tf.get_collection('losses'),name='loss_total')

        # net = {
        net.update({
                # 'target': target, 
                # 'inputs': inputs,
                # 'inputs_valid': inputs_valid,
                # 'inputs_HW': inputs_HW,
                # 'labels': labels,
                'outputs': outputs,
                'loss': loss_total,
                'h_init': h_init,
                'c_init': c_init,
                'h_last': h_curr,
                'c_last': c_curr,
                'dbg': dbg
                })
        return net


class RNN_conv_asymm(object):
    def __init__(self, o, stat=None):
        self.is_train = True if o.mode == 'train' else False
        # self.params = self._update_params(o)
        self.net = self._load_model(o, stat=stat)

    def _load_model(self, o, stat=None):
        net = make_input_placeholders(o, stat)
        inputs = net['inputs']
        x0     = net['x0']
        y0     = net['y0']
        masks = get_masks_from_rectangles(y0, o)
        init_input = tf.concat([x0, masks], axis=3)

        def input_cnn(x, num_outputs, name='input_cnn'):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                    with slim.arg_scope([slim.conv2d],
                                        weights_regularizer=slim.l2_regularizer(o.wd)):
                        x = slim.conv2d(x, 16, kernel_size=7, stride=2, scope='conv1')
                        x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool1')
                        x = slim.conv2d(x, 32, kernel_size=5, stride=1, scope='conv2')
                        x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool2')
                        x = slim.conv2d(x, num_outputs, kernel_size=3, stride=1, scope='conv3')
                        x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool3')
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
            return h, c

        def output_cnn(x, name='output_cnn'):
            with tf.name_scope(name):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                        weights_regularizer=slim.l2_regularizer(o.wd)):
                        x = slim.conv2d(x, 128, kernel_size=3, stride=2, scope='conv1')
                        x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool1')
                        x = slim.conv2d(x, 256, kernel_size=3, stride=1, scope='conv2')
                        x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool2')
                        x = slim.flatten(x)
                        x = slim.fully_connected(x, 4, scope='predict')
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

        outputs = []
        ht, ct = h_init, c_init
        for t in range(1, o.ntimesteps+1):
            xt = inputs[:, t]
            with tf.name_scope('frame_cnn_{}'.format(t)) as scope:
                with tf.variable_scope('frame_cnn', reuse=(t > 1)):
                    # Pass name scope from above, otherwise makes new name scope
                    # within name scope created by variable scope.
                    rt = input_cnn(xt, num_outputs=lstm_dim, name=scope)
            with tf.name_scope('conv_lstm_{}'.format(t)) as scope:
                with tf.variable_scope('conv_lstm', reuse=(t > 1)):
                    ht, ct = conv_lstm(rt, ht, ct, state_dim=lstm_dim, name=scope)
            with tf.name_scope('out_cnn_{}'.format(t)) as scope:
                with tf.variable_scope('out_cnn', reuse=(t > 1)):
                    yt = output_cnn(ht, name=scope)
            outputs.append(yt)
            # tf.get_variable_scope().reuse_variables()
        outputs = tf.stack(outputs, axis=1) # list to tensor
        h_last, c_last = ht, ct
        state_vars = [(h_init, h_last), (c_init, c_last)]

        field = cnnutil.find_rf(xt, rt)
        print 'CNN receptive field:'
        print '  size:', field.rect.size()
        print '  center offset:', field.rect.int_center()
        print '  stride:', field.stride

        with tf.name_scope('loss'):
            loss_pred = get_loss(outputs, net['labels'], net['inputs_valid'], net['inputs_HW'], o,
                                 outtype='rectangle',
                                 name='pred')
            loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss_total = loss_pred + loss_reg
            # tf.add_to_collection('losses', loss)
            # loss_total = tf.reduce_sum(tf.get_collection('losses'), name='loss_total')
            with tf.name_scope('summary'):
                tf.summary.scalar('pred', loss_pred)
                tf.summary.scalar('reg', loss_reg)
                tf.summary.scalar('total', loss_total)

        net.update({
            'outputs': outputs,
            'loss':    loss_total,
            'h_init':  h_init,
            'c_init':  c_init,
            'h_last':  ht,
            'c_last':  yt,
        })
        return net


class NonRecur(object):
    def __init__(self, o, stat=None):
        self.is_train = True if o.mode == 'train' else False
        self.params = {}
        self._update_params_cnn(o)
        self.net = self._load_model(o, stat=stat) 

    def _update_params_cnn(self, o):
        # non-learnable params; dataset dependent
        if o.dataset in ['moving_mnist', 'bouncing_mnist']: # easy datasets
            self.nlayers = 2
            self.nchannels = [16, 16]
            self.filtsz = [3, 3]
            self.strides = [3, 3]
        else: # ILSVRC or other data set of natural images 
            # TODO: potentially a lot of room for improvement here
            self.nlayers = 3
            self.nchannels = [16, 32, 64]
            self.filtsz = [7, 5, 3]
            self.strides = [3, 2, 1]
        assert(self.nlayers == len(self.nchannels))
        assert(self.nlayers == len(self.filtsz))
        assert(self.nlayers == len(self.strides))

        # learnable parameters; CNN params shared across all time steps
        w_conv = []
        b_conv = []
        # Previous image and current image.
        num_in = o.ninchannel*2
        if o.yprev_mode == 'concat_channel':
            num_in += 1
        if o.pass_yinit:
            # extra input image + mask
            num_in += o.ninchannel + 1
        for i in range(self.nlayers):
            with tf.name_scope('layer_{}'.format(i)):
                # two input images + 1 mask channel
                shape_w = [self.filtsz[i], self.filtsz[i], 
                        num_in if i==0 else self.nchannels[i-1], 
                        self.nchannels[i]]
                shape_b = [self.nchannels[i]]
                w_conv.append(get_weight_variable(shape_w,name_='w',wd=o.wd))
                b_conv.append(get_bias_variable(shape_b,name_='b',wd=o.wd))
        self.params['w_conv'] = w_conv
        self.params['b_conv'] = b_conv

    def _update_params_fc(self, x, o):
        # TODO: Assert that in_dim is known (static).
        in_dim = x.shape[-1]
        # TODO: Make this a parameter.
        m = 256
        # TODO: Variable number of layers?
        with tf.variable_scope('fc', reuse=False):
            w1 = tf.get_variable('w1', shape=(in_dim, m), dtype=tf.float32, initializer=None)
            b1 = tf.get_variable('b1', shape=(m,), initializer=tf.zeros_initializer(dtype=tf.float32))
            w2 = tf.get_variable('w2', shape=(m, o.outdim), dtype=tf.float32, initializer=None)
            b2 = tf.get_variable('b2', shape=(o.outdim,), initializer=tf.zeros_initializer(dtype=tf.float32))

    def _pass_cnn(self, x_curr, x_prev, y_prev, o, x0=None, y0=None):
        masks_yprev = get_masks_from_rectangles(y_prev, o)
        masks_yinit = get_masks_from_rectangles(y0, o)
        if not o.pass_yinit:
            if o.yprev_mode == 'concat_channel':
                cnnin = tf.concat((x_prev, x_curr, masks), 3)
            else:
                cnnin = tf.concat((x_prev, x_curr), 3)
        else:
            if o.yprev_mode == 'concat_channel':
                cnnin = tf.concat((x0, masks_yinit, x_prev, x_curr, masks_yprev), 3)
            else:
                cnnin = tf.concat((x0, masks_yinit, x_prev, x_curr), 3)

        # convolutions; feed-forward
        for i in range(self.nlayers):
            x = cnnin if i==0 else act
            conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], stride=self.strides[i])
            act = activate(conv, 'relu')
            if self.is_train and o.dropout_cnn and i==1: # NOTE: maybe only at 2nd layer
                act = tf.nn.dropout(act, o.keep_ratio_cnn)
        return act

    def _pass_fc(self, x, o):
        with tf.variable_scope('fc', reuse=True):
            w1 = tf.get_variable('w1')
            b1 = tf.get_variable('b1')
            w2 = tf.get_variable('w2')
            b2 = tf.get_variable('b2')

        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y = tf.matmul(h1, w2) + b2
        return y

    def _load_model(self, o, stat=None):
        net = make_input_placeholders(o, stat)
        inputs       = net['inputs']
        inputs_valid = net['inputs_valid']
        inputs_HW    = net['inputs_HW']
        labels       = net['labels']
        y_init       = net['y_init']
        x0           = net['x0']
        y0           = net['y0']
        target       = net['target']

        # # placeholders for inputs
        # inputs = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='inputs')
        # inputs_valid = tf.placeholder(tf.bool, 
        #         shape=[o.batchsz, o.ntimesteps+1], 
        #         name='inputs_valid')
        # inputs_HW = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, 2], 
        #         name='inputs_HW')
        # labels = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.ntimesteps+1, o.outdim], 
        #         name='labels')

        # # placeholders for initializations of full-length sequences
        # y_init = tf.placeholder_with_default(
        #         labels[:,0], 
        #         shape=[o.batchsz, o.outdim], name='y_init')

        # # placeholders for x0 and y0. This is used for full-length sequences
        # # NOTE: when it's not first segment, you should always feed x0, y0
        # # otherwise it will use GT as default. It will be wrong then.
        # x0 = tf.placeholder_with_default(
        #         inputs[:,0], 
        #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], name='x0')
        # y0 = tf.placeholder_with_default(
        #         labels[:,0],
        #         shape=[o.batchsz, o.outdim], name='y0')
        # # NOTE: when it's not first segment, be careful what is passed to target.
        # # Make sure to pass x0, not the first frame of every segments.
        # target = tf.placeholder(o.dtype, 
        #         shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], 
        #         name='target')

        # RNN unroll
        outputs = []
        fc_init = False
        for t in range(1, o.ntimesteps+1):
            if t==1:
                y_prev = y_init
            else:
                y_prev = y_curr
            x_prev = inputs[:,t-1]
            x_curr = inputs[:,t]

            feat = self._pass_cnn(x_curr, x_prev, y_prev, o, x0, y0)
            feat = tf.reshape(feat, [o.batchsz, -1])
            if not fc_init:
                self._update_params_fc(feat, o)
                fc_init = True
            y_curr = self._pass_fc(feat, o)
            outputs.append(y_curr)

        outputs = tf.stack(outputs, axis=1) # list to tensor

        loss = get_loss(outputs, labels, inputs_valid, inputs_HW, o, 'rectangle')
        tf.add_to_collection('losses', loss)
        loss_total = tf.reduce_sum(tf.get_collection('losses'),name='loss_total')

        # net = {
        net.update({
                # 'inputs': inputs,
                # 'inputs_valid': inputs_valid,
                # 'inputs_HW': inputs_HW,
                # 'labels': labels,
                'outputs': outputs,
                'loss': loss_total,
                # 'y_init': y_init,
                'y_last': y_curr,
                # 'x0': x0,
                # 'y0': y0,
                })
        return net


def load_model(o, stat=None):
    if o.model == 'RNN_basic':
        model = RNN_basic(o, stat=stat)
    elif o.model == 'RNN_new':
        model = RNN_new(o, stat=stat)
    elif o.model == 'RNN_conv_asymm':
        model = RNN_conv_asymm(o, stat=stat)
    elif o.model == 'CNN':
        model = NonRecur(o, stat=stat)
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

    if o.model == 'RNN_basic':
        o.pass_yinit = True
        m = RNN_basic(o)
    elif o.model == 'RNN_new':
        o.losses = ['ce', 'l2']
        m = RNN_new(o)

    pdb.set_trace()

