import pdb
import tensorflow as tf
import os


class CNN(object):
    def __init__(self, o):
        self.batchsz        = o.batchsz
        self.ntimesteps     = o.ntimesteps
        self.nlayers        = o.cnn_nlayers
        self.nchannels      = o.cnn_nchannels
        self.filtsz         = o.cnn_filtsz
        self.strides        = o.cnn_strides
        self.wd             = o.wd
        self.ninchannel     = o.ninchannel

    def create_network(self, inputs):
        def _run_sanitycheck(inputs):
            input_shape = inputs.get_shape().as_list()
            #assert(len(input_shape)==4) # TODO: assuming one channel image; change later
            assert(self.batchsz == input_shape[0])
            assert(self.ntimesteps == input_shape[1])

        def _get_cnn_params():
            # CNN params shared across all time steps
            def _weight_variable(shape, wd=0.0): # TODO: this can be used global (maybe with scope)
                initial = tf.truncated_normal(shape, stddev=0.1)
                weight_decay = tf.nn.l2_loss(initial) * wd
                tf.add_to_collection('losses', weight_decay)
                return tf.Variable(initial)
            def _bias_variable(shape): # TODO: this can be used global (maybe with scope)
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)
            w_conv = []
            b_conv = []
            for i in range(self.nlayers):
                shape_w = [
                        self.filtsz[i], self.filtsz[i], # TODO: try different sz
                        self.ninchannel if i==0 else self.nchannels[i-1], 
                        self.nchannels[i]]
                shape_b = [self.nchannels[i]]
                w_conv.append(_weight_variable(shape_w, wd=self.wd))
                b_conv.append(_bias_variable(shape_b))
            return w_conv, b_conv

        def _conv2d(x, w, b, strides_):
            return tf.nn.conv2d(x, w, strides=strides_, padding='SAME') + b

        def _activate(input_, activation_='relu'):
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

        _run_sanitycheck(inputs)
        w_conv, b_conv = _get_cnn_params()
        activations = []
        for t in range(self.ntimesteps): # TODO: variable length inputs!!!
            for i in range(self.nlayers):
                #x = tf.expand_dims(inputs[:,t],3) if i==0 else relu
                x = inputs[:,t] if i==0 else relu
                st = self.strides[i]
                # TODO: convolution different stride at each layer
                conv = _conv2d(x, w_conv[i], b_conv[i], strides_=[1,st,st,1])
                if False: # i == self.nlayers-1: # last layer
                    relu = _activate(conv, activation_='linear')
                else: 
                    relu = _activate(conv, activation_='relu')
            activations.append(relu)
        outputs = tf.stack(activations, axis=1)
        return outputs


class RNN_basic(object):
    def __init__(self, o, is_training=False):
        self._is_training = is_training
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

    def _create_network(self, o):
        inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel]
        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(tf.int32, shape=[o.batchsz])
        inputs_HW = tf.placeholder(o.dtype, shape=[o.batchsz, 2])
        labels = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, o.outdim])

        # CNN 
        if o.cnn_pretrain: # use pre-trained model
            raise ValueError('not implemented yet')
        else: 
            cnn = CNN(o)
            self._cnnout_update(cnn, inputs)

        # RNN
        if o.usetfapi: # TODO: will (and should) be deprecated. 
            outputs = self._rnn_pass_API(inputs_cnn, inputs_length, o)
        else: # manual rnn module
            outputs = self._rnn_pass(labels, o) 
 
        # TODO: once variable length, labels should respect that setting too!
        loss = get_loss(outputs, labels, inputs_length, inputs_HW)
        tf.add_to_collection('losses', loss)
        loss_total = tf.add_n(tf.get_collection('losses'), name='loss_total')

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
            if o.yprev_mode == 'nouse':
                shape_W = [o.nunits+self.featdim, o.nunits*4] 
            elif o.yprev_mode == 'concat_abs':
                shape_W = [o.nunits+self.featdim+o.outdim, o.nunits*4] 
            elif o.yprev_mode == 'weight':
                shape_W = [o.nunits+self.featdim, o.nunits*4] 
            else:
                raise ValueError('not implemented yet')
            shape_b = [o.nunits*4]
            with tf.variable_scope('LSTMcell'):
                self.params['W_lstm'] = tf.get_variable(
                    'W_lstm', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer(stddev=0.001)) 
                self.params['b_lstm'] = tf.get_variable(
                    'b_lstm', shape=shape_b, dtype=o.dtype, 
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
            # forget, input, memory cell, output, hidden 
            if o.yprev_mode == 'nouse':
                x_curr = tf.reshape(x_curr, [o.batchsz, -1])
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr), 1), W_lstm) + b_lstm, 4, 1)
            elif o.yprev_mode == 'concat_abs':
                x_curr = tf.reshape(x_curr, [o.batchsz, -1])
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr, y_prev), 1), W_lstm) + b_lstm, 4, 1)
            elif o.yprev_mode == 'weight':
                # beta and gamma
                # TODO: optimum beta?
                s_all = tf.constant(o.frmsz ** 2, dtype=o.dtype) 
                s_roi = (y_prev[:,2]-y_prev[:,0]) * (y_prev[:,3]-y_prev[:,1])
                tf.assert_positive(s_roi)
                tf.assert_greater_equal(s_all, s_roi)
                beta = 0.95 # TODO: make it optionable
                gamma = ((1-beta)/beta) * s_all / s_roi # NOTE: division?
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

                x_curr_weighted = tf.reshape(x_curr_weighted, [o.batchsz, -1])
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr_weighted), 1), W_lstm)+ b_lstm, 
                    4, 1)
            else:
                raise ValueError('not implemented yet')
            if o.lstmforgetbias:
                C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            else:
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
            h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            return h_curr, C_curr

        # get params
        _get_lstm_params(o)
        _get_rnnout_params(o)

        if self._is_training:
            # initial states 
            h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
            C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) # or normal init
            y_prev = labels[:,0]

            # unroll
            outputs = []
            for t in range(o.ntimesteps): # TODO: change if variable length input/out
                h_prev, C_prev = _activate_rnncell(
                    self.cnnout['feat'][:,t], h_prev, C_prev, y_prev, o) 
                if t == 0: # NOTE: pass ground-truth at t=2 (if t<n)
                    y_prev = labels[:,0]
                else:
                    y_prev = tf.matmul(h_prev, self.params['W_out']) \
                        + self.params['b_out']
                outputs.append(y_prev)
        else: # test
            pdb.set_trace() # WIP

            # initial states 
            # TODO: this changes whether t<n or t>=n; need such a signal.
            h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
            C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
            y_prev = labels[:,0]

            # unroll
            outputs = []
            for t in range(o.ntimesteps): # TODO: change if variable length input/out
                h_prev, C_prev = _activate_rnncell(
                    self.cnnout['feat'][:,t], h_prev, C_prev, y_prev, o) 
                if t == 0: # NOTE: pass ground-truth at t=2 (if t<n)
                    y_prev = labels[:,0]
                else:
                    y_prev = tf.matmul(h_prev, self.params['W_out']) \
                        + self.params['b_out']
                outputs.append(y_prev)
       
        # list to tensor
        outputs = tf.stack(outputs, axis=1)
        return outputs 

    def _rnn_pass_API(self, inputs_cnn, inputs_length, o):
        # get rnn cell
        def _get_rnncell(o, is_training=False):
            if o.cell_type == 'LSTM':
                '''
                cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=o.nunits, cell_clip=o.max_grad_norm) \
                        if o.grad_clip else tf.nn.rnn_cell.LSTMCell(num_units=o.nunits) 
                '''
                cell = tf.contrib.rnn.LSTMCell(num_units=o.nunits)
            elif o.cell_type == 'GRU':
                cell = tf.contrib.rnn.GRUCell(num_units=o.nunits)
            elif o.cell_type == 'basic':
                cell = tf.contrib.rnn.BasicRNNCell(num_units=o.nunits)
            else:
                raise ValueError('cell not implemented yet or simply wrong!')
            # rnn drop out (only during training)
            if is_training and o.dropout_rnn:
                cell = tf.contrib.rnn.DropoutWrapper(
                        cell,output_keep_prob=o.keep_ratio)
            # multi-layers
            if o.rnn_nlayers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*o.rnn_nlayers)
            return cell

        # reshape 
        inputs_cell = tf.reshape(inputs_cnn,[o.batchsz,o.ntimesteps,-1])

        cell = _get_rnncell(o, is_training=self._is_training)
        cell_outputs, _ = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=o.dtype,
                sequence_length=inputs_length,
                inputs=inputs_cell)
        cell_outputs = tf.reshape(cell_outputs, [o.batchsz*o.ntimesteps,o.nunits])
        # TODO: fixed the dimension of b_out, but didn't test it yet
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits, o.outdim], dtype=o.dtype)
        b_out = tf.get_variable('b_out', shape=[o.outdim], dtype=o.dtype)
        outputs = tf.matmul(cell_outputs, w_out) + b_out
        outputs = tf.reshape(outputs, [-1, o.ntimesteps, o.outdim]) # orig. shape
        return outputs


class RNN_attention_s(object):
    def __init__(self, o, is_training=False):
        self._is_training = is_training
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
 
        # TODO: once variable length, labels should respect that setting too!
        loss = get_loss(outputs, labels, inputs_length, inputs_HW)
        tf.add_to_collection('losses', loss)
        loss_total = tf.add_n(tf.get_collection('losses'), name='loss_total')

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
    def __init__(self, o, is_training=False):
        self._is_training = is_training
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

        # TODO: once variable length, labels should respect that setting too!
        loss = get_loss(outputs, labels, inputs_length, inputs_HW)
        tf.add_to_collection('losses', loss)
        loss_total = tf.add_n(tf.get_collection('losses'), name='loss_total')

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


def get_loss(outputs, labels, inputs_length, inputs_HW):
    # loss = tf.reduce_mean(tf.square(outputs-labels)) # previous loss 

    # loss1: sum of two L2 distances for left-top and right-bottom
    loss1_batch = []
    for i in range(labels.get_shape().as_list()[0]): # batchsz or # examples
        sq = tf.square(
                outputs[i,:inputs_length[i]]-labels[i,:inputs_length[i]])
        # NOTE: here it's mean over timesteps. If only want to have a loss for
        # the last time step, it should be changed.
        mean_of_l2_sum = tf.reduce_mean(tf.add(
            tf.sqrt(tf.add(sq[:,0],sq[:,1])), 
            tf.sqrt(tf.add(sq[:,2],sq[:,3]))))
        loss1_batch.append(mean_of_l2_sum)
    loss_box = tf.reduce_mean(tf.stack(loss1_batch, axis=0))

    # loss2: IoU
    scalar = tf.stack(
            (inputs_HW[:,1], inputs_HW[:,0], inputs_HW[:,1], inputs_HW[:,0]), 
            axis=1)
    boxA = outputs * tf.expand_dims(scalar, 1)
    boxB = labels * tf.expand_dims(scalar, 1)
    xA = tf.maximum(boxA[:,:,0], boxB[:,:,0])
    yA = tf.maximum(boxA[:,:,1], boxB[:,:,1])
    xB = tf.minimum(boxA[:,:,2], boxB[:,:,2])
    yB = tf.minimum(boxA[:,:,3], boxB[:,:,3])
    interArea = (xB - xA + 1) * (yB - yA + 1) # NEED actual image size
    boxAArea = (boxA[:,:,2] - boxA[:,:,0] + 1) * (boxA[:,:,3] - boxA[:,:,1] + 1) 
    boxBArea = (boxB[:,:,2] - boxB[:,:,0] + 1) * (boxB[:,:,3] - boxB[:,:,1] + 1) 
    # TODO: CHECK tf.div or tf.divide
    #iou = tf.div(interArea, (boxAArea + boxBArea - interArea) + 1e-4)
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-4) 
    iou_valid = []
    for i in range(labels.get_shape().as_list()[0]):
        iou_valid.append(iou[i, :inputs_length[i]])
    iou_mean = tf.reduce_mean(iou_valid)
    loss_iou = 1 - iou_mean

    # loss3: penalty loss considering the structure in (x1, x2, y1, y2)
    # loss4: cross-entropy between probabilty maps (need to change label) 

    loss = loss_box + loss_iou
    #loss = loss_box
    return loss

def load_model(o):
    is_training = True if o.mode == 'train' else False
    # TODO: check by actually logging device placement!
    with tf.device('/{}:{}'.format(o.device, o.device_number)):
        if o.model == 'rnn_basic':
            model = RNN_basic(o, is_training=is_training) 
        elif o.model == 'rnn_attention_s':
            model = RNN_attention_s(o, is_training=is_training)
        elif o.model == 'rnn_attention_st':
            model = RNN_attention_st(o, is_training=is_training)
        elif o.model == 'rnn_attention_st_bidirectional':
            raise ValueError('model not implemeted yet')
        else:
            raise ValueError('model not implemented yet or simply wrong..')
        return model 

if __name__ == '__main__':
    '''Test model 
    '''

    from opts import Opts
    o = Opts()
    o.mode = 'train'
    o.dataset = 'bouncing_mnist'
    o.batchsz = 10
    o._set_dataset_params()
    o.yprev_mode = 'weight' # nouse, concat_abs, weight
    o.model = 'rnn_attention_st' # rnn_basic, rnn_attention_s, rnn_attention_st
    #o.usetfapi = True
    m = load_model(o)
    pdb.set_trace()

