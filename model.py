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
            assert(len(input_shape)==4) # TODO: assuming one channel image; change later
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
                x = tf.expand_dims(inputs[:,t],3) if i==0 else relu
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
        if o.ninchannel == 1:
            inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz]
        else: 
            inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel]
        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
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
            #outputs = self._rnn_pass(inputs_cnn, labels[:,0], o) # NOTE: no y_prev passing; will be deprecated
            outputs = self._rnn_pass_wip(labels, o) 
 
        # TODO: once variable length, labels should respect that setting too!
        loss_l2 = tf.reduce_mean(tf.square(outputs-labels))
        tf.add_to_collection('losses', loss_l2)
        loss_total = tf.add_n(tf.get_collection('losses'), name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
                'labels': labels,
                'outputs': outputs,
                'loss': loss_total}
        return net

    def _rnn_pass_wip(self, labels, o):

        def _get_lstm_params(o):
            if o.yprev_mode == 'nouse':
                shape_W = [o.nunits+self.featdim, o.nunits*4] 
            elif o.yprev_mode == 'concat_abs':
                shape_W = [o.nunits+self.featdim+o.outdim, o.nunits*4] 
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
            x_curr = tf.reshape(x_curr, [o.batchsz, -1])
            if o.yprev_mode == 'nouse':
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr), 1), W_lstm) + b_lstm, 4, 1)
            elif o.yprev_mode == 'concat_abs':
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
                    tf.concat_v2((h_prev, x_curr, y_prev), 1), W_lstm) + b_lstm, 4, 1)
            elif o.yprev_mode == 'weight': # interpret as filtering
                raise ValueError('WIP')
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

        # initial states 
        #TODO: this changes whether t<n or t>=n
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
            
        # list to tensor
        outputs = tf.stack(outputs, axis=1)
        return outputs 

    def _rnn_pass(self, inputs, label_init, o): # TODO: will be deprecated.
        #raise ValueError('This will be deprecated!')
        ''' This is a (manually designed) rnn unrolling function.
        Note that this is not supposed to reproduce the same (or similar) 
        results that one would produce from using tensorflow API.
        One clear distinction is that this module uses the ground-truth label 
        for the first frame; that is specific for tracking task. 
        Also, besides the previous hidden state and current input, rnn cell 
        receives the previous output. How we are using this is undecided or 
        unexplored yet.
        '''
        def _get_lstm_params(nunits, featdim):
            shape_W = [nunits+featdim, nunits*4] 
            shape_b = [nunits*4]
            params = {}
            with tf.variable_scope('LSTMcell'):
                params['W'] = tf.get_variable('W', shape=shape_W, dtype=o.dtype, 
                        initializer=tf.truncated_normal_initializer(stddev=0.001)) 
                params['b'] = tf.get_variable('b', shape=shape_b, dtype=o.dtype, 
                        initializer=tf.constant_initializer())
            return params

        def _activate_rnncell(x_curr, h_prev, C_prev, params, o):
            if o.cell_type == 'LSTM': # standard LSTM (no peephole)
                W = params['W']
                b = params['b']
                # forget, input, memory cell, output, hidden 
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                        tf.matmul(tf.concat_v2((h_prev,x_curr),1), W) + b, 4, 1)
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
                h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            elif o.cell_type == 'LSTM_variant': # coupled input and forget gates
                raise ValueError('Not implemented yet!')
            else:
                raise ValueError('Not implemented yet!')
            return h_curr, C_curr

        # reshape 
        inputs = tf.reshape(inputs,[o.batchsz,o.ntimesteps,-1])

        # lstm weight and bias parameters
        params = _get_lstm_params(o.nunits, inputs.get_shape().as_list()[-1])
        
        # initial states
        # TODO: normal init state? -> seems zero state is okay. no difference.
        h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)

        # unroll
        hs = []
        for t in range(o.ntimesteps): # TODO: change if variable length input/out
            h_prev, C_prev = _activate_rnncell(inputs[:,t], h_prev, C_prev, params, o) 
            hs.append(h_prev)

        # list to tensor
        cell_outputs = tf.stack(hs, axis=1)
        cell_outputs = tf.reshape(cell_outputs, [-1, o.nunits])
        # TODO: fixed the dimension of b_out
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits, o.outdim], dtype=o.dtype)
        b_out = tf.get_variable('b_out', shape=[o.outdim], dtype=o.dtype)
        outputs = tf.matmul(cell_outputs, w_out) + b_out
        outputs = tf.reshape(outputs, [-1, o.ntimesteps, o.outdim]) # orig. shape
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
        if o.ninchannel == 1:
            inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz]
        else: 
            inputs_shape = [o.batchsz, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel]
        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
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
        loss_l2 = tf.reduce_mean(tf.square(outputs-labels))
        tf.add_to_collection('losses', loss_l2)
        loss_total = tf.add_n(tf.get_collection('losses'), name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
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
                '''This mode uses y_prev to put higher weights on the region in
                x_curr (before x_curr is combined with the hidden state). But, 
                the sum of x_curr is preserved after reweighting.
                (eq) beta*S_all + beta*gamma*S_roi = S_all 
                <-> gamma = ((1-beta)/beta) * (S_all/S_roi)
                beta and gamma: decreasing and increasing factor respectively.
                '''
                # beta and gamma
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
                    initial = tf.constant(beta, 
                            shape=[self.cnnout['h'], self.cnnout['w']])
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
            e_exp = tf.exp(e)
            alpha = e_exp/tf.expand_dims(tf.reduce_sum(e_exp,axis=1),1)

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
            #------------------------------------------------------------------
            # NOTE: A thought on passing y_prev to the next rnn cell.
            # I feel that this is unnecessary and can be wrong, especially for 
            # the spatial attention model. Currently, it is not being used for 
            # the attention model. 
            # However, this also means that I am not yet using the GT patch at
            # the first frame. Need to think about how you can use the GT patch.
            # Also, it means that it's not actually tracking a target, rather
            # it is performing some "unknown" object detection. 
            # 
            # *One thing to try out is to use both y_prev and h_prev to compute
            # attention model. 
            # You still need to give a careful thought on whether conv(Hidden)
            # is different from just Hidden, and if it actually has a meaning.
            #------------------------------------------------------------------
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


def load_model(o):
    is_training = True if o.mode == 'train' else False
    # TODO: check by actually logging device placement!
    with tf.device('/{}:{}'.format(o.device, o.device_number)):
        if o.model == 'rnn_basic':
            model = RNN_basic(o, is_training=is_training) 
        elif o.model == 'rnn_attention_s':
            model = RNN_attention_s(o, is_training=is_training)
        elif o.model == 'rnn_bidirectional_attention':
            raise ValueError('model not implemeted yet')
        else:
            raise ValueError('model not implemented yet or simply wrong..')
        return model 

if __name__ == '__main__':
    '''Test model 
    '''

    from opts import Opts
    import data
    o = Opts()
    o.mode = 'train'
    o.dataset = 'bouncing_mnist'
    o.batchsz = 20
    o.frmsz = 100
    o.ninchannel = 1
    o.outdim = 4
    o.yprev_mode = 'weight' # nouse, concat_abs, weight
    o.model = 'rnn_attention_s' # rnn_basic, rnn_attention_s
    #o.usetfapi = True
    m = load_model(o)
    pdb.set_trace()

