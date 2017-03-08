import pdb
import tensorflow as tf
import os


class CNN(object):
    def __init__(self, o, is_train):
        self.is_train = is_train # dropout
        self._update_params(o)

    def _update_params(self, o):
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
            '''
            self.nlayers = 4
            self.nchannels = [16, 32, 64, 64]
            self.filtsz = [7, 5, 3, 3]
            self.strides = [3, 2, 1, 1]
            '''
        assert(self.nlayers == len(self.nchannels))
        assert(self.nlayers == len(self.filtsz))
        assert(self.nlayers == len(self.strides))

        # learnable parameters; CNN params shared across all time steps
        w_conv = []
        b_conv = []
        for i in range(self.nlayers):
            with tf.name_scope('layer_{}'.format(i)):
                if o.yprev_mode == 'concat_channel':
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

                elif o.yprev_mode == 'concat_abs':
                    shape_w = [self.filtsz[i], self.filtsz[i],
                            o.ninchannel if i==0 else self.nchannels[i-1],
                            self.nchannels[i]]
                shape_b = [self.nchannels[i]]
                w_conv.append(get_weight_variable(shape_w,name_='w',wd=o.wd))
                b_conv.append(get_bias_variable(shape_b,name_='b',wd=o.wd))
        self.params = {}
        self.params['w_conv'] = w_conv
        self.params['b_conv'] = b_conv

    def pass_onetime(self, x_curr, x_prev, y_prev, o, x0=None, y0=None):
        if o.yprev_mode == 'concat_channel':
            if not o.pass_yinit:
                # create masks from y_prev
                masks = self._get_masks_from_rectangles(y_prev, o)

                # input concat
                cnnin = tf.concat((x_prev, x_curr, masks), 3)
            else:
                masks_yprev = self._get_masks_from_rectangles(y_prev, o)
                masks_yinit = self._get_masks_from_rectangles(y0, o)
                cnnin = tf.concat(
                        (x0, masks_yinit, x_prev, x_curr, masks_yprev), 3)

            # convolutions; feed-forward
            for i in range(self.nlayers):
                x = cnnin if i==0 else act
                conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], 
                        [1,self.strides[i],self.strides[i],1])
                act = activate(conv, 'relu')
                if self.is_train and o.dropout_cnn and i==1: # NOTE: maybe only at 2nd layer
                    act = tf.nn.dropout(act, o.keep_ratio_cnn)
            return act
        elif o.yprev_mode == 'concat_abs':
            # NOTE:  This is a test to debug concat_channel!
            # 1. For test purpose, not passing x_prev here
            # 2. In case of concat_abs, can't concat before CNN.
            for i in range(self.nlayers):
                x = x_curr if i==0 else act
                conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], 
                        [1,self.strides[i],self.strides[i],1])
                act = activate(conv, 'relu')
                if self.is_train and o.dropout_cnn and i==1:
                    act = tf.nn.dropout(act, o.keep_ratio_cnn)
            return tf.concat((tf.reshape(act, [o.batchsz, -1]), y_prev), 1)

    def _get_masks_from_rectangles(self, rec, o):
        # create mask using rec; typically rec=y_prev
        x1 = rec[:,0] * o.frmsz
        y1 = rec[:,1] * o.frmsz
        x2 = rec[:,2] * o.frmsz
        y2 = rec[:,3] * o.frmsz
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


class RNN_basic(object):
    def __init__(self, o, is_train=True):
        self._is_train = is_train

    def update_params(self, cnnout, o):
        params = {}

        if o.yprev_mode == 'concat_channel':
            # cnn params
            cnnout_shape = cnnout.get_shape().as_list()
            params['cnn_h'] = cnnout_shape[1]
            params['cnn_w'] = cnnout_shape[2]
            params['cnn_c'] = cnnout_shape[3]
            params['cnn_featdim'] = cnnout_shape[1]*cnnout_shape[2]*cnnout_shape[3]
        elif o.yprev_mode == 'concat_abs':
            params['cnn_featdim'] = cnnout.get_shape().as_list()[1]

        # lstm params
        with tf.variable_scope('LSTMcell'):
            shape_w = [o.nunits+params['cnn_featdim'], o.nunits*4] 
            shape_b = [o.nunits*4]
            params['w_lstm'] = tf.get_variable(
                'w_lstm', shape=shape_w, dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1)) 
            params['b_lstm'] = tf.get_variable(
                'b_lstm', shape=shape_b, dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))

        # rnnout params 
        with tf.variable_scope('rnnout'):
            '''
            params['w_out'] = tf.get_variable(
                'w_out', shape=[o.nunits, o.outdim], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            params['b_out'] = tf.get_variable(
                'b_out', shape=[o.outdim], dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))
            '''
            params['w_out1'] = tf.get_variable(
                'w_out1', shape=[o.nunits, o.nunits/2], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            params['b_out1'] = tf.get_variable(
                'b_out1', shape=[o.nunits/2], dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))
            params['w_out2'] = tf.get_variable(
                'w_out2', shape=[o.nunits/2, o.outdim], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            params['b_out2'] = tf.get_variable(
                'b_out2', shape=[o.outdim], dtype=o.dtype, 
                initializer=tf.constant_initializer(0.1))

        self.params = params

    def pass_onetime(self, cnnout, h_prev, C_prev, o):
        # directly flatten cnnout and concat with h_prev 
        # NOTE: possibly other options such as projection of cnnout before 
        # concatenating with h_prev; C. Olah -> no need. 
        # TODO: try Conv LSTM
        xy_in = tf.reshape(cnnout, [o.batchsz, -1])

        f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
            tf.concat((h_prev, xy_in), 1), self.params['w_lstm']) + 
            self.params['b_lstm'], 4, 1)

        if o.lstmforgetbias:
            C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        else:
            C_curr = tf.sigmoid(f_curr) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
        return h_curr, C_curr


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
 
        loss = get_loss(outputs, labels, inputs_length, inputs_HW, o)
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
                input_to_mlp = tf.concat(
                    (tf.reshape(x_curr, [o.batchsz, -1]), h_prev), 1) 
            elif o.yprev_mode == 'concat_abs':
                input_to_mlp = tf.concat(
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
                input_to_mlp = tf.concat(
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
                    tf.concat((h_prev, x_curr), 1), W_lstm) + b_lstm, 4, 1)
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
                tf.matmul(tf.concat((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

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

        loss = get_loss(outputs, labels, inputs_length, inputs_HW, o)
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
            input_to_mlp = tf.concat(
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
                tf.matmul(tf.concat((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

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
            input_to_mlp = tf.concat(
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
                tf.matmul(tf.concat((h_prev, z), 1), W_lstm) + b_lstm, 4, 1)

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


def get_loss(outputs, labels, inputs_valid, inputs_HW, o):
    # loss = tf.reduce_mean(tf.square(outputs-labels)) # previous loss 
    ''' previous L2 loss
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
    '''

    # NOTE: Be careful about length of labels and outputs. 
    # labels and inputs_valid will be of T+1 length, and y0 shouldn't be used.

    assert(outputs.get_shape().as_list()[1] == o.ntimesteps)
    assert(labels.get_shape().as_list()[1] == o.ntimesteps+1)

    loss = []

    # loss1: sum of two l1 distances for left-top and right-bottom
    if 'l1' in o.losses: # TODO: double check
        labels_valid = tf.boolean_mask(labels[:,1:], inputs_valid[:,1:])
        outputs_valid = tf.boolean_mask(outputs, inputs_valid[:,1:])
        loss_l1 = tf.reduce_mean(tf.abs(labels_valid - outputs_valid))
        loss.append(loss_l1)

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
        loss.append(loss_iou)

    # loss3: CLE
    # loss4: cross-entropy between probabilty maps (need to change label) 

    return tf.reduce_sum(loss)

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

def conv2d(x, w, b, strides_):
    return tf.nn.conv2d(x, w, strides=strides_, padding='SAME') + b

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


class RNN_basic(object):
    def __init__(self, o):
        self.is_train = True if o.mode == 'train' else False

        self.params = {}
        self._update_params_cnn(o)
        self.net = self._load_model(o) 

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
            conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], 
                    [1,self.strides[i],self.strides[i],1])
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

    def _load_model(self, o):
        # placeholders for inputs
        inputs = tf.placeholder(o.dtype, 
                shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
                name='inputs')
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
        h_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zeor or normal
                shape=[o.batchsz, o.nunits], name='h_init')
        C_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zeor or normal
                shape=[o.batchsz, o.nunits], name='C_init')
        y_init = tf.placeholder_with_default(
                labels[:,0], 
                shape=[o.batchsz, o.outdim], name='y_init')

        # placeholders for x0 and y0. This is used for full-length sequences
        # NOTE: when it's not first segment, you should always feed x0, y0
        # otherwise it will use GT as default. It will be wrong then.
        x0 = tf.placeholder_with_default(
                inputs[:,0], 
                shape=[o.batchsz, o.frmsz, o.frmsz, o.ninchannel], name='x0')
        y0 = tf.placeholder_with_default(
                labels[:,0],
                shape=[o.batchsz, o.outdim], name='y0')

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
        loss_total = tf.add_n(tf.get_collection('losses'),name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_valid': inputs_valid,
                'inputs_HW': inputs_HW,
                'labels': labels,
                'outputs': outputs,
                'loss': loss_total,
                'h_init': h_init,
                'C_init': C_init,
                'y_init': y_init,
                'h_last': h_curr,
                'C_last': C_curr,
                'y_last': y_curr,
                'x0': x0,
                'y0': y0
                }
        return net


class RNN_a(object):
    def __init__(self, o):
        self.is_train = True if o.mode == 'train' else False

        self.params = {}
        self._update_params_cnn(o)
        self.net = self._load_model(o) 

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

        # learnable parameters; TODO: add name or variable scope

        # CNN for RNN input images; shared across all time steps
        w_conv = []
        b_conv = []
        for i in range(self.nlayers):
            # only one input image
            shape_w = [self.filtsz[i], self.filtsz[i], 
                    o.ninchannel if i==0 else self.nchannels[i-1], 
                    self.nchannels[i]]
            shape_b = [self.nchannels[i]]
            w_conv.append(get_weight_variable(shape_w, name_='w', wd=o.wd))
            b_conv.append(get_bias_variable(shape_b, name_='b', wd=o.wd))
        self.params['w_conv'] = w_conv
        self.params['b_conv'] = b_conv

        # 1x1 CNN params; used before putting cnn features to RNN cell
        self.params['w_cellin1'] = get_weight_variable(
                [1, 1, self.nchannels[-1], self.nchannels[-1]/2], 
                name_='w_cellin1', wd=o.wd)
        self.params['b_cellin1'] = get_bias_variable(
                [self.nchannels[-1]/2], name_='b_cellin1', wd=o.wd)
        self.params['w_cellin2'] = get_weight_variable(
                [1, 1, self.nchannels[-1]/2, self.nchannels[-1]/4], 
                name_='w_cellin2', wd=o.wd)
        self.params['b_cellin2'] = get_bias_variable(
                [self.nchannels[-1]/4], name_='b_cellin2', wd=o.wd)

        # CNN for initial target image (x0)
        # TODO: many options 
        # 1. crop x0 (before or after) based on y0
        # 2. concat x0 with y0
        # 3. multiply x0 with a binary mask created based on y0
        # for now, I am doing 2.
        w_conv_target = []
        b_conv_target = []
        for i in range(self.nlayers):
            # only one input image
            shape_w = [self.filtsz[i], self.filtsz[i], 
                    o.ninchannel+1 if i==0 else self.nchannels[i-1], 
                    self.nchannels[i]]
            shape_b = [self.nchannels[i]]
            w_conv_target.append(get_weight_variable(shape_w, name_='w', wd=o.wd))
            b_conv_target.append(get_bias_variable(shape_b, name_='b', wd=o.wd))
        self.params['w_conv_target'] = w_conv_target
        self.params['b_conv_target'] = b_conv_target

    def _update_params_rnn(self, cnnout, o):
        # cnn params
        cnnout_shape = cnnout.get_shape().as_list()
        self.params['cnn_h'] = cnnout_shape[1]
        self.params['cnn_w'] = cnnout_shape[2]
        self.params['cnn_c'] = cnnout_shape[3]
        self.params['cnn_featdim'] = cnnout_shape[1]*cnnout_shape[2]*cnnout_shape[3]

        # lstm params
        shape_w = [o.nunits+self.params['cnn_featdim']/2/2, o.nunits*4] # 2 1x1 conv
        shape_b = [o.nunits*4]
        self.params['w_lstm'] = get_weight_variable(
                shape_w, name_='w_lstm', wd=o.wd)
        self.params['b_lstm'] = get_bias_variable(
                shape_b, name_='w_lstm', wd=o.wd)

        # rnnout params 
        self.params['w_out1'] = get_weight_variable(
                [o.nunits, o.nunits/2], name_='w_out1', wd=o.wd)
        self.params['b_out1'] = get_weight_variable(
                [o.nunits/2], name_='b_out1', wd=o.wd)
        self.params['w_out2'] = get_weight_variable(
                [o.nunits/2, o.outdim], name_='w_out2', wd=o.wd)
        self.params['b_out2'] = get_weight_variable(
                [o.outdim], name_='b_out2', wd=o.wd)

    def _update_params_sim(self, o):
        shape_w = [o.nunits, self.params['cnn_c']] 
        shape_b = [self.params['cnn_c']]
        self.params['w_sim'] = get_weight_variable(
                shape_w, name_='w_sim', wd=o.wd)
        self.params['b_sim'] = get_bias_variable(
                shape_b, name_='b_sim', wd=o.wd)

    def _pass_cnn(self, x_curr, o):
        cnnin = x_curr
        for i in range(self.nlayers):
            x = cnnin if i==0 else act
            conv = conv2d(x, self.params['w_conv'][i], self.params['b_conv'][i], 
                    [1,self.strides[i],self.strides[i],1])
            act = activate(conv, 'relu')
        # TODO: can add dropout here
        #if self.is_train and o.dropout_cnn:
        #    act = tf.nn.dropout(act, o.keep_ratio_cnn)
            if self.is_train and o.dropout_cnn and i==1: # NOTE: maybe only at 2nd layer
                act = tf.nn.dropout(act, o.keep_ratio_cnn)
        return act

    def _pass_cnn_target(self, x0, y0, o):
        # create a binary mask based on y0
        masks = get_masks_from_rectangles(y0, o)
        cnnin = tf.concat_v2((x0, masks),3)
        
        # convolutions
        for i in range(self.nlayers):
            x = cnnin if i==0 else act
            conv = conv2d(x, self.params['w_conv_target'][i], 
                    self.params['b_conv_target'][i], 
                    [1,self.strides[i],self.strides[i],1])
            act = activate(conv, 'relu')
        # TODO: can add dropout here

        # following fc layer; it needs to match the dimension of C
        act_flat = tf.reshape(act, [o.batchsz, -1])
        act_shape = act.get_shape().as_list()
        shape_w_fc_target = [reduce(lambda x, y: x*y, act_shape[1:4]), o.nunits]
        shape_b_fc_target = [o.nunits]
        self.params['w_fc_target'] = get_weight_variable(
                shape_w_fc_target, name_='w_fc_target', wd=o.wd)
        self.params['b_fc_target'] = get_bias_variable(
                shape_b_fc_target, name_='b_fc_target', wd=o.wd)
        feat = activate(tf.matmul(act_flat, self.params['w_fc_target']) 
                + self.params['b_fc_target'], 'relu')
        return feat

    def _pass_cnn_1x1s(self, cnnin, o):
        conv1 = conv2d(cnnin, 
            self.params['w_cellin1'], self.params['b_cellin1'], [1,1,1,1])
        act1 = activate(conv1, 'relu')
        conv2 = conv2d(conv1, 
            self.params['w_cellin2'], self.params['b_cellin2'], [1,1,1,1])
        act2 = activate(conv2, 'relu')
        # TODO: can add dropout here 
        return act2

    def _pass_rnn(self, cellin, h_prev, C_prev, o):
        cellin = tf.reshape(cellin, [o.batchsz, -1]) # flatten

        f_curr, i_curr, C_curr_tilda, o_curr = tf.split(tf.matmul(
            tf.concat_v2((h_prev, cellin), 1), self.params['w_lstm']) + 
            self.params['b_lstm'], 4, 1)

        if o.lstmforgetbias:
            C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        else:
            C_curr = tf.sigmoid(f_curr) * C_prev + \
                    tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
        h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
        return h_curr, C_curr

    def _pass_spatial_attention(self, C_prev, cnnout):
        # fc layer for C_prev to meet the dimension with features  
        context = tf.matmul(C_prev, self.params['w_sim']) + self.params['b_sim']

        # similarity 
        context = tf.expand_dims(tf.expand_dims(context,axis=1),axis=1)
        dotprod = tf.reduce_sum(context*cnnout, axis=3)
        context_norm = tf.sqrt(tf.reduce_sum(tf.pow(context,2),3))
        feat_norm = tf.sqrt(tf.reduce_sum(tf.pow(cnnout, 2), 3))
        e = tf.div(dotprod, (context_norm * feat_norm))

        # attention weight alpha
        alpha = tf.nn.softmax(e)
        
        # compute (expected) attention overlayed context vector z
        # TODO: check other approaches
        z = tf.expand_dims(alpha, 3) * cnnout
        return z

    def _load_model(self, o):
        # indicator whether it's the first segment of fulllength sequence
        #firstseg = tf.placeholder_with_default(False, shape=[], name='firstseg')
        firstseg = tf.placeholder(tf.bool, name='firstseg')

        # placeholders for inputs
        inputs = tf.placeholder(o.dtype, 
                shape=[o.batchsz, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
                name='inputs')
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
        h_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zero or normal
                shape=[o.batchsz, o.nunits], name='h_init')
        C_init = tf.placeholder_with_default(
                tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype), # NOTE: zero or normal
                shape=[o.batchsz, o.nunits], name='C_init')
        # NOTE: make sure to feed C_last to C_init if it's not first segment.

        # extract target feature from {x0,y0} for C; only used if firstseg
        targetfeat = self._pass_cnn_target(inputs[:,0], labels[:,0], o)

        # C_init. Due to tf's weird behavior of 'global_variables_initializer',
        # I can't use condition flow without firstseg being True. 
        C_init = tf.cond(firstseg, lambda: targetfeat, lambda: C_init)

        # RNN unroll
        outputs = []
        for t in range(1, o.ntimesteps+1):
            if t==1:
                h_prev = h_init
                C_prev = C_init
            else:
                h_prev = h_curr
                C_prev = C_curr
            x_curr = inputs[:,t]

            cnnout = self._pass_cnn(x_curr, o)
            if t==1: # some shared params need to be updated here
                self._update_params_rnn(cnnout, o) 
                self._update_params_sim(o)

            cnnout = self._pass_spatial_attention(C_prev, cnnout)

            cnnout = self._pass_cnn_1x1s(cnnout, o)

            h_curr, C_curr = self._pass_rnn(cnnout, h_prev, C_prev, o)

            h_out = tf.matmul(h_curr, self.params['w_out1']) + self.params['b_out1']
            y_curr = tf.matmul(h_out, self.params['w_out2']) + self.params['b_out2']
            outputs.append(y_curr)
        outputs = tf.stack(outputs, axis=1) # list to tensor

        loss = get_loss(outputs, labels, inputs_valid, inputs_HW, o)
        tf.add_to_collection('losses', loss)
        loss_total = tf.add_n(tf.get_collection('losses'),name='loss_total')

        net = {
                'inputs': inputs,
                'inputs_valid': inputs_valid,
                'inputs_HW': inputs_HW,
                'labels': labels,
                'outputs': outputs,
                'loss': loss_total,
                'h_init': h_init,
                'C_init': C_init,
                'h_last': h_curr,
                'C_last': C_curr,
                'firstseg': firstseg,
                }
        return net


def load_model(o):
    if o.model == 'RNN_basic':
        model = RNN_basic(o)
    elif o.model == 'RNN_a':
        model = RNN_a(o)
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

    o.model = 'RNN_a' # RNN_basic, RNN_a 

    if o.model == 'RNN_basic':
        o.yprev_mode = 'concat_channel' # concat_abs, concat_channel
        o.pass_yinit = True
        m = RNN_basic(o)
    elif o.model == 'RNN_a':
        m = RNN_a(o)

    pdb.set_trace()

