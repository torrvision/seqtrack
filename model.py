import pdb
import tensorflow as tf


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
            def _weight_variable(shape, wd=0.0): # TODO: this can be used global
                initial = tf.truncated_normal(shape, stddev=0.1)
                weight_decay = tf.nn.l2_loss(initial) * wd
                tf.add_to_collection('losses', weight_decay)
                return tf.Variable(initial)
            def _bias_variable(shape): # TODO: this can be used global
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
        for t in range(self.ntimesteps):
            for i in range(self.nlayers):
                x = tf.expand_dims(inputs[:,t],3) if i==0 else relu
                st = self.strides[i]
                # TODO: convolution different stride at each layer
                conv = _conv2d(x, w_conv[i], b_conv[i], strides_=[1,st,st,1])
                relu = _activate(conv, activation_='relu')
            activations.append(relu)
        outputs = tf.stack(activations, axis=1)
        return outputs


class Model_rnn_basic(object):
    def __init__(self, o, is_training=False):
        self._is_training = is_training
        self.net = None
        self.net = self._create_network(o)

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
        use_cnnfeat = True # TODO: will be deprecated. always true. 
        if o.cnn_pretrain: # use pre-trained model
            raise ValueError('not implemented yet')
        else: 
            if use_cnnfeat: # train from scratch
                cnn = CNN(o)
                inputs_cnn = cnn.create_network(inputs)
                inputs_cell = tf.reshape(inputs_cnn,[o.batchsz,o.ntimesteps,-1])
            else: # no use of cnn
                inputs_cell = tf.reshape(inputs, [o.batchsz, o.ntimesteps, -1])

        # RNN
        if o.usetfapi: # TODO: will (and should) be deprecated. 
            outputs = self._rnn_pass_API(inputs_cell, inputs_length, o)
        else: # manual rnn module
            #outputs = self._rnn_pass(inputs_cell, labels[:,0], o)
            outputs = self._rnn_pass_wip(inputs_cell, labels[:,0], o) # TODO: WIP
 
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

    def _rnn_pass_wip(self, inputs, label_init, o):
        def _get_lstm_params(nunits, featdim, outdim):
            #shape_W = [nunits+featdim, nunits*4] 
            shape_W = [nunits+featdim+outdim, nunits*4] # TODO: lots of variants
            shape_b = [nunits*4]
            params = {}
            with tf.variable_scope('LSTMcell'):
                params['W'] = tf.get_variable('W', shape=shape_W, dtype=o.dtype, 
                        initializer=tf.truncated_normal_initializer(stddev=0.001)) #TODO: 0.0001
                params['b'] = tf.get_variable('b', shape=shape_b, dtype=o.dtype, 
                        initializer=tf.constant_initializer())
            return params

        def _activate_rnncell(x_curr, h_prev, C_prev, y_prev, params, o):
            if o.cell_type == 'LSTM': # standard LSTM (no peephole)
                W = params['W']
                b = params['b']
                # forget, input, memory cell, output, hidden 
                #f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                        #tf.matmul(tf.concat_v2((h_prev,x_curr),1), W) + b, 4, 1)
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                        tf.matmul(tf.concat_v2((h_prev,x_curr,y_prev),1), W) + b
                        , 4, 1)
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
                # TODO: a version with forget bias same as tensorflow LSTMCell
                #C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        #tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
                h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            elif o.cell_type == 'LSTM_variant': # coupled input and forget gates
                raise ValueError('Not implemented yet!')
            else:
                raise ValueError('Not implemented yet!')
            return h_curr, C_curr

        params = _get_lstm_params(o.nunits, inputs.get_shape().as_list()[-1], o.outdim)
        
        # initial states
        # TODO: normal init state? -> seems zero state is okay. no difference.
        h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        y_prev = tf.zeros([o.batchsz, o.outdim], dtype=o.dtype)

        # TODO: check if these variables are shared or not
        # case1: variable shared across timestep
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits, o.outdim], dtype=o.dtype, 
                initializer=tf.truncated_normal_initializer()) # TODO stddev
        b_out = tf.get_variable(
                'b_out', shape=[o.outdim], dtype=o.dtype, 
                initializer=tf.constant_initializer())

        # unroll
        outputs = []
        for t in range(o.ntimesteps): # TODO: change if variable length input/out
            h_prev, C_prev = _activate_rnncell(inputs[:,t], h_prev, C_prev, y_prev, params, o) 
            if t == 0: # no matter what the output is, pass the ground truth to t=2
                y_prev = label_init
            else:
                y_prev = tf.matmul(h_prev, w_out) + b_out
            outputs.append(y_prev)
            
        # list to tensor
        outputs = tf.stack(outputs, axis=1)
        return outputs 

    def _rnn_pass(self, inputs, label_init, o):
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

        #def _activate_rnncell(x_curr, h_prev, y_prev, C_prev, params, o):
        def _activate_rnncell(x_curr, h_prev, C_prev, params, o):
            if o.cell_type == 'LSTM': # standard LSTM (no peephole)
                W = params['W']
                b = params['b']
                # forget, input, memory cell, output, hidden 
                f_curr, i_curr, C_curr_tilda, o_curr = tf.split(
                        tf.matmul(tf.concat_v2((h_prev,x_curr),1), W) + b, 4, 1)
                C_curr = tf.sigmoid(f_curr) * C_prev + \
                        tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
                # TODO: a version with forget bias same as tensorflow LSTMCell
                #C_curr = tf.sigmoid(f_curr + 1.0) * C_prev + \
                        #tf.sigmoid(i_curr) * tf.tanh(C_curr_tilda) 
                h_curr = tf.sigmoid(o_curr) * tf.tanh(C_curr)
            elif o.cell_type == 'LSTM_variant': # coupled input and forget gates
                raise ValueError('Not implemented yet!')
            else:
                raise ValueError('Not implemented yet!')
            return h_curr, C_curr

        # lstm weight and bias parameters
        params = _get_lstm_params(o.nunits, inputs.get_shape().as_list()[-1])
        
        # initial states
        # TODO: normal init state? -> seems zero state is okay. no difference.
        h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        #h_prev = tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype)
        #C_prev = tf.truncated_normal([o.batchsz, o.nunits], dtype=o.dtype)
        #y_prev = label_init # TODO: use output!

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
        pdb.set_trace()
        outputs = tf.matmul(cell_outputs, w_out) + b_out
        outputs = tf.reshape(outputs, [-1, o.ntimesteps, o.outdim]) # orig. shape
        return outputs 

    def _rnn_pass_API(self, inputs_cell, inputs_length, o):
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


class Model_someothermodel(object):
    def __init__(self, o):
        print 'not implemented yet'


def load_model(o):
    is_training = True if o.mode == 'train' else False
    # TODO: check by actually logging device placement!
    with tf.device('/{}:{}'.format(o.device, o.device_number)):
        if o.model == 'rnn_basic':
            model = Model_rnn_basic(o, is_training=is_training) 
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
    o.dataset = 'bouncing_mnist'
    o.batchsz = 20
    o.frmsz = 100
    o.ninchannel = 1
    o.outdim = 4
    #o.usetfapi = True
    m = load_model(o)
    pdb.set_trace()

