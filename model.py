import pdb
import tensorflow as tf


class Model_rnn_basic(object):
    def __init__(self, o, loader, is_training=False):
        self._is_training = is_training
        self.loader = loader
        self.net = None
        self.net = self._create_network(o) 

    def _create_network(self, o):
        # TODO: variable scope when reusing 
        # tf.Variable() vs. tf.get_variable()

        # this is easy way otherwise need to specify dataset name too
        frmsz = self.loader.frmsz
        featdim = self.loader.featdim
        ninchannel = self.loader.ninchannel # TODO: use this
        outdim = self.loader.outdim

        if ninchannel == 1:
            inputs_shape = [o.batchsz, o.ntimesteps, frmsz, frmsz]
        else:
            inputs_shape = [o.batchsz, o.ntimesteps, frmsz, frmsz, ninchannel]

        inputs = tf.placeholder(o.dtype, shape=inputs_shape)
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
        labels = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, outdim])

        use_cnnfeat = True# TODO: make it optionable..
        if o.cnn_pretrain: # use pre-trained model
            raise ValueError('not implemented yet')
        else: 
            if use_cnnfeat: # train from scratch
                inputs_cnn = _cnn_filter(inputs, o)
                inputs_cell = tf.reshape(inputs_cnn,[o.batchsz,o.ntimesteps,-1])
            else: # no use of cnn
                inputs_cell = tf.reshape(inputs, [o.batchsz, o.ntimesteps, -1])

        if o.usetfapi:
            cell = _get_rnncell(o, is_training=self._is_training)
            cell_outputs, cell_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    dtype=o.dtype,
                    sequence_length=inputs_length,
                    inputs=inputs_cell)
        else: # manual rnn module
            label_init = labels[:,0]
            cell_outputs = _rnn_pass( # TODO: work on passing gt init
                    inputs_cell, label_init, o, is_training=self._is_training)

        cell_outputs = tf.reshape(cell_outputs, [o.batchsz*o.ntimesteps,o.nunits])
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits,outdim], dtype=o.dtype)
        b_out = tf.get_variable(
                'b_out', shape=[o.batchsz*o.ntimesteps,outdim], dtype=o.dtype)
        outputs = tf.matmul(cell_outputs, w_out) + b_out
        outputs = tf.reshape(outputs, [-1, o.ntimesteps, outdim]) # orig. shape
 
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


class Model_someothermodel(object):
    def __init__(self, o):
        print 'not implemented yet'


def _rnn_pass(inputs, label_init, o, is_training=False):
    ''' This is a (manually designed) rnn unrolling function.
    Note that this is not supposed to reproduce the same (or similar) results
    that one would produce from using tensorflow API.
    One clear distinction is that this module uses the ground-truth label for
    the first frame; that is specific for tracking task. 
    Also, besides the previous hidden state and current input, rnn cell receives
    the previous output. How we are using this is undecided or unexplored yet.
    '''

    def _get_lstm_params(nunits, featdim):
        shape_W = [nunits+featdim, nunits]
        shape_b = [nunits]
        params = {}
        with tf.variable_scope('LSTMcell'):
            params['Wf'] = tf.get_variable('Wf', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer()) 
            params['Wi'] = tf.get_variable('Wi', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer())
            params['Wc'] = tf.get_variable('Wc', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer())
            params['Wo'] = tf.get_variable('Wo', shape=shape_W, dtype=o.dtype, 
                    initializer=tf.truncated_normal_initializer())
            params['bf'] = tf.get_variable('bf', shape=shape_b, dtype=o.dtype, 
                    initializer=tf.constant_initializer())
            params['bi'] = tf.get_variable('bi', shape=shape_b, dtype=o.dtype, 
                    initializer=tf.constant_initializer())
            params['bc'] = tf.get_variable('bc', shape=shape_b, dtype=o.dtype, 
                    initializer=tf.constant_initializer())
            params['bo'] = tf.get_variable('bo', shape=shape_b, dtype=o.dtype, 
                    initializer=tf.constant_initializer())
        return params

    #def _activate_rnncell(x_curr, h_prev, y_prev, C_prev, params, o):
    def _activate_rnncell(x_curr, h_prev, C_prev, params, o):
        if o.cell_type == 'LSTM': # standard LSTM (no peephole)
            Wf = params['Wf']
            Wi = params['Wi']
            Wc = params['Wc']
            Wo = params['Wo']
            bf = params['bf']
            bi = params['bi']
            bc = params['bc']
            bo = params['bo']

            # forget, input, memory cell, output, hidden 
            if o.tfversion == '0.12': # TODO: remove once upgrade to 0.12
                f_curr = tf.sigmoid(tf.matmul(tf.concat_v2((h_prev,x_curr),1), Wf) + bf)
                i_curr = tf.sigmoid(tf.matmul(tf.concat_v2((h_prev,x_curr),1), Wi) + bi )
                C_curr_tilda = tf.tanh(tf.matmul(tf.concat_v2((h_prev,x_curr),1), Wc) + bc)
                C_curr = tf.multiply(f_curr, C_prev) + tf.multiply(i_curr, C_curr_tilda)
                o_curr = tf.sigmoid(tf.matmul(tf.concat_v2((h_prev,x_curr),1), Wo) + bo )
                h_curr = tf.multiply(o_curr, tf.tanh(C_curr)) 
            elif o.tfversion == '0.11':
                f_curr = tf.sigmoid(tf.matmul(tf.concat(1,(h_prev,x_curr)), Wf) + bf)
                i_curr = tf.sigmoid(tf.matmul(tf.concat(1,(h_prev,x_curr)), Wi) + bi )
                C_curr_tilda = tf.tanh(tf.matmul(tf.concat(1,(h_prev,x_curr)), Wc) + bc)
                C_curr = tf.mul(f_curr, C_prev) + tf.mul(i_curr, C_curr_tilda)
                o_curr = tf.sigmoid(tf.matmul(tf.concat(1,(h_prev,x_curr)), Wo) + bo )
                h_curr = tf.mul(o_curr, tf.tanh(C_curr)) 
            else:
                raise ValueError('no avaialble tensorflow version')
        elif o.cell_type == 'LSTM_variant': # peephole or coupled input/forget
            raise ValueError('Not implemented yet!')
        else:
            raise ValueError('Not implemented yet!')
        return h_curr, C_curr

    # lstm weight and bias parameters
    params = _get_lstm_params(o.nunits, inputs.get_shape().as_list()[-1])

    hs = []
    for t in range(o.ntimesteps): # TODO: change if variable length input/out
        if t == 0:
            h_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype) #TODO: correct?
            #y_prev = label_init # TODO: use output!
            C_prev = tf.zeros([o.batchsz, o.nunits], dtype=o.dtype)
        x_curr = inputs[:,t] 
        #h_prev, C_prev = _activate_rnncell(x_curr, h_prev, y_prev, C_prev, o) 
        h_prev, C_prev = _activate_rnncell(x_curr, h_prev, C_prev, params, o) #TODO: use output!
        hs.append(h_prev)

    # list to tensor
    if o.tfversion == '0.12': 
        hs = tf.stack(hs, axis=1)
    elif o.tfversion == '0.11':
        hs = tf.pack(hs, axis=1)
    return hs 

def _cnn_filter(inputs, o):
    input_shape = inputs.get_shape().as_list()
    assert(o.batchsz == input_shape[0])
    assert(o.ntimesteps == input_shape[1])
    assert(len(input_shape)==4) # TODO: assuming one channel image; change later 

    # CNN shared
    w_conv1 = _weight_variable([3,3,1,16], o, wd=o.wd)
    b_conv1 = _bias_variable([16])
    w_conv2 = _weight_variable([3,3,16,16], o, wd=o.wd)
    b_conv2 = _bias_variable([16])
    activations = []
    for t in range(o.ntimesteps):
        x = tf.expand_dims(inputs[:,t],3) 
        conv1 = _conv2d(x, w_conv1, b_conv1, strides_=[1,3,3,1]) # TODO: low st
        relu1 = _activate(conv1, activation_='relu')
        conv2 = _conv2d(relu1, w_conv2, b_conv2, strides_=[1,3,3,1]) #TODO: low st
        #conv2 = _conv2d(relu1, w_conv2, b_conv2, strides_=[1,1,1,1]) #TODO: low st
        relu2 = _activate(conv2, activation_='relu')
        activations.append(relu2)
    if o.tfversion == '0.12':
        outputs = tf.stack(activations, axis=1)
    elif o.tfversion == '0.11':
        outputs = tf.pack(activations, axis=1)
    return outputs

    # CNN no shared
    '''
    activations = []
    for t in range(o.ntimesteps):
        w = _weight_variable([3,3,1,16], o, wd=o.wd)
        b = _bias_variable([16])
        x = tf.expand_dims(inputs[:,t], 3) # TODO: double check this
        conv = _conv2d(x, w, b, strides_=[1,3,3,1])
        activations.append(_activate(conv, activation_='relu'))
    if o.tfversion == '0.12':
        outputs = tf.stack(activations, axis=1)
    elif o.tfversion == '0.11':
        outputs = tf.pack(activations, axis=1) 
    return outputs
    '''

def _weight_variable(shape, o, wd=0.0):
    initial = tf.truncated_normal(shape, stddev=0.1)
    #if wd is not none:
    if o.tfversion == '0.12':
        weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd, name='weight_loss')
    elif o.tfversion == '0.11':
        weight_decay = tf.mul(tf.nn.l2_loss(initial), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, w, b, strides_):
    return tf.nn.conv2d(x, w, strides=strides_, padding='SAME') + b

def _activate(input_, activation_='relu'):
    if activation_ == 'relu':
        return tf.nn.relu(input_)
    elif activation_ == 'tanh':
        return tf.nn.tanh(input_)
    elif activation_ == 'sigm':
        return tf.nn.sigmoid(input_)
    elif activation_ == 'linear': # no activation!
        return input_
    else:
        raise ValueError('no available activation type!')

def _get_rnncell(o, is_training=False):
    # TODO: currently tensorflow changes rnn cells back to contrib. This should
    # be treated better instead of changing based on version..

    # basic cell
    if o.cell_type == 'LSTM':
        '''
        cell = tf.nn.rnn_cell.LSTMCell(
                num_units=o.nunits, cell_clip=o.max_grad_norm) \
                if o.grad_clip else tf.nn.rnn_cell.LSTMCell(num_units=o.nunits) 
        '''
        if o.tfversion == '0.12':
            cell = tf.contrib.rnn.LSTMCell(num_units=o.nunits)
        elif o.tfversion == '0.11':
            cell = tf.nn.rnn_cell.LSTMCell(num_units=o.nunits)
    elif o.cell_type == 'GRU':
        if o.tfversion == '0.12':
            cell = tf.contrib.rnn.GRUCell(num_units=o.nunits)
        elif o.tfversion == '0.11':
            cell = tf.nn.rnn_cell.GRUCell(num_units=o.nunits)
    else:
        raise ValueError('cell not implemented yet or simply wrong!')

    # rnn drop out (only during training)
    if is_training and o.dropout_rnn:
        if o.tfversion == '0.12':
            cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=o.keep_ratio)
        elif o.tfversion == '0.11':
            cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=o.keep_ratio)

    # multi-layers
    if o.nlayers > 1:
        if o.tfversion == '0.12':
            cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*o.nlayers)
        elif o.tfversion == '0.11':
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell]*o.nlayers)

    return cell

def load_model(o, loader):
    is_training = True if o.mode == 'train' else False
    # TODO: check by actually logging device placement!
    with tf.device('/{}:{}'.format(o.device, o.device_number)):
        if o.model == 'rnn_basic':
            model = Model_rnn_basic(o, loader, is_training=is_training) 
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
    loader = data.load_data(o)
    m = load_model(o, loader)
    pdb.set_trace()

