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
        outdim = self.loader.outdim

        # TODO: if input image is color (ie, has 3 channels), change 
        # dimension of placeholder and convolution operations
        inputs = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, frmsz, frmsz])
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
        labels = tf.placeholder(
                o.dtype, shape=[o.batchsz, o.ntimesteps, outdim])

        use_cnnfeat = True # TODO: make it optionable..
        if use_cnnfeat:
            inputs_cnn = _cnn_filter(inputs, o)
            inputs_cell = tf.reshape(inputs_cnn, [o.batchsz, o.ntimesteps, -1])
        else:
            inputs_cell = tf.reshape(inputs, [o.batchsz, o.ntimesteps, -1])

        cell = _get_rnncell(o, is_training=self._is_training)
        cell_outputs, cell_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=o.dtype,
                sequence_length=inputs_length,
                inputs=inputs_cell)

        cell_outputs = tf.reshape(cell_outputs, [o.batchsz*o.ntimesteps,o.nunits])
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits,outdim], dtype=o.dtype)
        b_out = tf.get_variable(
                'b_out', shape=[o.batchsz*o.ntimesteps,outdim], dtype=o.dtype)
        outputs = tf.matmul(cell_outputs, w_out) + b_out
        outputs = tf.reshape(outputs, [-1, o.ntimesteps, outdim]) # orig. shape
 
        loss = tf.reduce_mean(tf.square(outputs-labels))

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
                'labels': labels,
                'outputs': outputs,
                'loss': loss}
        return net

    def update_network(self, net_new):
        self.net = net_new

class Model_someothermodel(object):
    def __init__(self, o):
        print 'not implemented yet'

def _cnn_filter(inputs, o):
    input_shape = inputs.get_shape().as_list()
    assert(o.batchsz == input_shape[0])
    assert(o.ntimesteps == input_shape[1])
    assert(len(input_shape)==4) # TODO: assuming one channel image; change later 

    # CNN shared
    w_conv1 = _weight_variable([3,3,1,16])
    b_conv1 = _bias_variable([16])
    w_conv2 = _weight_variable([3,3,16,16])
    b_conv2 = _bias_variable([16])
    activations = []
    for t in range(o.ntimesteps):
        x = tf.expand_dims(inputs[:,t],3) 
        conv1 = _conv2d(x, w_conv1, b_conv1, strides_=[1,3,3,1])
        relu1 = _activate(conv1, activation_='relu')
        conv2 = _conv2d(relu1, w_conv2, b_conv2, strides_=[1,3,3,1])
        relu2 = _activate(conv2, activation_='relu')
        activations.append(relu2)
    outputs = tf.pack(activations, axis=1)
    return outputs

    # CNN no shared
    '''
    activations = []
    for t in range(o.ntimesteps):
        w = _weight_variable([3,3,1,16])
        b = _bias_variable([16])
        x = tf.expand_dims(inputs[:,t], 3) # TODO: double check this
        conv = _conv2d(x, w, b, strides_=[1,3,3,1])
        activations.append(_activate(conv, activation_='relu'))
    outputs = tf.pack(activations, axis=1) 
    return outputs
    '''

def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
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
    # basic cell
    if o.cell_type == 'LSTM':
        '''
        cell = tf.nn.rnn_cell.LSTMCell(
                num_units=o.nunits, cell_clip=o.max_grad_norm) \
                if o.grad_clip else tf.nn.rnn_cell.LSTMCell(num_units=o.nunits) 
        '''
        cell = tf.nn.rnn_cell.LSTMCell(num_units=o.nunits)
    elif o.cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(num_units=o.nunits)
    else:
        raise ValueError('cell not implemented yet or simply wrong!')

    # rnn drop out (only during training)
    if is_training and o.dropout_rnn:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=o.keep_ratio)

    # multi-layers
    if o.nlayers > 1:
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
    loader = data.load_data(o)
    m = load_model(o, loader)
    pdb.set_trace()

