import pdb
import tensorflow as tf


class Model_rnn_basic(object):
    def __init__(self, o, loader, is_training=False):
        self._is_training = is_training
        self.loader = loader
        self.net = None
        self.net = self._create_network(o) 

    def _create_network(self, o):
        # TODO: CNN feature later
        # CNN features 
        # variable scope when reusing 

        # TODO: inputs shape can depend on which dataset is being used, thus
        # change it accordingly.
        featdim = self.loader.featdim
        outdim = self.loader.outdim

        inputs = tf.placeholder(o.dtype, shape=[o.batchsz,o.ntimesteps,featdim])
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
        labels = tf.placeholder(o.dtype, shape=[o.batchsz,o.ntimesteps,outdim])

        cell = _get_rnncell(o, is_training=self._is_training)

        cell_outputs, cell_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=o.dtype,
                sequence_length=inputs_length,
                inputs=inputs)

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

def load_model(o, loader, is_training=False):
    if o.model == 'rnn_basic':
        model = Model_rnn_basic(o, loader, is_training=is_training) 
    elif o.model == 'rnn_bidirectional_attention':
        raise ValueError('model not implemeted yet')
    else:
        raise ValueError('model not implemented yet or simply wrong..')
    return model 

if __name__ == '__main__':
    '''
    test model 
    '''
