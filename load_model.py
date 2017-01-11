import pdb
import tensorflow as tf


class M_rnn_basic(object):
    def __init__(self, o, is_training=False):
        self._is_training = is_training

        self.net = self._create_net(o) 

    def _create_net(self, o):

        '''TEST
        # Create two variables.
        weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
        biases = tf.Variable(tf.zeros([200]), name="biases")
        # ..
        # ..
        # ..
        # after fully constructed the model
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        '''
        
        # CNN feature later
        # Use variable scope (or op scope) if you are reusing things

        inputs = tf.placeholder(o.dtype, shape=[o.batchsz,o.ntimesteps,100*100])
        inputs_length = tf.placeholder(o.dtype, shape=[o.batchsz])
        targets = tf.placeholder(o.dtype, shape=[o.batchsz,o.ntimesteps,4])

        cell = get_rnncell(o, is_training=self._is_training)

        outputs, states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=o.dtype,
                sequence_length=inputs_length,
                inputs=inputs)

        # get logits
        # (in this process, reshape outputs for matrix mutliplication)
        outputs = tf.reshape(outputs, [o.batchsz*o.ntimesteps,o.nunits])
        w_out = tf.get_variable(
                'w_out', shape=[o.nunits,4], dtype=o.dtype)
        b_out = tf.get_variable('b_out', shape=[o.batchsz*o.ntimesteps,4], dtype=o.dtype)
        logits = tf.matmul(outputs, w_out) + b_out
        logits = tf.reshape(logits, [-1, o.ntimesteps, 4]) # back to orig. shape
        
        # loss
        '''cross entropy loss -> wrong
        loss = tf.nn.seq2seq.sequence_loss_by_example(
                [logits],
                [targets],
                [tf.ones([o.batchsz,o.ntimesteps,4], dtype=o.dtype)])
        '''
        loss = tf.reduce_mean(logits-targets)

        net = {
                'inputs': inputs,
                'inputs_length': inputs_length,
                'targets': targets,
                'loss': loss}
        return net

class M_someothermodel(object):
    def __init__(self, o):
        print 'not implemented yet'


def get_rnncell(o, is_training=False):
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

def load_model(o, is_training=False):
    if o.model == 'rnn_basic':
        model = M_rnn_basic(o, is_training=is_training) 
    elif o.model == 'rnn_bidirectional_attention':
        raise ValueError('model not implemeted yet')
    else:
        raise ValueError('model not implemented yet or simply wrong..')
    return model 
