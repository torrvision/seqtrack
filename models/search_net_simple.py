import tensorflow as tf


def search_net_simple(inputs, o,
        summaries_collections=None,
        # Model parameters:
        # lstm_depth=1,
        ):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
        # Process initial image and label to get "template".
        with tf.variable_scope('template'):
            p0 = get_masks_from_rectangles(inputs['y0'], o)
            first_image_with_mask = concat([inputs['x0'], p0], axis=3)
            template = template_net(first_image_with_mask)
        # Process all images from all sequences with feature net.
        with tf.variable_scope('features'):
            x, unmerge = merge_dims(inputs['x'], 0, 2)
            feat = feat_net(x)
            feat = unmerge(feat, 0)
        # Search each image using result of template network.
        response = search(feat, template)

        with tf.variable_scope('process') as scope:
            init_state = initial_state_net(inputs['y0'])
            curr_state = init_state
            response = tf.unstack(response, axis=1)
            output = [None] * o.ntimesteps
            for t in range(o.ntimesteps):
                output[t], curr_state = process_response(response[t], curr_state)
                output.append(out)
                scope.reuse_variables()
            output = tf.stack(output, axis=1)
    return output

def feat_net(x):
    with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='SAME'):
        # conv1
        x = slim.conv2d(x, 64, 11, stride=4)
        x = slim.max_pool2d(x)
        # conv2
        x = slim.conv2d(x, 128, 5)
        x = slim.max_pool2d(x)
        # conv3
        x = slim.conv2d(x, 192, 3)
        # conv4
        x = slim.conv2d(x, 192, 3)
        # conv5
        # TODO: Keep RELU here?
        x = slim.conv2d(x, 128, 3)
    return x

def template_net(x):
    x = feat_net(x)
    x = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return x

def search(x, f):
    # x.shape is [b, t, hx, wx, c]
    # f.shape is [b, hf, wf, c] = [b, 1, 1, c]
    # linear(concat(a, b)) = linear(a) + linear(b)
    f = slim.conv2d(f, 256, 3)
    f = tf.expand_dims(f, 1)
    x, unmerge = merge_dims(x, 0, 2)
    x = slim.conv2d(x, 256, 3)
    x = unmerge(x, 0)
    x = tf.plus(x, f)
    return x

def initial_state_net(x):
    x = slim.fully_connected(
    return x

def process_response(x, prev_state):
    '''Convert response maps to rectangle.'''
    h = prev_state['h']
    c = prev_state['c']
    h, c = conv_lstm(x, h, c, 256)
    x = h
    state = {'h': h, 'c': c}
    with slim.arg_scope([slim.max_pool2d], kernel_size=3, padding='VALID'):
        x = slim.conv2d(x, 16, 3)
        x = slim.max_pool2d(x)
        x = slim.conv2d(x, 32, 3)
        x = slim.max_pool2d(x)
        x = slim.conv2d(x, 1024, 3)
        x = slim.max_pool2d(x)
        x = slim.conv2d(x, 1024, 3)
    return x, state

    # model parameters
    self.lstm1_nlayers = lstm1_nlayers
    self.lstm2_nlayers = lstm2_nlayers
    self.use_cnn3      = use_cnn3
    self.pass_hmap     = pass_hmap
    self.dropout_rnn   = dropout_rnn
    self.dropout_cnn   = dropout_cnn
    self.keep_prob     = keep_prob
    self.init_memory   = init_memory
    # Ignore sumaries_collections - model does not generate any summaries.
    self.outputs, self.state, self.memory, self.dbg = self._load_model(inputs, o)
    self.image_size   = (o.frmsz, o.frmsz)
    self.sequence_len = o.ntimesteps
    self.batch_size   = o.batchsz

    x           = inputs['x']  # shape [b, ntimesteps, h, w, 3]
    x0          = inputs['x0'] # shape [b, h, w, 3]
    y0          = inputs['y0'] # shape [b, 4]
    y           = inputs['y']  # shape [b, ntimesteps, 4]
    use_gt      = inputs['use_gt']
    gt_ratio    = inputs['gt_ratio']
    is_training = inputs['is_training']

    # Add identity op to ensure that we can feed state here.
    x_init = tf.identity(x0)
    y_init = tf.identity(y0)
    hmap_init = tf.identity(get_masks_from_rectangles(y0, o, kind='bg'))

    # lstm initial memory states. {random or CNN}.
    h1_init = [None] * self.lstm1_nlayers
    c1_init = [None] * self.lstm1_nlayers
    h2_init = [None] * self.lstm2_nlayers
    c2_init = [None] * self.lstm2_nlayers
    if not self.init_memory:
        with tf.name_scope('lstm_initial'):
            with slim.arg_scope([slim.model_variable],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    regularizer=slim.l2_regularizer(o.wd)):
                for i in range(self.lstm1_nlayers):
                    h1_init_single = slim.model_variable('lstm1_h_init_{}'.format(i+1), shape=[o.nunits])
                    c1_init_single = slim.model_variable('lstm1_c_init_{}'.format(i+1), shape=[o.nunits])
                    h1_init[i] = tf.stack([h1_init_single] * o.batchsz)
                    c1_init[i] = tf.stack([c1_init_single] * o.batchsz)
                for i in range(self.lstm2_nlayers):
                    h2_init_single = slim.model_variable('lstm2_h_init_{}'.format(i+1), shape=[81, 81, 2]) # TODO: adaptive
                    c2_init_single = slim.model_variable('lstm2_c_init_{}'.format(i+1), shape=[81, 81, 2])
                    h2_init[i] = tf.stack([h2_init_single] * o.batchsz)
                    c2_init[i] = tf.stack([c2_init_single] * o.batchsz)
    else:
        with tf.name_scope('lstm_initial'):
            # lstm1
            hmap_from_rec = get_masks_from_rectangles(y_init, o)
            if self.pass_hmap:
                xy = concat([x_init, hmap_from_rec, hmap_init], axis=3)
                xy = tf.stop_gradient(xy)
            else:
                xy = concat([x_init, hmap_from_rec], axis=3)
            for i in range(self.lstm1_nlayers):
                with tf.variable_scope('lstm1_layer_{}'.format(i+1)):
                    with tf.variable_scope('h_init'):
                        h1_init[i] = pass_cnn2(xy, o.nunits)
                    with tf.variable_scope('c_init'):
                        c1_init[i] = pass_cnn2(xy, o.nunits)
            # lstm2
            for i in range(self.lstm2_nlayers):
                with tf.variable_scope('lstm2_layer_{}'.format(i+1)):
                    with tf.variable_scope('h_init'):
                        h2_init[i] = pass_init_lstm2(hmap_init)
                    with tf.variable_scope('c_init'):
                        c2_init[i] = pass_init_lstm2(hmap_init)

    with tf.name_scope('noise'):
        noise = tf.truncated_normal(tf.shape(y), mean=0.0, stddev=0.05,
                                    dtype=o.dtype, seed=o.seed_global, name='noise')


    x_prev = x_init
    y_prev = y_init
    hmap_prev = hmap_init
    h1_prev, c1_prev = h1_init, c1_init
    h2_prev, c2_prev = h2_init, c2_init

    y_pred = []
    hmap_pred = []
    memory_h2 = []
    memory_c2 = []

    for t in range(o.ntimesteps):
        x_curr = x[:, t]
        y_curr = y[:, t]
        with tf.name_scope('cnn1_{}'.format(t)) as scope:
            with tf.variable_scope('cnn1', reuse=(t > 0)):
                cnn1out = pass_cnn1(x_curr, scope)

        with tf.name_scope('cnn2_{}'.format(t)) as scope:
            with tf.variable_scope('cnn2', reuse=(t > 0)):
                # use both `hmap_prev` along with `y_prev_{GT or pred}`
                hmap_from_rec = get_masks_from_rectangles(y_prev, o)
                if self.pass_hmap:
                    xy = concat([x_prev, hmap_from_rec, hmap_prev], axis=3) # TODO: backpropagation-able?
                    xy = tf.stop_gradient(xy)
                else:
                    xy = concat([x_prev, hmap_from_rec], axis=3)
                cnn2out = pass_cnn2(xy, name=scope)

        if self.use_cnn3:
            with tf.name_scope('cnn3_{}'.format(t)) as scope:
                with tf.variable_scope('cnn3', reuse=(t > 0)):
                    cnn3out = pass_cnn3(tf.concat([x_prev, x_curr], axis=3), scope)

        h1_curr = [None] * self.lstm1_nlayers
        c1_curr = [None] * self.lstm1_nlayers
        with tf.name_scope('lstm1_{}'.format(t)) as scope:
            with tf.variable_scope('lstm1', reuse=(t > 0)):
                input_to_lstm1 = tf.identity(cnn2out)
                for i in range(self.lstm1_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h1_curr[i], c1_curr[i] = pass_lstm1(input_to_lstm1, h1_prev[i], c1_prev[i], scope)
                    if self.dropout_rnn:
                        input_to_lstm1 = slim.dropout(h1_curr[i],
                                                      keep_prob=self.keep_prob,
                                                      is_training=is_training, scope='dropout')
                    else:
                        input_to_lstm1 = h1_curr[i]


        with tf.name_scope('multi_level_cross_correlation_{}'.format(t)) as scope:
            with tf.variable_scope('multi_level_cross_correlation', reuse=(t > 0)):
                scoremap = pass_multi_level_cross_correlation(cnn1out, h1_curr[-1], scope) # multi-layer lstm1

        if self.use_cnn3:
            with tf.name_scope('multi_level_integration_correlation_and_flow_{}'.format(t)) as scope:
                with tf.variable_scope('multi_level_integration_correlation_and_flow', reuse=(t > 0)):
                    scoremap = pass_multi_level_integration_correlation_and_flow(
                            scoremap, cnn3out, scope)

        with tf.name_scope('multi_level_deconvolution_{}'.format(t)) as scope:
            with tf.variable_scope('multi_level_deconvolution', reuse=(t > 0)):
                scoremap = pass_multi_level_deconvolution(scoremap, scope)

        with tf.name_scope('cnn_out_hmap_{}'.format(t)) as scope:
            with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                hmap_curr = pass_out_heatmap(scoremap, scope)

        h2_curr = [None] * self.lstm2_nlayers
        c2_curr = [None] * self.lstm2_nlayers
        with tf.name_scope('lstm2_{}'.format(t)) as scope:
            with tf.variable_scope('lstm2', reuse=(t > 0)):
                input_to_lstm2 = tf.identity(scoremap)
                for i in range(self.lstm2_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h2_curr[i], c2_curr[i] = pass_lstm2(input_to_lstm2, h2_prev[i], c2_prev[i], scope)
                    if self.dropout_rnn:
                        input_to_lstm2 = slim.dropout(h2_curr[i],
                                                      keep_prob=self.keep_prob,
                                                      is_training=is_training, scope='dropout')
                    else:
                        input_to_lstm2 = h2_curr[i]

        with tf.name_scope('cnn_out_rec_{}'.format(t)) as scope:
            with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                if self.lstm2_nlayers > 0:
                    y_curr_pred = pass_out_rectangle(h2_curr[-1], scope) # multi-layer lstm2
                else:
                    y_curr_pred = pass_out_rectangle(scoremap, scope) # No LSTM2

        #with tf.name_scope('cnn_out_hmap_{}'.format(t)) as scope:
        #    with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
        #        hmap_curr = pass_out_heatmap(h2_curr[-1], scope) # multi-layer lstm2

        x_prev = x_curr
        rand_prob = tf.random_uniform([], minval=0, maxval=1)
        gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
        y_prev = tf.cond(gt_condition, lambda: y_curr + noise[:,t], # TODO: should noise be gone?
                                       lambda: y_curr_pred)
        h1_prev, c1_prev = h1_curr, c1_curr
        h2_prev, c2_prev = h2_curr, c2_curr
        hmap_prev = hmap_curr

        y_pred.append(y_curr_pred)
        hmap_pred.append(hmap_curr)
        memory_h2.append(h2_curr[-1] if self.lstm2_nlayers > 0 else None)
        memory_c2.append(c2_curr[-1] if self.lstm2_nlayers > 0 else None)

    y_pred = tf.stack(y_pred, axis=1) # list to tensor
    hmap_pred = tf.stack(hmap_pred, axis=1)
    if self.lstm2_nlayers > 0:
        memory_h2 = tf.stack(memory_h2, axis=1)
        memory_c2 = tf.stack(memory_c2, axis=1)

    outputs = {'y': y_pred, 'hmap': hmap_pred}
    state = {}
    state.update({'h1_{}'.format(i+1): (h1_init[i], h1_curr[i]) for i in range(self.lstm1_nlayers)})
    state.update({'c1_{}'.format(i+1): (c1_init[i], c1_curr[i]) for i in range(self.lstm1_nlayers)})
    state.update({'h2_{}'.format(i+1): (h2_init[i], h2_curr[i]) for i in range(self.lstm2_nlayers)})
    state.update({'c2_{}'.format(i+1): (c2_init[i], c2_curr[i]) for i in range(self.lstm2_nlayers)})
    state.update({'x': (x_init, x_prev), 'y': (y_init, y_prev)})
    state.update({'hmap': (hmap_init, hmap_prev)})
    memory = {'h2': memory_h2, 'c2': memory_c2}

    #dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
    dbg = {}
    return outputs, state, memory, dbg


def _load_model(self, inputs, o):

def pass_init_lstm2(x, name='pass_init_lstm2'):
    ''' CNN for memory states in lstm2. Used to initialize.
    '''
    with tf.name_scope(name):
        with slim.arg_scope([slim.conv2d],
                weights_regularizer=slim.l2_regularizer(o.wd)):
            x = slim.conv2d(x, 2, [7, 7], stride=3, scope='conv1')
            x = slim.conv2d(x, 2, [1, 1], stride=1, activation_fn=None, scope='conv2')
    return x

def pass_cnn1(x, name):
    ''' CNN for search space
    '''
    out = []
    with tf.name_scope(name):
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
    return out

def pass_cnn2(x, outsize=1024, name='pass_cnn2'):
    ''' CNN for appearance
    '''
    with tf.name_scope(name):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(o.wd)):
            x = slim.conv2d(x, 16, [7, 7], stride=3, scope='conv1')
            x = slim.conv2d(x, 32, [5, 5], stride=2, scope='conv2')
            x = slim.max_pool2d(x, 2, scope='pool1')
            x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv3')
            x = slim.conv2d(x, 64, [3, 3], stride=1, scope='conv4')
            x = slim.max_pool2d(x, 2, scope='pool2')
            x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv5')
            x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv6')
            x = slim.conv2d(x, 128, [3, 3], stride=1, scope='conv7')
            x = slim.max_pool2d(x, 2, scope='pool3')
            x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv8')
            x = slim.conv2d(x, 256, [3, 3], stride=1, scope='conv9')
            x = slim.flatten(x)
            x = slim.fully_connected(x, 1024, scope='fc1')
            if self.dropout_cnn:
                x = slim.dropout(x, keep_prob=self.keep_prob, is_training=is_training, scope='dropout1')
            x = slim.fully_connected(x, outsize, scope='fc2')
    return x

def pass_cnn3(x, name):
    ''' CNN for flow
    '''
    out = []
    with tf.name_scope(name):
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
    return out

def pass_lstm1(x, h_prev, c_prev, name):
    with tf.name_scope(name):
        with slim.arg_scope([slim.fully_connected],
                num_outputs=o.nunits,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(o.wd)):
            # NOTE: `An Empirical Exploration of Recurrent Neural Network Architecture`.
            # Initialize forget bias to be 1.
            # They also use `tanh` instead of `sigmoid` for input gate. (yet not employed here)
            ft = slim.fully_connected(concat((h_prev, x), 1), biases_initializer=tf.ones_initializer(), scope='hf')
            it = slim.fully_connected(concat((h_prev, x), 1), scope='hi')
            ct_tilda = slim.fully_connected(concat((h_prev, x), 1), scope='hc')
            ot = slim.fully_connected(concat((h_prev, x), 1), scope='ho')
            ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
            ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
    return ht, ct

def pass_multi_level_cross_correlation(search, filt, name):
    ''' Multi-level cross-correlation function producing scoremaps.
    Option 1: depth-wise convolution
    Option 2: similarity score (-> doesn't work well)
    Note that depth-wise convolution with 1x1 filter is actually same as
    channel-wise (and element-wise) multiplication.
    '''
    # TODO: sigmoid or softmax over scoremap?
    # channel-wise l2 normalization as in Universal Correspondence Network?
    scoremap = []
    with tf.name_scope(name):
        with slim.arg_scope([slim.fully_connected],
                weights_regularizer=slim.l2_regularizer(o.wd)):
            for i in range(len(search)):
                depth = search[i].shape.as_list()[-1]
                scoremap.append(search[i] *
                        tf.expand_dims(tf.expand_dims(slim.fully_connected(filt, depth), 1), 1))
    return scoremap

def pass_multi_level_integration_correlation_and_flow(correlation, flow, name):
    ''' Multi-level integration of correlation and flow outputs.
    Using sum.
    '''
    with tf.name_scope(name):
        scoremap = [correlation[i]+flow[i] for i in range(len(correlation))]
    return scoremap

def pass_multi_level_deconvolution(x, name):
    ''' Multi-level deconvolutions.
    This is in a way similar to HourglassNet.
    Using sum.
    '''
    deconv = x[-1]
    with tf.name_scope(name):
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

def pass_lstm2(x, h_prev, c_prev, name):
    ''' ConvLSTM
    h and c have the same spatial dimension as x.
    '''
    # TODO: increase size of hidden
    with tf.name_scope(name):
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

def pass_out_rectangle(x, name):
    ''' Regress output rectangle.
    '''
    with tf.name_scope(name):
        with slim.arg_scope([slim.fully_connected, slim.conv2d],
                weights_regularizer=slim.l2_regularizer(o.wd)):
            if not self.lstm2_nlayers > 0:
                x = slim.conv2d(x, 2, 1, scope='conv1')
            x = slim.flatten(x)
            x = slim.fully_connected(x, 1024, scope='fc1')
            x = slim.fully_connected(x, 1024, scope='fc2')
            x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
    return x

def pass_out_heatmap(x, name):
    ''' Upsample and generate spatial heatmap.
    '''
    with tf.name_scope(name):
        with slim.arg_scope([slim.conv2d],
                #num_outputs=x.shape.as_list()[-1],
                num_outputs=2, # NOTE: hmap before lstm2 -> reduce the output channel to 2 here.
                weights_regularizer=slim.l2_regularizer(o.wd)):
            x = slim.conv2d(tf.image.resize_images(x, [241, 241]),
                            kernel_size=[3, 3], scope='deconv')
            x = slim.conv2d(x, kernel_size=[1, 1], scope='conv1')
            x = slim.conv2d(x, kernel_size=[1, 1], activation_fn=None, scope='conv2')
    return x

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
