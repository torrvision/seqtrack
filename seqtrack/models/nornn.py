class Nornn(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 conv1_stride=2,
                 feat_act='tanh', # NOTE: tanh ~ linear >>>>> relu. Do not use relu!
                 new_target=False,
                 new_target_combine='add', # {'add', 'concat', 'gau_sum', concat_gau', 'share_gau_sum'}
                 supervision_score_A0=False,
                 supervision_score_At=False,
                 target_is_vector=False,
                 join_method='dot', # {'dot', 'concat'}
                 scale_target_num=1, # odd number, e.g., {1, 3, 5}
                 scale_target_mode='add', # {'add', 'weight'}
                 divide_target=False,
                 bnorm_xcorr=False,
                 normalize_input_range=False,
                 target_share=True,
                 target_concat_mask=False, # can only be True if share is False
                 interm_supervision=False,
                 rnn=False,
                 rnn_cell_type='lstm',
                 rnn_num_layers=1,
                 rnn_residual=False,
                 rnn_perturb_prob=0.0, # sampling rate of batch-wise scoremap perturbation.
                 rnn_skip=False,
                 rnn_skip_support=1,
                 rnn_hglass=False,
                 coarse_hmap=False,
                 use_hmap_prior=False,
                 use_cosine_penalty=False,
                 boxreg=False,
                 boxreg_delta=False,
                 boxreg_stop_grad=False,
                 boxreg_regularize=False,
                 sc=False, # scale classification
                 sc_net=True, # Use a separate network?
                 sc_pass_hmap=False,
                 sc_shift_amount=0.0,
                 sc_score_threshold=0.9,
                 sc_num_class=3,
                 sc_step_size=0.03,
                 light=False,
                 ):
        self.summaries_collections = summaries_collections
        # model parameters
        self.conv1_stride      = conv1_stride
        self.feat_act          = feat_act
        self.new_target        = new_target
        self.new_target_combine= new_target_combine
        self.supervision_score_A0 = supervision_score_A0
        self.supervision_score_At = supervision_score_At
        self.target_is_vector  = target_is_vector
        self.join_method       = join_method
        self.scale_target_num  = scale_target_num
        self.scale_target_mode= scale_target_mode
        self.divide_target     = divide_target
        self.bnorm_xcorr       = bnorm_xcorr
        self.normalize_input_range = normalize_input_range
        self.target_share          = target_share
        self.target_concat_mask    = target_concat_mask
        self.interm_supervision= interm_supervision
        self.rnn               = rnn
        self.rnn_cell_type     = rnn_cell_type
        self.rnn_num_layers    = rnn_num_layers
        self.rnn_residual      = rnn_residual
        self.rnn_perturb_prob  = rnn_perturb_prob
        self.rnn_skip          = rnn_skip
        self.rnn_skip_support  = rnn_skip_support
        self.rnn_hglass        = rnn_hglass
        self.coarse_hmap       = coarse_hmap
        self.use_hmap_prior    = use_hmap_prior
        self.use_cosine_penalty= use_cosine_penalty
        self.boxreg            = boxreg
        self.boxreg_delta      = boxreg_delta
        self.boxreg_stop_grad  = boxreg_stop_grad
        self.boxreg_regularize = boxreg_regularize
        self.sc                = sc
        self.sc_net            = sc_net
        self.sc_pass_hmap      = sc_pass_hmap
        self.sc_shift_amount   = sc_shift_amount
        self.sc_score_threshold= sc_score_threshold
        self.sc_num_class      = sc_num_class
        self.sc_step_size      = sc_step_size
        self.light             = light
        self.outputs, self.state_init, self.state_final, self.gt, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = None # Batch size of model instance, or None if dynamic.

    def _load_model(self, inputs, o):

        def pass_stn_localization(x):
            ''' Localization network in Spatial Transformer Network
            '''
            with slim.arg_scope([slim.fully_connected],
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.zeros_initializer,
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.flatten(x)
                x = slim.fully_connected(x, 1024, scope='fc1')
                b_init = np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten() # identity at start.
                x = slim.fully_connected(x, 6, biases_initializer=tf.constant_initializer(b_init),scope='fc2')
            return x

        def pass_cnn(x, o, is_training, act, is_target=False):
            ''' Fully convolutional cnn.
            Either custom or other popular model.
            Note that Slim pre-defined networks can't be directly used, as they
            have some requirements that don't fit my inputs. Thus, for popular
            models I define network on my own following Slim definition.
            '''
            # NOTE: Use padding 'SAME' in convolutions and pooling so that
            # corners of before/after convolutions match! If training example
            # pairs are centered, it may not need to be this way though.
            # TODO: Try DenseNet! Diverse feature, lower parameters, data augmentation effect.
            if o.cnn_model == 'custom':
                with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'fused': True},
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, 11, stride=self.conv1_stride, scope='conv1')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool1')
                    x = slim.conv2d(x, 32, 5, stride=1, scope='conv2')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool2')
                    x = slim.conv2d(x, 64, 3, stride=1, scope='conv3')
                    ## x = slim.conv2d(x, 128, 3, stride=1, scope='conv4')
                    x = slim.conv2d(x, 64 if self.light else 128, 3, stride=1, scope='conv4')
                    if not self.target_is_vector:
                        ## x = slim.conv2d(x, 256, 3, stride=1, activation_fn=get_act(act), scope='conv5')
                        x = slim.conv2d(x, 64 if self.light else 256, 3, stride=1,
                            activation_fn=get_act(act), scope='conv5')
                    else:
                        # Use stride 2 at conv5.
                        ## x = slim.conv2d(x, 256, 3, stride=2, scope='conv5')
                        x = slim.conv2d(x, 64 if self.light else 256, 3, stride=2, scope='conv5')
                        # Total stride is 8 * conv1_stride.
                        total_stride = self.conv1_stride * 8
                        # If frmsz = 241, conv1_stride = 3, search_scale = 2*target_scale
                        # then 240 / 2 / 24 = 5 gives a template of size 6.
                        kernel_size = (o.frmsz-1) * o.target_scale / o.search_scale / total_stride + 1
                        # print 'conv6 kernel size:', kernel_size
                        assert kernel_size > 0
                        # (kernel_size-1) == (frmsz-1) * (target_scale / search_scale) / total_stride
                        assert (kernel_size-1)*total_stride*o.search_scale == (o.frmsz-1)*o.target_scale
                        assert np.all(np.array(kernel_size) % 2 == 1)
                        x = slim.conv2d(x, 64 if self.light else 256, kernel_size,
                            activation_fn=get_act(act),
                            padding='VALID' if is_target else 'SAME',
                            scope='conv6')
                        if is_target:
                            assert x.shape.as_list()[-3:-1] == [1, 1]

            elif o.cnn_model =='siamese': # exactly same as Siamese
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 96, 11, stride=self.conv1_stride, scope='conv1')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool1')
                    x = slim.conv2d(x, 256, 5, stride=1, scope='conv2')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool2')
                    x = slim.conv2d(x, 192, 3, stride=1, scope='conv3')
                    x = slim.conv2d(x, 192, 3, stride=1, scope='conv4')
                    x = slim.conv2d(x, 128, 3, stride=1, activation_fn=get_act(act), scope='conv5')
            elif o.cnn_model == 'vgg_16':
                # TODO: NEED TO TEST AGAIN. Previously I had activation at the end.. :(
                #with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                #    nets, end_points = alexnet.alexnet_v2(x, spatial_squeeze=False)
                with slim.arg_scope([slim.conv2d],
                        trainable=o.cnn_trainable,
                        variables_collections=[o.cnn_model],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='pool1')
                    x = slim.repeat(x, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='pool2')
                    x = slim.repeat(x, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='pool3')
                    x = slim.repeat(x, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    # No use repeat. Split so that no activation applied at the end.
                    #x = slim.repeat(x, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    x = slim.conv2d(x, 512, [3, 3], scope='conv5/conv5_1')
                    x = slim.conv2d(x, 512, [3, 3], scope='conv5/conv5_2')
                    x = slim.conv2d(x, 512, [3, 3], activation_fn=get_act(act), scope='conv5/conv5_3')
            else:
                assert False, 'Model not available'
            return x

        def pass_cross_correlation(search, target, o):
            ''' Perform cross-correlation (convolution) to find the target object.
            I use channel-wise convolution, instead of normal convolution.
            '''
            dims = target.shape.as_list()
            assert dims[1] % 2 == 1, 'target size has to be odd number: {}'.format(dims[-3:-1])

            # multi-scale targets.
            # NOTE: To confirm the feature in this module, there should be
            # enough training pairs different in scale during training -> data augmentation.
            targets = []
            scales = range(dims[1]-(self.scale_target_num/2)*2, dims[1]+(self.scale_target_num/2)*2+1, 2)
            for s in scales:
                targets.append(tf.image.resize_images(target, [s, s],
                                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                      align_corners=True))

            if self.divide_target:
                assert False, 'Do not use it now.'
                height = target.shape.as_list()[1]
                patchsz = [5] # TODO: diff sizes
                for n in range(len(patchsz)):
                    for i in range(0,height,patchsz[n]):
                        for j in range(0,height,patchsz[n]):
                            #targets.append(target[:,i:i+patchsz[n], j:j+patchsz[n], :])
                            # Instead of divide, use mask operation to preserve patch size.
                            grid_x = tf.expand_dims(tf.range(height), 0)
                            grid_y = tf.expand_dims(tf.range(height), 1)
                            x1 = tf.expand_dims(j, -1)
                            x2 = tf.expand_dims(j+patchsz[n]-1, -1)
                            y1 = tf.expand_dims(i, -1)
                            y2 = tf.expand_dims(i+patchsz[n]-1, -1)
                            mask = tf.logical_and(
                                tf.logical_and(tf.less_equal(x1, grid_x), tf.less_equal(grid_x, x2)),
                                tf.logical_and(tf.less_equal(y1, grid_y), tf.less_equal(grid_y, y2)))
                            targets.append(target * tf.expand_dims(tf.cast(mask, tf.float32), -1))
                            weights.append(float(patchsz[n])/height)

            if self.join_method == 'dot':
                # cross-correlation. (`scale_target_num` # of scoremaps)
                scoremap = []
                for k in range(len(targets)):
                    scoremap.append(diag_conv(search, targets[k], strides=[1, 1, 1, 1], padding='SAME'))

                if self.scale_target_mode == 'add':
                    scoremap = tf.add_n(scoremap)
                elif self.scale_target_mode == 'weight':
                    scoremap = tf.concat(scoremap, -1)
                    dims_combine = scoremap.shape.as_list()
                    with slim.arg_scope([slim.conv2d],
                            kernel_size=1,
                            weights_regularizer=slim.l2_regularizer(o.wd)):
                        scoremap = slim.conv2d(scoremap, num_outputs=dims_combine[-1], scope='combine_scoremap1')
                        scoremap = slim.conv2d(scoremap, num_outputs=dims[-1], scope='combine_scoremap2')
                else:
                    assert False, 'Wrong combine mode for multi-scoremap.'

            elif self.join_method == 'concat':
                with tf.variable_scope('concat'):
                    target_size = target.shape.as_list()[-3:-1]
                    assert target_size == [1, 1], 'target size: {}'.format(str(target_size))
                    dim = target.shape.as_list()[-1]
                    # Concat and perform 1x1 convolution.
                    # relu(linear(concat(search, target)))
                    # = relu(linear(search) + linear(target))
                    scoremap = tf.nn.relu(slim.conv2d(target, dim, kernel_size=1, activation_fn=None) +
                                          slim.conv2d(search, dim, kernel_size=1, activation_fn=None))
            else:
                raise ValueError('unknown join: {}'.format(self.join_method))

            # After cross-correlation, put some convolutions (separately from deconv).
            bnorm_args = {} if not self.bnorm_xcorr else {
                'normalizer_fn': slim.batch_norm,
                'normalizer_params': {'is_training': is_training, 'fused': True},
            }
            with slim.arg_scope([slim.conv2d],
                    num_outputs=scoremap.shape.as_list()[-1],
                    kernel_size=3,
                    weights_regularizer=slim.l2_regularizer(o.wd),
                    **bnorm_args):
                scoremap = slim.conv2d(scoremap, scope='conv1')
                scoremap = slim.conv2d(scoremap, scope='conv2')
            return scoremap

        def combine_scoremaps(scoremap_init, scoremap_curr, o, is_training):
            dims = scoremap_init.shape.as_list()
            if self.new_target_combine in ['add', 'multiply', 'concat']:
                if self.new_target_combine == 'add':
                    scoremap = scoremap_init + scoremap_curr
                elif self.new_target_combine == 'multiply':
                    scoremap = scoremap_init * scoremap_curr
                elif self.new_target_combine == 'concat':
                    scoremap = tf.concat([scoremap_init, scoremap_curr], -1)
                with slim.arg_scope([slim.conv2d],
                        num_outputs=dims[-1],
                        kernel_size=3,
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    scoremap = slim.conv2d(scoremap, scope='conv1')
                    scoremap = slim.conv2d(scoremap, scope='conv2')
            elif self.new_target_combine == 'gau':
                scoremap = tf.concat([scoremap_init, scoremap_curr], -1)
                scoremap_residual = tf.identity(scoremap)
                with slim.arg_scope([slim.conv2d],
                        num_outputs=dims[-1],
                        kernel_size=3,
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    scoremap = slim.conv2d(scoremap, rate=2, scope='conv_dilated')
                    scoremap = tf.nn.tanh(scoremap) * tf.nn.sigmoid(scoremap)
                    scoremap = slim.conv2d(scoremap, kernel_size=1, scope='conv_1x1')
                    scoremap += slim.conv2d(scoremap_residual, kernel_size=1, scope='conv_residual')
            else:
                assert False, 'Not available scoremap combine mode.'
            return scoremap

        def pass_interm_supervision(x, o):
            with slim.arg_scope([slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                dim = x.shape.as_list()[-1]
                x = slim.conv2d(x, num_outputs=dim, kernel_size=3, scope='conv1')
                x = slim.conv2d(x, num_outputs=2, kernel_size=1, activation_fn=None, scope='conv2')
            return x

        def pass_hourglass_rnn(x, rnn_state, is_training, o):
            dims = x.shape.as_list()
            x_skip = []
            with slim.arg_scope([slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x_skip.append(x)
                x = slim.conv2d(x, dims[-1]*2, 5, 2,  scope='encoder_conv1')
                x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='encoder_pool1')
                x_skip.append(x)
                x = slim.conv2d(x, dims[-1]*2, 5, 2, scope='encoder_conv2')
                x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='encoder_pool2')
                x_skip.append(x)
                x, rnn_state = pass_rnn(x, rnn_state, self.rnn_cell_type, o, self.rnn_skip)
                x = slim.conv2d(tf.image.resize_images(x + x_skip[2], [9, 9], align_corners=True),
                                dims[-1]*2, 3, 1, scope='decoder1')
                x = slim.conv2d(tf.image.resize_images(x + x_skip[1], [33, 33], align_corners=True),
                                dims[-1],   3, 1, scope='decoder2')
                x = slim.conv2d(x + x_skip[0], dims[-1], 3, 1, scope='decoder3')
            return x, rnn_state

        def pass_deconvolution(x, is_training, o, num_outputs=2):
            ''' Upsampling layers.
            The last layer should not have an activation!
            '''
            with slim.arg_scope([slim.conv2d],
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                if o.cnn_model in ['custom', 'siamese']:
                    dim = x.shape.as_list()[-1]
                    if self.coarse_hmap: # No upsample layers.
                        x = slim.conv2d(x, num_outputs=dim, kernel_size=3, scope='deconv1')
                        x = slim.conv2d(x, num_outputs=num_outputs, kernel_size=1,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope='deconv2')
                    else:
                        x = tf.image.resize_images(x, [(o.frmsz-1)/4+1]*2, align_corners=True)
                        x = slim.conv2d(x, num_outputs=dim/2, kernel_size=3, scope='deconv1')
                        x = tf.image.resize_images(x, [(o.frmsz-1)/2+1]*2, align_corners=True)
                        x = slim.conv2d(x, num_outputs=dim/4, kernel_size=3, scope='deconv2')
                        x = tf.image.resize_images(x, [o.frmsz]*2, align_corners=True)
                        x = slim.conv2d(x, num_outputs=num_outputs, kernel_size=1,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope='deconv3')
                # elif o.cnn_model == 'vgg_16':
                #     assert False, 'Please update this better before using it..'
                #     x = slim.conv2d(x, num_outputs=512, scope='deconv1')
                #     x = tf.image.resize_images(x, [61, 61])
                #     x = slim.conv2d(x, num_outputs=256, scope='deconv2')
                #     x = tf.image.resize_images(x, [121, 121])
                #     x = slim.conv2d(x, num_outputs=2, scope='deconv3')
                #     x = tf.image.resize_images(x, [o.frmsz, o.frmsz])
                #     x = slim.conv2d(x, num_outputs=2, activation_fn=None, scope='deconv4')
                else:
                    assert False, 'Not available option.'
            return x

        def pass_conv_after_rnn(x, o):
            with slim.arg_scope([slim.conv2d],
                    kernel_size=3,
                    num_outputs=x.shape.as_list()[-1],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(x, scope='conv1')
                x = slim.conv2d(x, scope='conv2')
            return x

        def pass_conv_boxreg_branch(x, o, is_training):
            dims = x.shape.as_list()
            with slim.arg_scope([slim.conv2d],
                    kernel_size=1,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(x, num_outputs=dims[-1], scope='conv1')
                x = slim.conv2d(x, num_outputs=dims[-1], scope='conv2')
            return x

        def pass_conv_boxreg_project(x, o, is_training, dim):
            '''Projects to lower dimension after concat with scoremap.'''
            with slim.arg_scope([slim.conv2d],
                    kernel_size=1,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(x, num_outputs=dim, scope='conv1')
                x = slim.conv2d(x, num_outputs=dim, scope='conv2')
            return x

        def pass_regress_box(x, is_training):
            ''' Regress output rectangle.
            '''
            dims = x.shape.as_list()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'fused': True}):
                    if not self.light:
                        x = slim.conv2d(x, dims[-1]*2, 5, 2, padding='VALID', scope='conv1')
                        x = slim.conv2d(x, dims[-1]*4, 5, 2, padding='VALID', scope='conv2')
                        x = slim.max_pool2d(x, 2, 2, scope='pool1')
                        x = slim.conv2d(x, dims[-1]*8, 3, 1, padding='VALID', scope='conv3')
                    else:
                        x = slim.conv2d(x, dims[-1]*2, 3, stride=2, scope='conv1')
                        x = slim.conv2d(x, dims[-1]*2, 3, stride=2, scope='conv2')
                        x = slim.conv2d(x, dims[-1]*4, 3, stride=2, scope='conv3')
                        x = slim.conv2d(x, dims[-1]*4, 3, stride=2, scope='conv4')
                        kernel_size = x.shape.as_list()[-3:-1]
                        x = slim.conv2d(x, dims[-1]*8, kernel_size, padding='VALID', scope='conv5')
                    assert x.shape.as_list()[-3:-1] == [1, 1]
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1')
                    x = slim.fully_connected(x, 1024, scope='fc2')
                    x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        def pass_scale_classification(x, is_training):
            ''' Classification network for scale change.
            '''
            dims = x.shape.as_list()
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    normalizer_fn=slim.batch_norm,
                    normalizer_params={'is_training': is_training, 'fused': True},
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(x, 16,  5, 2, padding='VALID', scope='conv1')
                x = slim.conv2d(x, 32,  5, 2, padding='VALID', scope='conv2')
                x = slim.max_pool2d(x, 2, 2, scope='pool1')
                x = slim.conv2d(x, 64,  3, 1, padding='VALID', scope='conv3')
                x = slim.conv2d(x, 128, 3, 1, padding='VALID', scope='conv4')
                x = slim.conv2d(x, 256, 3, 1, padding='VALID', scope='conv5')
                assert x.shape.as_list()[1] == 1
                x = slim.flatten(x)
                x = slim.fully_connected(x, 256, scope='fc1')
                x = slim.fully_connected(x, 256, scope='fc2')
                x = slim.fully_connected(x, self.sc_num_class, activation_fn=None, normalizer_fn=None, scope='fc3')
            return x

        # Inputs to the model.
        x           = inputs['x']  # shape [b, ntimesteps, h, w, 3]
        x0          = inputs['x0'] # shape [b, h, w, 3]
        y0          = inputs['y0'] # shape [b, 4]
        y           = inputs['y']  # shape [b, ntimesteps, 4]
        use_gt      = inputs['use_gt']
        gt_ratio    = inputs['gt_ratio']
        is_training = inputs['is_training']
        y_is_valid  = inputs['y_is_valid']

        # TODO: This would better be during loading data.
        y0 = enforce_inside_box(y0) # In OTB, Panda has GT error (i.e., > 1.0).
        y  = enforce_inside_box(y)

        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0) # for `delta` regression type output.
        x_prev = x_init
        y_prev = y_init

        # Outputs from the model.
        y_pred = []
        hmap_pred = []
        # Case of object-centric approach.
        y_pred_oc = []
        hmap_pred_oc = []
        hmap_pred_oc_A0 = []
        box_s_raw = []
        box_s_val = []
        #target = []
        target_init_list = []
        target_curr_list = []
        search = []

        hmap_interm = {k: [] for k in ['score', 'score_A0', 'score_At']}
        sc_out_list = []
        sc_active_list = []

        scales = (np.arange(self.sc_num_class) - (self.sc_num_class / 2)) * self.sc_step_size + 1

        target_scope = o.cnn_model if self.target_share else o.cnn_model+'_target'
        search_scope = o.cnn_model if self.target_share else o.cnn_model+'_search'

        # Some pre-processing.
        target_size = (o.frmsz - 1) * o.target_scale / o.search_scale + 1
        assert (target_size-1)*o.search_scale == (o.frmsz-1)*o.target_scale
        target_init, box_context, _ = process_image_with_box(x0, y0, o,
            crop_size=target_size, scale=o.target_scale, aspect=inputs['aspect'])
        if self.normalize_input_range:
            target_init *= 1. / 255.
        if self.target_concat_mask:
            target_mask_oc = get_masks_from_rectangles(geom.crop_rect(y0, box_context), o,
                output_size=target_init.shape.as_list()[-3:-1])
            # Magnitude of mask should be similar to colors.
            if not self.normalize_input_range:
                target_mask_oc *= 255.
            target_input = tf.concat([target_init, target_mask_oc], axis=-1)
        else:
            target_input = target_init

        with tf.variable_scope(target_scope):
            target_init_feat = pass_cnn(target_input, o, is_training, self.feat_act, is_target=True)

        if self.new_target:
            batchsz = tf.shape(x)[0]
            max_score_A0_init = tf.fill([batchsz], 1.0)
            max_score_A0_prev = tf.identity(max_score_A0_init)
            target_curr_init = tf.identity(target_init)
            target_curr_prev = tf.identity(target_curr_init)

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]

            search_curr, box_s_raw_curr, box_s_val_curr = process_image_with_box(
                    x_curr, y_prev, o, crop_size=o.frmsz, scale=o.search_scale, aspect=inputs['aspect'])
            if self.normalize_input_range:
                search_curr *= 1. / 255.

            with tf.variable_scope(search_scope, reuse=(t > 0 or self.target_share)) as scope:
                search_feat = pass_cnn(search_curr, o, is_training, self.feat_act, is_target=False)

            if self.new_target:
                target_curr, _, _ = process_image_with_box(x_prev, y_prev, o,
                    crop_size=target_size, scale=o.target_scale, aspect=inputs['aspect'])

                update_target = tf.greater_equal(max_score_A0_prev, 0.9) # TODO: threshold val.
                target_curr = tf.where(update_target, target_curr, target_curr_prev)
                target_curr_prev = target_curr

                with tf.variable_scope(target_scope, reuse=True):
                    target_curr_feat = pass_cnn(target_curr, o, is_training, self.feat_act, is_target=True)

                with tf.variable_scope('cross_correlation', reuse=(t > 0)) as scope:
                    scoremap_init = pass_cross_correlation(search_feat, target_init_feat, o)
                    scope.reuse_variables()
                    scoremap_curr = pass_cross_correlation(search_feat, target_curr_feat, o)

                # Deconvolution branch from "score_init".
                with tf.variable_scope('deconvolution', reuse=(t > 0)):
                    hmap_curr_pred_oc_A0 = pass_deconvolution(scoremap_init, is_training, o)
                    if self.coarse_hmap:
                        # Upsample to compute translation but not to compute loss.
                        hmap_upsample = tf.image.resize_images(hmap_curr_pred_oc_A0, [o.frmsz, o.frmsz],
                            method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
                        hmap_curr_pred_oc_fg_A0 = tf.expand_dims(tf.unstack(tf.nn.softmax(hmap_upsample), axis=-1)[0], -1)
                    else:
                        hmap_curr_pred_oc_fg_A0 = tf.expand_dims(tf.nn.softmax(hmap_curr_pred_oc_A0)[:,:,:,0], -1)
                    # max score will be used to make a decision about updating target.
                    max_score_A0_prev = tf.reduce_max(hmap_curr_pred_oc_fg_A0, axis=(1,2,3))

                with tf.variable_scope('supervision_scores', reuse=(t > 0)) as scope:
                    if self.supervision_score_A0 and not self.supervision_score_At:
                        hmap_interm['score_A0'].append(pass_interm_supervision(scoremap_init, o))
                    elif not self.supervision_score_A0 and self.supervision_score_At:
                        hmap_interm['score_At'].append(pass_interm_supervision(scoremap_curr, o))
                    elif self.supervision_score_A0 and self.supervision_score_At:
                        hmap_interm['score_A0'].append(pass_interm_supervision(scoremap_init, o))
                        scope.reuse_variables()
                        hmap_interm['score_At'].append(pass_interm_supervision(scoremap_curr, o))

                with tf.variable_scope('combine_scoremaps', reuse=(t > 0)):
                    scoremap = combine_scoremaps(scoremap_init, scoremap_curr, o, is_training)
            else:
                with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                    scoremap = pass_cross_correlation(search_feat, target_init_feat, o)

            if self.interm_supervision:
                with tf.variable_scope('interm_supervision', reuse=(t > 0)):
                    hmap_interm['score'].append(pass_interm_supervision(scoremap, o))

            # JV: Define RNN state after obtaining scoremap to get size automatically.
            if self.rnn:
                if t == 0:
                    # RNN cell states
                    if self.rnn_hglass:
                        assert self.rnn_num_layers == 1
                        state_dim = [3, 3, scoremap.shape.as_list()[-1]*2]
                    else:
                        state_dim = scoremap.shape.as_list()[-3:]
                    assert all(state_dim) # Require that size is static (no None values).
                    rnn_state_init = get_initial_rnn_state(
                        cell_type=self.rnn_cell_type,
                        cell_size=[tf.shape(x0)[0], self.rnn_skip_support] + state_dim,
                        num_layers=self.rnn_num_layers)
                    rnn_state = [
                        {k: tf.identity(rnn_state_init[l][k]) for k in rnn_state_init[l].keys()}
                        for l in range(len(rnn_state_init))]

                # Apply probabilsitic dropout to scoremap before passing it to RNN.
                scoremap_dropout = tf.layers.dropout(scoremap, rate=0.1,
                    noise_shape=[tf.shape(scoremap)[i] for i in range(3)]+[1], training=is_training) # same along channels
                prob = tf.random_uniform(shape=[tf.shape(scoremap)[0]], minval=0, maxval=1)
                use_dropout = tf.logical_and(is_training, tf.less(prob, self.rnn_perturb_prob)) # apply batch-wise.
                scoremap = tf.where(use_dropout, scoremap_dropout, scoremap)

                if self.rnn_hglass:
                    with tf.variable_scope('hourglass_rnn', reuse=(t > 0)):
                        scoremap, rnn_state[l] = pass_hourglass_rnn(scoremap, rnn_state[l], is_training, o)
                else:
                    for l in range(self.rnn_num_layers):
                        with tf.variable_scope('rnn_layer{}'.format(l), reuse=(t > 0)):
                            if self.rnn_residual:
                                scoremap_ori = tf.identity(scoremap)
                                scoremap, rnn_state[l] = pass_rnn(scoremap, rnn_state[l], self.rnn_cell_type, o, self.rnn_skip)
                                scoremap += scoremap_ori
                            else:
                                scoremap, rnn_state[l] = pass_rnn(scoremap, rnn_state[l], self.rnn_cell_type, o, self.rnn_skip)

                with tf.variable_scope('convolutions_after_rnn', reuse=(t > 0)):
                    scoremap = pass_conv_after_rnn(scoremap, o)

            with tf.variable_scope('deconvolution', reuse=(True if self.new_target else t > 0)):
                hmap_curr_pred_oc = pass_deconvolution(scoremap, is_training, o, num_outputs=2)
                if self.use_hmap_prior:
                    hmap_shape = hmap_curr_pred_oc.shape.as_list()
                    hmap_prior = tf.get_variable('hmap_prior', hmap_shape[-3:],
                                                 initializer=tf.zeros_initializer(),
                                                 regularizer=slim.l2_regularizer(o.wd))
                    hmap_curr_pred_oc += hmap_prior
                p_loc_coarse = tf.unstack(tf.nn.softmax(hmap_curr_pred_oc), axis=-1)[0]
                if self.coarse_hmap:
                    # Upsample to compute translation but not to compute loss.
                    hmap_upsample = tf.image.resize_images(hmap_curr_pred_oc, [o.frmsz, o.frmsz],
                        method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
                    hmap_curr_pred_oc_fg = tf.expand_dims(tf.unstack(tf.nn.softmax(hmap_upsample), axis=-1)[0], -1)
                else:
                    hmap_curr_pred_oc_fg = tf.expand_dims(tf.nn.softmax(hmap_curr_pred_oc)[:,:,:,0], -1)
                if self.use_cosine_penalty:
                    hann_1d = np.expand_dims(np.hanning(hmap_curr_pred_oc_fg.shape.as_list()[1]), axis=0)
                    penalty = np.expand_dims(np.transpose(hann_1d) * hann_1d, -1)
                    #penalty = penalty / np.sum(penalty)
                    window_influence = 0.1
                    hmap_curr_pred_oc_fg = (1-window_influence) * hmap_curr_pred_oc_fg + window_influence * penalty

            if self.boxreg: # regress box from `scoremap`.
                assert self.coarse_hmap, 'Do not up-sample scoremap in the case of box-regression.'

                # CNN processing target-in-search raw image pair.
                search_0, _, _ = process_image_with_box(
                        x0, y0, o, crop_size=o.frmsz, scale=o.search_scale, aspect=inputs['aspect'])
                search_pair = tf.concat([search_0, search_curr], 3)
                with tf.variable_scope('process_search_pair', reuse=(t > 0)):
                    search_pair_feat = pass_cnn(search_pair, o, is_training, 'relu')

                # Create input to box-regression network.
                # 1. Enable/disable `boxreg_stop_grad`.
                # 2. Perform convolution on scoremap. Effect of batchnorm and relu (possibly after RNN).
                # 3. Combine with scoremap. Simple concat for now.
                # (additionally, flow, flow-unsupervised, flow-part)
                if self.boxreg_stop_grad:
                    scoremap = tf.stop_gradient(tf.identity(scoremap))
                with tf.variable_scope('conv_boxreg_branch', reuse=(t > 0)):
                    scoremap = pass_conv_boxreg_branch(scoremap, o, is_training)
                boxreg_inputs = tf.concat([search_pair_feat, scoremap], -1)

                if self.light:
                    with tf.variable_scope('conv_boxreg_project', reuse=(t > 0)):
                        boxreg_inputs = pass_conv_boxreg_project(boxreg_inputs, o, is_training,
                            dim=scoremap.shape.as_list()[-1])

                # Box regression.
                with tf.variable_scope('regress_box', reuse=(t > 0)):
                    if self.boxreg_delta:
                        y_curr_pred_oc_delta = pass_regress_box(boxreg_inputs, is_training)
                        y_curr_pred_oc = y_curr_pred_oc_delta + \
                                         tf.stack([0.5 - 1./o.search_scale/2., 0.5 + 1./o.search_scale/2.]*2)
                    else:
                        y_curr_pred_oc = pass_regress_box(boxreg_inputs, is_training)
                    y_curr_pred = to_image_centric_coordinate(y_curr_pred_oc, box_s_raw_curr, o)

                # Regularize box scale manually.
                if self.boxreg_regularize:
                    y_curr_pred = regularize_scale(y_prev, y_curr_pred, y0, 0.5, 1.0)

            elif self.sc:
                y_curr_pred_oc, y_curr_pred = get_rectangles_from_hmap(
                        hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, o, y_prev)
                if self.sc_net:
                    # scale-classification network.
                    target_init_pad, _, _ = process_image_with_box(x0, y0, o,
                            crop_size=(o.frmsz - 1) * o.target_scale / o.search_scale + 1, scale=1, aspect=inputs['aspect'])
                    #target_pred_pad, _, _ = process_image_with_box(x_curr, y_curr_pred, o,
                    #        crop_size=(o.frmsz - 1) * o.target_scale / o.search_scale + 1, scale=1, aspect=inputs['aspect'])
                    target_pred_pad, _, _ = process_image_with_box(x_curr,
                            tf.cond(is_training, lambda: geom.rect_translate_random(y_curr_pred, self.sc_shift_amount),
                                                 lambda: y_curr_pred), o,
                            crop_size=(o.frmsz - 1) * o.target_scale / o.search_scale + 1, scale=1, aspect=inputs['aspect'])
                    sc_in = tf.concat([target_init_pad, target_pred_pad], -1)
                    if self.sc_pass_hmap:
                        hmap_crop, _, _ = process_image_with_box(hmap_curr_pred_oc_fg, y_curr_pred, o,
                            crop_size=(o.frmsz - 1) * o.target_scale / o.search_scale + 1, scale=1, aspect=inputs['aspect'])
                        sc_in = tf.concat([sc_in, hmap_crop*255], -1)
                    with tf.variable_scope('scale_classfication',reuse=(t > 0)):
                        sc_out = pass_scale_classification(sc_in, is_training)
                        sc_out_list.append(sc_out)
                else:
                    with tf.variable_scope('sc_deconv', reuse=(t > 0)):
                        sc_out = pass_deconvolution(scoremap, is_training, o, num_outputs=self.sc_num_class)
                        sc_out_list.append(sc_out)

                # compute scale and update box.
                p_scale = tf.nn.softmax(sc_out)
                is_max_scale = tf.equal(p_scale, tf.reduce_max(p_scale, axis=-1, keep_dims=True))
                is_max_scale = tf.to_float(is_max_scale)
                scale = tf.reduce_sum(scales * is_max_scale, axis=-1) / tf.reduce_sum(is_max_scale, axis=-1)
                if not self.sc_net:
                    # Scale estimation is convolutional. Take location with max score.
                    sc_active = tf.greater_equal(p_loc_coarse,
                        tf.reduce_max(p_loc_coarse, axis=(-2, -1), keep_dims=True)*0.95)
                    sc_active = tf.to_float(sc_active)
                    scale = (tf.reduce_sum(scale * sc_active, axis=(-2, -1)) /
                             tf.reduce_sum(sc_active, axis=(-2, -1)))
                    sc_active_list.append(sc_active)
                if self.sc_score_threshold > 0:
                    # Use scale = 1 if max location score is not above threshold.
                    max_score = tf.reduce_max(hmap_curr_pred_oc_fg, axis=(1,2,3))
                    is_pass = tf.greater_equal(max_score, self.sc_score_threshold)
                    scale = tf.where(is_pass, scale, tf.ones_like(scale))
                y_curr_pred = grow_rect(tf.expand_dims(scale, -1), y_curr_pred)
            else: # argmax to find center (then use x0's box)
                y_curr_pred_oc, y_curr_pred = get_rectangles_from_hmap(
                    hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, o, y0)

            # Post-processing.
            y_curr_pred = enforce_inside_box(y_curr_pred, translate=True)

            # Get image-centric outputs. Some are used for visualization purpose.
            hmap_curr_pred = to_image_centric_hmap(hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, o)

            # Outputs.
            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr_pred)
            y_pred_oc.append(y_curr_pred_oc_delta if self.boxreg_delta else y_curr_pred_oc)
            hmap_pred_oc.append(hmap_curr_pred_oc)
            hmap_pred_oc_A0.append(hmap_curr_pred_oc_A0 if self.new_target else None)
            box_s_raw.append(box_s_raw_curr)
            box_s_val.append(box_s_val_curr)
            #target.append(target_init) # To visualize what network sees.
            target_init_list.append(target_init) # To visualize what network sees.
            target_curr_list.append(target_curr if self.new_target else None) # To visualize what network sees.
            search.append(search_curr) # To visualize what network sees.

            # Update for next time-step.
            x_prev = x_curr
            # Scheduled sampling. In case label is invalid, use prediction.
            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            y_prev = tf.cond(gt_condition,
                             lambda: tf.where(y_is_valid[:,t], y_curr, y_curr_pred),
                             lambda: y_curr_pred)

        y_pred        = tf.stack(y_pred, axis=1)
        hmap_pred     = tf.stack(hmap_pred, axis=1)
        y_pred_oc     = tf.stack(y_pred_oc, axis=1)
        hmap_pred_oc  = tf.stack(hmap_pred_oc, axis=1)
        hmap_pred_oc_A0 = tf.stack(hmap_pred_oc_A0, axis=1) if self.new_target else None
        box_s_raw     = tf.stack(box_s_raw, axis=1)
        box_s_val     = tf.stack(box_s_val, axis=1)
        #target        = tf.stack(target, axis=1)
        target_init   = tf.stack(target_init_list, axis=1)
        target_curr   = tf.stack(target_curr_list, axis=1) if self.new_target else None
        search        = tf.stack(search, axis=1)

        # Summaries from within model.
        if self.use_hmap_prior:
            with tf.variable_scope('deconvolution', reuse=True):
                hmap_prior = tf.get_variable('hmap_prior')
            with tf.name_scope('model_summary'):
                # Swap channels and (non-existent) batch dimension.
                hmap_prior = tf.transpose(tf.expand_dims(hmap_prior, 0), [3, 1, 2, 0])
                tf.summary.image('hmap_prior', hmap_prior, max_outputs=2,
                    collections=self.summaries_collections)

        for k in hmap_interm.keys():
            if not hmap_interm[k]:
                hmap_interm[k] = None
            else:
                hmap_interm[k] = tf.stack(hmap_interm[k], axis=1)

        outputs = {'y':         {'ic': y_pred,    'oc': y_pred_oc}, # NOTE: Do not use 'ic' to compute loss.
                   'hmap':      {'ic': hmap_pred, 'oc': hmap_pred_oc}, # NOTE: hmap_pred_oc is no softmax yet.
                   'hmap_A0': hmap_pred_oc_A0,
                   'hmap_interm': hmap_interm,
                   'box_s_raw': box_s_raw,
                   'box_s_val': box_s_val,
                   #'target':    target,
                   'target_init': target_init,
                   'target_curr': target_curr,
                   'search':    search,
                   'boxreg_delta': self.boxreg_delta,
                   'sc':           {'out': tf.stack(sc_out_list, axis=1),
                                    'active': tf.stack(sc_active_list, axis=1) if not self.sc_net else None,
                                    'scales': scales,
                                    } if self.sc else None
                   }
        state_init, state_final = {}, {}
        # TODO: JV: From evaluate_test, it seems that 'x' may not be required?
        state_init['x'], state_final['x'] = x_init, x_prev
        state_init['y'], state_final['y'] = y_init, y_prev
        if self.new_target:
            state_init['max_score_A0'], state_final['max_score_A0'] = max_score_A0_init, max_score_A0_prev
            state_init['target_curr'], state_final['target_curr'] = target_curr_init, target_curr_prev
        # JV: Use nested collection of state.
        if self.rnn:
            state_init['rnn'] = rnn_state_init
            state_final['rnn'] = rnn_state
        gt = {}
        dbg = {} # dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        return outputs, state_init, state_final, gt, dbg
