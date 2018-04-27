import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from seqtrack import geom
from seqtrack.models import interface
from seqtrack.models import util
from seqtrack.helpers import merge_dims, diag_xcorr, modify_aspect_ratio, get_act


class Nornn(interface.Model):

    def __init__(self,
                 search_size=257,  # Takes the role of frmsz.
                 cnn_model='custom',
                 target_scale=1,
                 search_scale=4,
                 aspect_method='stretch',
                 cnn_trainable=True,
                 wd=0.,
                 losses=None,
                 heatmap_params=None,  # Default is {'Gaussian': true}
                 conv1_stride=2,
                 feat_act='tanh',  # NOTE: tanh ~ linear >>>>> relu. Do not use relu!
                 new_target=False,
                 new_target_combine='add',  # {'add', 'concat', 'gau_sum', concat_gau', 'share_gau_sum'}
                 supervision_score_A0=False,
                 supervision_score_At=False,
                 target_is_vector=False,
                 join_method='dot',  # {'dot', 'concat'}
                 scale_target_num=1,  # odd number, e.g., {1, 3, 5}
                 scale_target_mode='add',  # {'add', 'weight'}
                 divide_target=False,
                 bnorm_xcorr=False,
                 normalize_input_range=False,
                 target_share=True,
                 target_concat_mask=False,  # can only be True if share is False
                 interm_supervision=False,
                 rnn=False,
                 rnn_cell_type='lstm',
                 rnn_num_layers=1,
                 rnn_residual=False,
                 rnn_perturb_prob=0.0,  # sampling rate of batch-wise scoremap perturbation.
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
                 sc=False,  # scale classification
                 sc_net=True,  # Use a separate network?
                 sc_pass_hmap=False,
                 sc_shift_amount=0.0,
                 sc_score_threshold=0.9,
                 sc_num_class=3,
                 sc_step_size=0.03,
                 light=False,
                 ):
        losses = losses or []
        heatmap_params = heatmap_params or {'Gaussian': True}
        wd = float(wd)

        self.search_size = search_size
        self.cnn_model = cnn_model
        self.target_scale = target_scale
        self.search_scale = search_scale
        self.aspect_method = aspect_method
        self.cnn_trainable = cnn_trainable
        self.wd = wd
        self.losses = losses
        self.heatmap_params = heatmap_params
        # model parameters
        self.conv1_stride = conv1_stride
        self.feat_act = feat_act
        self.new_target = new_target
        self.new_target_combine = new_target_combine
        self.supervision_score_A0 = supervision_score_A0
        self.supervision_score_At = supervision_score_At
        self.target_is_vector = target_is_vector
        self.join_method = join_method
        self.scale_target_num = scale_target_num
        self.scale_target_mode = scale_target_mode
        self.divide_target = divide_target
        self.bnorm_xcorr = bnorm_xcorr
        self.normalize_input_range = normalize_input_range
        self.target_share = target_share
        self.target_concat_mask = target_concat_mask
        self.interm_supervision = interm_supervision
        self.rnn = rnn
        self.rnn_cell_type = rnn_cell_type
        self.rnn_num_layers = rnn_num_layers
        self.rnn_residual = rnn_residual
        self.rnn_perturb_prob = rnn_perturb_prob
        self.rnn_skip = rnn_skip
        self.rnn_skip_support = rnn_skip_support
        self.rnn_hglass = rnn_hglass
        self.coarse_hmap = coarse_hmap
        self.use_hmap_prior = use_hmap_prior
        self.use_cosine_penalty = use_cosine_penalty
        self.boxreg = boxreg
        self.boxreg_delta = boxreg_delta
        self.boxreg_stop_grad = boxreg_stop_grad
        self.boxreg_regularize = boxreg_regularize
        self.sc = sc
        self.sc_net = sc_net
        self.sc_pass_hmap = sc_pass_hmap
        self.sc_shift_amount = sc_shift_amount
        self.sc_score_threshold = sc_score_threshold
        self.sc_num_class = sc_num_class
        self.sc_step_size = sc_step_size
        self.light = light

        # self.outputs, self.state_init, self.state_final, self.gt, self.dbg = self._load_model(inputs, o)
        # self.image_size   = (o.frmsz, o.frmsz)
        # self.sequence_len = o.ntimesteps
        # self.batch_size   = None # Batch size of model instance, or None if dynamic.

    def instantiate(self, example, run_opts, enable_loss, image_summaries_collections=None):
        if not self.normalize_input_range:
            # TODO: This is a hack to avoid modifying the implementation.
            example = dict(example)
            example['x0'] *= 255.
            example['x'] *= 255.

        self.image_summaries_collections = image_summaries_collections
        outputs, init_state, final_state = self._load_model(example, run_opts)
        if enable_loss:
            losses, gt = get_loss(
                example, outputs, search_scale=self.search_scale, search_size=self.search_size,
                losses=self.losses, perspective='oc', heatmap_params=self.heatmap_params)
        else:
            losses = {}
        with tf.name_scope('summary'):
            add_summaries(example, outputs, gt, self.image_summaries_collections)
        outputs = {'y': outputs['y']['ic']}
        return outputs, losses, init_state, final_state

    def _load_model(self, example, run_opts):
        # Inputs to the model.
        x = example['x']  # shape [b, ntimesteps, h, w, 3]
        x0 = example['x0']  # shape [b, h, w, 3]
        y0 = example['y0']  # shape [b, 4]
        y = example['y']  # shape [b, ntimesteps, 4]
        y_is_valid = example['y_is_valid']
        use_gt = run_opts['use_gt']
        gt_ratio = run_opts['gt_ratio']
        is_training = run_opts['is_training']

        x_shape = x.shape.as_list()
        ntimesteps = x_shape[1]
        imheight, imwidth = x_shape[2], x_shape[3]

        # TODO: This would better be during loading data.
        y0 = enforce_inside_box(y0)  # In OTB, Panda has GT error (i.e., > 1.0).
        y = enforce_inside_box(y)

        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0)  # for `delta` regression type output.
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

        target_scope = self.cnn_model if self.target_share else self.cnn_model + '_target'
        search_scope = self.cnn_model if self.target_share else self.cnn_model + '_search'

        # Some pre-processing.
        target_size = (self.search_size - 1) * self.target_scale / self.search_scale + 1
        assert (target_size - 1) * self.search_scale == (self.search_size - 1) * self.target_scale
        target_init, box_context, _ = process_image_with_box(
            x0, y0, crop_size=target_size, scale=self.target_scale,
            aspect=example['aspect'], aspect_method=self.aspect_method)
        if self.normalize_input_range:
            target_init *= 1. / 255.
        if self.target_concat_mask:
            target_mask_oc = get_masks_from_rectangles(
                geom.crop_rect(y0, box_context), target_init.shape.as_list()[-3:-1])
            # Magnitude of mask should be similar to colors.
            if not self.normalize_input_range:
                target_mask_oc *= 255.
            target_input = tf.concat([target_init, target_mask_oc], axis=-1)
        else:
            target_input = target_init

        with tf.variable_scope(target_scope):
            target_init_feat = self.pass_cnn(target_input, is_training, self.feat_act, is_target=True)

        if self.new_target:
            batchsz = tf.shape(x)[0]
            max_score_A0_init = tf.fill([batchsz], 1.0)
            max_score_A0_prev = tf.identity(max_score_A0_init)
            target_curr_init = tf.identity(target_init)
            target_curr_prev = tf.identity(target_curr_init)

        for t in range(ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]

            search_curr, box_s_raw_curr, box_s_val_curr = process_image_with_box(
                x_curr, y_prev, crop_size=self.search_size, scale=self.search_scale,
                aspect=example['aspect'], aspect_method=self.aspect_method)
            if self.normalize_input_range:
                search_curr *= 1. / 255.

            with tf.variable_scope(search_scope, reuse=(t > 0 or self.target_share)) as scope:
                search_feat = self.pass_cnn(search_curr, is_training, self.feat_act, is_target=False)

            if self.new_target:
                target_curr, _, _ = process_image_with_box(
                    x_prev, y_prev, crop_size=target_size, scale=self.target_scale,
                    aspect=example['aspect'], aspect_method=self.aspect_method)

                update_target = tf.greater_equal(max_score_A0_prev, 0.9)  # TODO: threshold val.
                target_curr = tf.where(update_target, target_curr, target_curr_prev)
                target_curr_prev = target_curr

                with tf.variable_scope(target_scope, reuse=True):
                    target_curr_feat = self.pass_cnn(target_curr, is_training, self.feat_act, is_target=True)

                with tf.variable_scope('cross_correlation', reuse=(t > 0)) as scope:
                    scoremap_init = self.pass_cross_correlation(search_feat, target_init_feat)
                    scope.reuse_variables()
                    scoremap_curr = self.pass_cross_correlation(search_feat, target_curr_feat)

                # Deconvolution branch from "score_init".
                with tf.variable_scope('deconvolution', reuse=(t > 0)):
                    hmap_curr_pred_oc_A0 = self.pass_deconvolution(scoremap_init, is_training)
                    if self.coarse_hmap:
                        # Upsample to compute translation but not to compute loss.
                        hmap_upsample = tf.image.resize_images(hmap_curr_pred_oc_A0, [self.search_size] * 2,
                                                               method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
                        hmap_curr_pred_oc_fg_A0 = tf.expand_dims(tf.unstack(tf.nn.softmax(hmap_upsample), axis=-1)[0], -1)
                    else:
                        hmap_curr_pred_oc_fg_A0 = tf.expand_dims(tf.nn.softmax(hmap_curr_pred_oc_A0)[:, :, :, 0], -1)
                    # max score will be used to make a decision about updating target.
                    max_score_A0_prev = tf.reduce_max(hmap_curr_pred_oc_fg_A0, axis=(1, 2, 3))

                with tf.variable_scope('supervision_scores', reuse=(t > 0)) as scope:
                    if self.supervision_score_A0 and not self.supervision_score_At:
                        hmap_interm['score_A0'].append(self.pass_interm_supervision(scoremap_init))
                    elif not self.supervision_score_A0 and self.supervision_score_At:
                        hmap_interm['score_At'].append(self.pass_interm_supervision(scoremap_curr))
                    elif self.supervision_score_A0 and self.supervision_score_At:
                        hmap_interm['score_A0'].append(self.pass_interm_supervision(scoremap_init))
                        scope.reuse_variables()
                        hmap_interm['score_At'].append(self.pass_interm_supervision(scoremap_curr))

                with tf.variable_scope('combine_scoremaps', reuse=(t > 0)):
                    scoremap = self.combine_scoremaps(scoremap_init, scoremap_curr, is_training)
            else:
                with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                    scoremap = self.pass_cross_correlation(search_feat, target_init_feat)

            if self.interm_supervision:
                with tf.variable_scope('interm_supervision', reuse=(t > 0)):
                    hmap_interm['score'].append(self.pass_interm_supervision(scoremap))

            # JV: Define RNN state after obtaining scoremap to get size automatically.
            if self.rnn:
                if t == 0:
                    # RNN cell states
                    if self.rnn_hglass:
                        assert self.rnn_num_layers == 1
                        state_dim = [3, 3, scoremap.shape.as_list()[-1] * 2]
                    else:
                        state_dim = scoremap.shape.as_list()[-3:]
                    assert all(state_dim)  # Require that size is static (no None values).
                    rnn_state_init = get_initial_rnn_state(
                        cell_type=self.rnn_cell_type,
                        cell_size=[tf.shape(x0)[0], self.rnn_skip_support] + state_dim,
                        num_layers=self.rnn_num_layers)
                    rnn_state = [
                        {k: tf.identity(rnn_state_init[l][k]) for k in rnn_state_init[l].keys()}
                        for l in range(len(rnn_state_init))]

                # Apply probabilsitic dropout to scoremap before passing it to RNN.
                scoremap_dropout = tf.layers.dropout(scoremap, rate=0.1,
                                                     noise_shape=[tf.shape(scoremap)[i] for i in range(3)] + [1], training=is_training)  # same along channels
                prob = tf.random_uniform(shape=[tf.shape(scoremap)[0]], minval=0, maxval=1)
                use_dropout = tf.logical_and(is_training, tf.less(prob, self.rnn_perturb_prob))  # apply batch-wise.
                scoremap = tf.where(use_dropout, scoremap_dropout, scoremap)

                if self.rnn_hglass:
                    with tf.variable_scope('hourglass_rnn', reuse=(t > 0)):
                        scoremap, rnn_state[l] = self.pass_hourglass_rnn(scoremap, rnn_state[l], is_training)
                else:
                    for l in range(self.rnn_num_layers):
                        with tf.variable_scope('rnn_layer{}'.format(l), reuse=(t > 0)):
                            if self.rnn_residual:
                                scoremap_ori = tf.identity(scoremap)
                                scoremap, rnn_state[l] = pass_rnn(
                                    scoremap, rnn_state[l], self.rnn_cell_type, self.wd, self.rnn_skip)
                                scoremap += scoremap_ori
                            else:
                                scoremap, rnn_state[l] = pass_rnn(
                                    scoremap, rnn_state[l], self.rnn_cell_type, self.wd, self.rnn_skip)

                with tf.variable_scope('convolutions_after_rnn', reuse=(t > 0)):
                    scoremap = self.pass_conv_after_rnn(scoremap)

            with tf.variable_scope('deconvolution', reuse=(True if self.new_target else t > 0)):
                hmap_curr_pred_oc = self.pass_deconvolution(scoremap, is_training, num_outputs=2)
                if self.use_hmap_prior:
                    hmap_shape = hmap_curr_pred_oc.shape.as_list()
                    hmap_prior = tf.get_variable('hmap_prior', hmap_shape[-3:],
                                                 initializer=tf.zeros_initializer(),
                                                 regularizer=slim.l2_regularizer(self.wd))
                    hmap_curr_pred_oc += hmap_prior
                p_loc_coarse = tf.unstack(tf.nn.softmax(hmap_curr_pred_oc), axis=-1)[0]
                if self.coarse_hmap:
                    # Upsample to compute translation but not to compute loss.
                    hmap_upsample = tf.image.resize_images(hmap_curr_pred_oc, [self.search_size] * 2,
                                                           method=tf.image.ResizeMethod.BICUBIC, align_corners=True)
                    hmap_curr_pred_oc_fg = tf.expand_dims(tf.unstack(tf.nn.softmax(hmap_upsample), axis=-1)[0], -1)
                else:
                    hmap_curr_pred_oc_fg = tf.expand_dims(tf.nn.softmax(hmap_curr_pred_oc)[:, :, :, 0], -1)
                if self.use_cosine_penalty:
                    hann_1d = np.expand_dims(np.hanning(hmap_curr_pred_oc_fg.shape.as_list()[1]), axis=0)
                    penalty = np.expand_dims(np.transpose(hann_1d) * hann_1d, -1)
                    #penalty = penalty / np.sum(penalty)
                    window_influence = 0.1
                    hmap_curr_pred_oc_fg = (1 - window_influence) * hmap_curr_pred_oc_fg + window_influence * penalty

            if self.boxreg:  # regress box from `scoremap`.
                assert self.coarse_hmap, 'Do not up-sample scoremap in the case of box-regression.'

                # CNN processing target-in-search raw image pair.
                search_0, _, _ = process_image_with_box(
                    x0, y0, crop_size=self.search_size, scale=self.search_scale,
                    aspect=example['aspect'], aspect_method=self.aspect_method)
                search_pair = tf.concat([search_0, search_curr], 3)
                with tf.variable_scope('process_search_pair', reuse=(t > 0)):
                    search_pair_feat = self.pass_cnn(search_pair, is_training, 'relu')

                # Create input to box-regression network.
                # 1. Enable/disable `boxreg_stop_grad`.
                # 2. Perform convolution on scoremap. Effect of batchnorm and relu (possibly after RNN).
                # 3. Combine with scoremap. Simple concat for now.
                # (additionally, flow, flow-unsupervised, flow-part)
                if self.boxreg_stop_grad:
                    scoremap = tf.stop_gradient(tf.identity(scoremap))
                with tf.variable_scope('conv_boxreg_branch', reuse=(t > 0)):
                    scoremap = self.pass_conv_boxreg_branch(scoremap, is_training)
                boxreg_inputs = tf.concat([search_pair_feat, scoremap], -1)

                if self.light:
                    with tf.variable_scope('conv_boxreg_project', reuse=(t > 0)):
                        boxreg_inputs = self.pass_conv_boxreg_project(boxreg_inputs, is_training,
                                                                      dim=scoremap.shape.as_list()[-1])

                # Box regression.
                with tf.variable_scope('regress_box', reuse=(t > 0)):
                    if self.boxreg_delta:
                        y_curr_pred_oc_delta = self.pass_regress_box(boxreg_inputs, is_training)
                        y_curr_pred_oc = y_curr_pred_oc_delta + \
                            tf.stack([0.5 - 1. / self.search_scale / 2.,
                                      0.5 + 1. / self.search_scale / 2.] * 2)
                    else:
                        y_curr_pred_oc = self.pass_regress_box(boxreg_inputs, is_training)
                    y_curr_pred = to_image_centric_coordinate(y_curr_pred_oc, box_s_raw_curr)

                # Regularize box scale manually.
                if self.boxreg_regularize:
                    y_curr_pred = regularize_scale(y_prev, y_curr_pred, y0, 0.5, 1.0)

            elif self.sc:
                y_curr_pred_oc, y_curr_pred = get_rectangles_from_hmap(
                    hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, y_prev)
                if self.sc_net:
                    # scale-classification network.
                    target_size = (self.search_size - 1) * self.target_scale / self.search_scale + 1
                    target_init_pad, _, _ = process_image_with_box(
                        x0, y0, crop_size=target_size, scale=1,
                        aspect=example['aspect'], aspect_method=self.aspect_method)
                    target_pred_pad, _, _ = process_image_with_box(
                        x_curr,
                        tf.cond(is_training,
                                lambda: geom.rect_translate_random(y_curr_pred, self.sc_shift_amount),
                                lambda: y_curr_pred),
                        crop_size=target_size, scale=1,
                        aspect=example['aspect'], aspect_method=self.aspect_method)
                    sc_in = tf.concat([target_init_pad, target_pred_pad], -1)
                    if self.sc_pass_hmap:
                        hmap_crop, _, _ = process_image_with_box(
                            hmap_curr_pred_oc_fg, y_curr_pred, crop_size=target_size, scale=1,
                            aspect=example['aspect'], aspect_method=self.aspect_method)
                        sc_in = tf.concat([sc_in, hmap_crop * 255], -1)
                    with tf.variable_scope('scale_classfication', reuse=(t > 0)):
                        sc_out = self.pass_scale_classification(sc_in, is_training)
                        sc_out_list.append(sc_out)
                else:
                    with tf.variable_scope('sc_deconv', reuse=(t > 0)):
                        sc_out = self.pass_deconvolution(scoremap, is_training, num_outputs=self.sc_num_class)
                        sc_out_list.append(sc_out)

                # compute scale and update box.
                p_scale = tf.nn.softmax(sc_out)
                is_max_scale = tf.equal(p_scale, tf.reduce_max(p_scale, axis=-1, keep_dims=True))
                is_max_scale = tf.to_float(is_max_scale)
                scale = tf.reduce_sum(scales * is_max_scale, axis=-1) / tf.reduce_sum(is_max_scale, axis=-1)
                if not self.sc_net:
                    # Scale estimation is convolutional. Take location with max score.
                    sc_active = tf.greater_equal(p_loc_coarse,
                                                 tf.reduce_max(p_loc_coarse, axis=(-2, -1), keep_dims=True) * 0.95)
                    sc_active = tf.to_float(sc_active)
                    scale = (tf.reduce_sum(scale * sc_active, axis=(-2, -1)) /
                             tf.reduce_sum(sc_active, axis=(-2, -1)))
                    sc_active_list.append(sc_active)
                if self.sc_score_threshold > 0:
                    # Use scale = 1 if max location score is not above threshold.
                    max_score = tf.reduce_max(hmap_curr_pred_oc_fg, axis=(1, 2, 3))
                    is_pass = tf.greater_equal(max_score, self.sc_score_threshold)
                    scale = tf.where(is_pass, scale, tf.ones_like(scale))
                y_curr_pred = geom.grow_rect(tf.expand_dims(scale, -1), y_curr_pred)
            else:  # argmax to find center (then use x0's box)
                y_curr_pred_oc, y_curr_pred = get_rectangles_from_hmap(
                    hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, y0)

            # Post-processing.
            y_curr_pred = enforce_inside_box(y_curr_pred, translate=True)

            # Get image-centric outputs. Some are used for visualization purpose.
            hmap_curr_pred = to_image_centric_hmap(hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr,
                                                   (imheight, imwidth))

            # Outputs.
            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr_pred)
            y_pred_oc.append(y_curr_pred_oc_delta if self.boxreg_delta else y_curr_pred_oc)
            hmap_pred_oc.append(hmap_curr_pred_oc)
            hmap_pred_oc_A0.append(hmap_curr_pred_oc_A0 if self.new_target else None)
            box_s_raw.append(box_s_raw_curr)
            box_s_val.append(box_s_val_curr)
            #target.append(target_init) # To visualize what network sees.
            target_init_list.append(target_init)  # To visualize what network sees.
            target_curr_list.append(target_curr if self.new_target else None)  # To visualize what network sees.
            search.append(search_curr)  # To visualize what network sees.

            # Update for next time-step.
            x_prev = x_curr
            # Scheduled sampling. In case label is invalid, use prediction.
            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            y_prev = tf.cond(gt_condition,
                             lambda: tf.where(y_is_valid[:, t], y_curr, y_curr_pred),
                             lambda: y_curr_pred)

        y_pred = tf.stack(y_pred, axis=1)
        hmap_pred = tf.stack(hmap_pred, axis=1)
        y_pred_oc = tf.stack(y_pred_oc, axis=1)
        hmap_pred_oc = tf.stack(hmap_pred_oc, axis=1)
        hmap_pred_oc_A0 = tf.stack(hmap_pred_oc_A0, axis=1) if self.new_target else None
        box_s_raw = tf.stack(box_s_raw, axis=1)
        box_s_val = tf.stack(box_s_val, axis=1)
        #target        = tf.stack(target, axis=1)
        target_init = tf.stack(target_init_list, axis=1)
        target_curr = tf.stack(target_curr_list, axis=1) if self.new_target else None
        search = tf.stack(search, axis=1)

        # Summaries from within model.
        if self.use_hmap_prior:
            with tf.variable_scope('deconvolution', reuse=True):
                hmap_prior = tf.get_variable('hmap_prior')
            with tf.name_scope('model_summary'):
                # Swap channels and (non-existent) batch dimension.
                hmap_prior = tf.transpose(tf.expand_dims(hmap_prior, 0), [3, 1, 2, 0])
                tf.summary.image('hmap_prior', hmap_prior, max_outputs=2,
                                 collections=self.image_summaries_collections)

        for k in hmap_interm.keys():
            if not hmap_interm[k]:
                hmap_interm[k] = None
            else:
                hmap_interm[k] = tf.stack(hmap_interm[k], axis=1)

        outputs = {'y': {'ic': y_pred, 'oc': y_pred_oc},  # NOTE: Do not use 'ic' to compute loss.
                   'hmap': {'ic': hmap_pred, 'oc': hmap_pred_oc},  # NOTE: hmap_pred_oc is no softmax yet.
                   'hmap_A0': hmap_pred_oc_A0,
                   'hmap_interm': hmap_interm,
                   'box_s_raw': box_s_raw,
                   'box_s_val': box_s_val,
                   #'target':    target,
                   'target_init': target_init,
                   'target_curr': target_curr,
                   'search': search,
                   'boxreg_delta': self.boxreg_delta,
                   'sc': {'out': tf.stack(sc_out_list, axis=1),
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
        # gt = {}
        # dbg = {} # dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        # return outputs, state_init, state_final, gt, dbg
        return outputs, state_init, state_final

    def pass_cnn(self, x, is_training, act, is_target=False):
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
        if self.cnn_model == 'custom':
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'fused': True},
                                weights_regularizer=slim.l2_regularizer(self.wd)):
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
                    kernel_size = (self.search_size - 1) * self.target_scale / self.search_scale / total_stride + 1
                    # print 'conv6 kernel size:', kernel_size
                    assert kernel_size > 0
                    # (kernel_size-1) == (frmsz-1) * (target_scale / search_scale) / total_stride
                    assert (kernel_size - 1) * total_stride * self.search_scale == (self.search_size - 1) * self.target_scale
                    assert np.all(np.array(kernel_size) % 2 == 1)
                    x = slim.conv2d(x, 64 if self.light else 256, kernel_size,
                                    activation_fn=get_act(act),
                                    padding='VALID' if is_target else 'SAME',
                                    scope='conv6')
                    if is_target:
                        assert x.shape.as_list()[-3:-1] == [1, 1]

        elif self.cnn_model == 'siamese':  # exactly same as Siamese
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=slim.l2_regularizer(self.wd)):
                x = slim.conv2d(x, 96, 11, stride=self.conv1_stride, scope='conv1')
                x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool1')
                x = slim.conv2d(x, 256, 5, stride=1, scope='conv2')
                x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool2')
                x = slim.conv2d(x, 192, 3, stride=1, scope='conv3')
                x = slim.conv2d(x, 192, 3, stride=1, scope='conv4')
                x = slim.conv2d(x, 128, 3, stride=1, activation_fn=get_act(act), scope='conv5')
        elif self.cnn_model == 'vgg_16':
            # TODO: NEED TO TEST AGAIN. Previously I had activation at the end.. :(
            #with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            #    nets, end_points = alexnet.alexnet_v2(x, spatial_squeeze=False)
            with slim.arg_scope([slim.conv2d],
                                trainable=self.cnn_trainable,
                                variables_collections=[self.cnn_model],
                                weights_regularizer=slim.l2_regularizer(self.wd)):
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

    def pass_cross_correlation(self, search, target):
        ''' Perform cross-correlation (convolution) to find the target object.
        I use channel-wise convolution, instead of normal convolution.
        '''
        dims = target.shape.as_list()
        assert dims[1] % 2 == 1, 'target size has to be odd number: {}'.format(dims[-3:-1])

        # multi-scale targets.
        # NOTE: To confirm the feature in this module, there should be
        # enough training pairs different in scale during training -> data augmentation.
        targets = []
        scales = range(dims[1] - (self.scale_target_num / 2) * 2, dims[1] + (self.scale_target_num / 2) * 2 + 1, 2)
        for s in scales:
            targets.append(tf.image.resize_images(target, [s, s],
                                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                  align_corners=True))

        if self.divide_target:
            assert False, 'Do not use it now.'
            height = target.shape.as_list()[1]
            patchsz = [5]  # TODO: diff sizes
            for n in range(len(patchsz)):
                for i in range(0, height, patchsz[n]):
                    for j in range(0, height, patchsz[n]):
                        #targets.append(target[:,i:i+patchsz[n], j:j+patchsz[n], :])
                        # Instead of divide, use mask operation to preserve patch size.
                        grid_x = tf.expand_dims(tf.range(height), 0)
                        grid_y = tf.expand_dims(tf.range(height), 1)
                        x1 = tf.expand_dims(j, -1)
                        x2 = tf.expand_dims(j + patchsz[n] - 1, -1)
                        y1 = tf.expand_dims(i, -1)
                        y2 = tf.expand_dims(i + patchsz[n] - 1, -1)
                        mask = tf.logical_and(
                            tf.logical_and(tf.less_equal(x1, grid_x), tf.less_equal(grid_x, x2)),
                            tf.logical_and(tf.less_equal(y1, grid_y), tf.less_equal(grid_y, y2)))
                        targets.append(target * tf.expand_dims(tf.cast(mask, tf.float32), -1))
                        weights.append(float(patchsz[n]) / height)

        if self.join_method == 'dot':
            # cross-correlation. (`scale_target_num` # of scoremaps)
            scoremap = []
            for k in range(len(targets)):
                scoremap.append(diag_xcorr(search, targets[k], strides=[1, 1, 1, 1], padding='SAME'))

            if self.scale_target_mode == 'add':
                scoremap = tf.add_n(scoremap)
            elif self.scale_target_mode == 'weight':
                scoremap = tf.concat(scoremap, -1)
                dims_combine = scoremap.shape.as_list()
                with slim.arg_scope([slim.conv2d],
                                    kernel_size=1,
                                    weights_regularizer=slim.l2_regularizer(self.wd)):
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
                            weights_regularizer=slim.l2_regularizer(self.wd),
                            **bnorm_args):
            scoremap = slim.conv2d(scoremap, scope='conv1')
            scoremap = slim.conv2d(scoremap, scope='conv2')
        return scoremap

    def combine_scoremaps(self, scoremap_init, scoremap_curr, is_training):
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
                                weights_regularizer=slim.l2_regularizer(self.wd)):
                scoremap = slim.conv2d(scoremap, scope='conv1')
                scoremap = slim.conv2d(scoremap, scope='conv2')
        elif self.new_target_combine == 'gau':
            scoremap = tf.concat([scoremap_init, scoremap_curr], -1)
            scoremap_residual = tf.identity(scoremap)
            with slim.arg_scope([slim.conv2d],
                                num_outputs=dims[-1],
                                kernel_size=3,
                                weights_regularizer=slim.l2_regularizer(self.wd)):
                scoremap = slim.conv2d(scoremap, rate=2, scope='conv_dilated')
                scoremap = tf.nn.tanh(scoremap) * tf.nn.sigmoid(scoremap)
                scoremap = slim.conv2d(scoremap, kernel_size=1, scope='conv_1x1')
                scoremap += slim.conv2d(scoremap_residual, kernel_size=1, scope='conv_residual')
        else:
            assert False, 'Not available scoremap combine mode.'
        return scoremap

    def pass_interm_supervision(self, x):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            dim = x.shape.as_list()[-1]
            x = slim.conv2d(x, num_outputs=dim, kernel_size=3, scope='conv1')
            x = slim.conv2d(x, num_outputs=2, kernel_size=1, activation_fn=None, scope='conv2')
        return x

    def pass_hourglass_rnn(self, x, rnn_state, is_training):
        dims = x.shape.as_list()
        x_skip = []
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            x_skip.append(x)
            x = slim.conv2d(x, dims[-1] * 2, 5, 2, scope='encoder_conv1')
            x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='encoder_pool1')
            x_skip.append(x)
            x = slim.conv2d(x, dims[-1] * 2, 5, 2, scope='encoder_conv2')
            x = slim.max_pool2d(x, [2, 2], padding='SAME', scope='encoder_pool2')
            x_skip.append(x)
            x, rnn_state = pass_rnn(x, rnn_state, self.rnn_cell_type, self.wd, self.rnn_skip)
            x = slim.conv2d(tf.image.resize_images(x + x_skip[2], [9, 9], align_corners=True),
                            dims[-1] * 2, 3, 1, scope='decoder1')
            x = slim.conv2d(tf.image.resize_images(x + x_skip[1], [33, 33], align_corners=True),
                            dims[-1], 3, 1, scope='decoder2')
            x = slim.conv2d(x + x_skip[0], dims[-1], 3, 1, scope='decoder3')
        return x, rnn_state

    def pass_deconvolution(self, x, is_training, num_outputs=2):
        ''' Upsampling layers.
        The last layer should not have an activation!
        '''
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'fused': True},
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            if self.cnn_model in ['custom', 'siamese']:
                dim = x.shape.as_list()[-1]
                if self.coarse_hmap:  # No upsample layers.
                    x = slim.conv2d(x, num_outputs=dim, kernel_size=3, scope='deconv1')
                    x = slim.conv2d(x, num_outputs=num_outputs, kernel_size=1,
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='deconv2')
                else:
                    x = tf.image.resize_images(x, [(self.search_size - 1) / 4 + 1] * 2, align_corners=True)
                    x = slim.conv2d(x, num_outputs=dim / 2, kernel_size=3, scope='deconv1')
                    x = tf.image.resize_images(x, [(self.search_size - 1) / 2 + 1] * 2, align_corners=True)
                    x = slim.conv2d(x, num_outputs=dim / 4, kernel_size=3, scope='deconv2')
                    x = tf.image.resize_images(x, [self.search_size] * 2, align_corners=True)
                    x = slim.conv2d(x, num_outputs=num_outputs, kernel_size=1,
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='deconv3')
            # elif self.cnn_model == 'vgg_16':
            #     assert False, 'Please update this better before using it..'
            #     x = slim.conv2d(x, num_outputs=512, scope='deconv1')
            #     x = tf.image.resize_images(x, [61, 61])
            #     x = slim.conv2d(x, num_outputs=256, scope='deconv2')
            #     x = tf.image.resize_images(x, [121, 121])
            #     x = slim.conv2d(x, num_outputs=2, scope='deconv3')
            #     x = tf.image.resize_images(x, [self.search_size, self.search_size])
            #     x = slim.conv2d(x, num_outputs=2, activation_fn=None, scope='deconv4')
            else:
                assert False, 'Not available option.'
        return x

    def pass_conv_after_rnn(self, x):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=3,
                            num_outputs=x.shape.as_list()[-1],
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            x = slim.conv2d(x, scope='conv1')
            x = slim.conv2d(x, scope='conv2')
        return x

    def pass_conv_boxreg_branch(self, x, is_training):
        dims = x.shape.as_list()
        with slim.arg_scope([slim.conv2d],
                            kernel_size=1,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'fused': True},
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            x = slim.conv2d(x, num_outputs=dims[-1], scope='conv1')
            x = slim.conv2d(x, num_outputs=dims[-1], scope='conv2')
        return x

    def pass_conv_boxreg_project(self, x, is_training, dim):
        '''Projects to lower dimension after concat with scoremap.'''
        with slim.arg_scope([slim.conv2d],
                            kernel_size=1,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'fused': True},
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            x = slim.conv2d(x, num_outputs=dim, scope='conv1')
            x = slim.conv2d(x, num_outputs=dim, scope='conv2')
        return x

    def pass_regress_box(self, x, is_training):
        ''' Regress output rectangle.
        '''
        dims = x.shape.as_list()
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'fused': True}):
                if not self.light:
                    x = slim.conv2d(x, dims[-1] * 2, 5, 2, padding='VALID', scope='conv1')
                    x = slim.conv2d(x, dims[-1] * 4, 5, 2, padding='VALID', scope='conv2')
                    x = slim.max_pool2d(x, 2, 2, scope='pool1')
                    x = slim.conv2d(x, dims[-1] * 8, 3, 1, padding='VALID', scope='conv3')
                else:
                    x = slim.conv2d(x, dims[-1] * 2, 3, stride=2, scope='conv1')
                    x = slim.conv2d(x, dims[-1] * 2, 3, stride=2, scope='conv2')
                    x = slim.conv2d(x, dims[-1] * 4, 3, stride=2, scope='conv3')
                    x = slim.conv2d(x, dims[-1] * 4, 3, stride=2, scope='conv4')
                    kernel_size = x.shape.as_list()[-3:-1]
                    x = slim.conv2d(x, dims[-1] * 8, kernel_size, padding='VALID', scope='conv5')
                assert x.shape.as_list()[-3:-1] == [1, 1]
                x = slim.flatten(x)
                x = slim.fully_connected(x, 1024, scope='fc1')
                x = slim.fully_connected(x, 1024, scope='fc2')
                x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
        return x

    def pass_scale_classification(self, x, is_training):
        ''' Classification network for scale change.
        '''
        dims = x.shape.as_list()
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training, 'fused': True},
                            weights_regularizer=slim.l2_regularizer(self.wd)):
            x = slim.conv2d(x, 16, 5, 2, padding='VALID', scope='conv1')
            x = slim.conv2d(x, 32, 5, 2, padding='VALID', scope='conv2')
            x = slim.max_pool2d(x, 2, 2, scope='pool1')
            x = slim.conv2d(x, 64, 3, 1, padding='VALID', scope='conv3')
            x = slim.conv2d(x, 128, 3, 1, padding='VALID', scope='conv4')
            x = slim.conv2d(x, 256, 3, 1, padding='VALID', scope='conv5')
            assert x.shape.as_list()[1] == 1
            x = slim.flatten(x)
            x = slim.fully_connected(x, 256, scope='fc1')
            x = slim.fully_connected(x, 256, scope='fc2')
            x = slim.fully_connected(x, self.sc_num_class, activation_fn=None, normalizer_fn=None, scope='fc3')
        return x


def enforce_min_size(x1, y1, x2, y2, min_size, name='min_size'):
    with tf.name_scope(name) as scope:
        # Ensure that x2-x1 > 1
        xc, xs = 0.5 * (x1 + x2), x2 - x1
        yc, ys = 0.5 * (y1 + y2), y2 - y1
        # TODO: Does this propagate NaNs?
        xs = tf.maximum(min_size, xs)
        ys = tf.maximum(min_size, ys)
        x1, x2 = xc - xs / 2, xc + xs / 2
        y1, y2 = yc - ys / 2, yc + ys / 2
        return x1, y1, x2, y2


def enforce_inside_box(y, translate=False, name='inside_box'):
    ''' Force the box to be in range [0,1]
    '''
    assert y.shape.as_list()[-1] == 4
    # inside range [0,1]
    with tf.name_scope(name) as scope:
        if translate:
            dims = tf.shape(y)
            y = tf.reshape(y, [-1, dims[-1]])
            translate_x = tf.maximum(tf.maximum(y[:, 0], y[:, 2]) - 1, 0) + \
                tf.minimum(tf.minimum(y[:, 0], y[:, 2]) - 0, 0)
            translate_y = tf.maximum(tf.maximum(y[:, 1], y[:, 3]) - 1, 0) + \
                tf.minimum(tf.minimum(y[:, 1], y[:, 3]) - 0, 0)
            y = y - tf.stack([translate_x, translate_y] * 2, -1)
            y = tf.reshape(y, dims)
        y = tf.clip_by_value(y, 0.0, 1.0)
    return y


def process_image_with_box(img, box, crop_size, scale, aspect=None, aspect_method=None):
    ''' Crop image using box and scale.

    crop_size: output size after crop-and-resize.
    scale:     uniform scalar for box.
    '''
    if aspect is not None:
        stretch = tf.stack([tf.pow(aspect, 0.5), tf.pow(aspect, -0.5)], axis=-1)
        box = geom.rect_mul(box, stretch)
    box = modify_aspect_ratio(box, aspect_method)
    if aspect is not None:
        box = geom.rect_mul(box, 1. / stretch)

    box = geom.grow_rect(scale, box)
    box_val = geom.rect_intersect(box, geom.unit_rect())

    batch_len = tf.shape(img)[0]
    crop = tf.image.crop_and_resize(img, geom.rect_to_tf_box(box),
                                    box_ind=tf.range(batch_len),
                                    crop_size=[crop_size] * 2,
                                    extrapolation_value=128)
    return crop, box, box_val


def to_image_centric_coordinate(coord, box_s_raw):
    ''' Convert object-centric coordinates to image-centric coordinates.
    Assume that `coord` is either center [x_center,y_center] or box [x1,y1,x2,y2].
    '''
    # scale the size in image-centric and move.
    w_raw = box_s_raw[:, 2] - box_s_raw[:, 0]
    h_raw = box_s_raw[:, 3] - box_s_raw[:, 1]
    if coord.shape.as_list()[-1] == 2:  # center
        x_center = coord[:, 0] * w_raw + box_s_raw[:, 0]
        y_center = coord[:, 1] * h_raw + box_s_raw[:, 1]
        return tf.stack([x_center, y_center], 1)
    elif coord.shape.as_list()[-1] == 4:  # box
        x1 = coord[:, 0] * w_raw + box_s_raw[:, 0]
        y1 = coord[:, 1] * h_raw + box_s_raw[:, 1]
        x2 = coord[:, 2] * w_raw + box_s_raw[:, 0]
        y2 = coord[:, 3] * h_raw + box_s_raw[:, 1]
        return tf.stack([x1, y1, x2, y2], 1)
    else:
        raise ValueError('coord is not expected form.')


def to_object_centric_coordinate(y, box_s_raw, box_s_val):
    ''' Convert image-centric coordinates to object-centric coordinates.
    Assume that `coord` is either center [x_center,y_center] or box [x1,y1,x2,y2].
    '''
    #NOTE: currently only assuming input as box.
    coord_axis = len(y.shape.as_list()) - 1
    x1, y1, x2, y2 = tf.unstack(y, axis=coord_axis)
    x1_raw, y1_raw, x2_raw, y2_raw = tf.unstack(box_s_raw, axis=coord_axis)
    x1_val, y1_val, x2_val, y2_val = tf.unstack(box_s_val, axis=coord_axis)
    s_raw_w = x2_raw - x1_raw  # [0,1] range
    s_raw_h = y2_raw - y1_raw  # [0,1] range
    # NOTE: Remember that due to the limitation of `enforce_inside_box`,
    # `process_search_with_box` can yield box_s with size of 0.
    #with tf.control_dependencies([tf.assert_greater(s_raw_w, 0.0), tf.assert_greater(s_raw_h, 0.0)]):
    x1_oc = (x1 - x1_raw) / (s_raw_w + 1e-5)
    y1_oc = (y1 - y1_raw) / (s_raw_h + 1e-5)
    x2_oc = (x2 - x1_raw) / (s_raw_w + 1e-5)
    y2_oc = (y2 - y1_raw) / (s_raw_h + 1e-5)
    y_oc = tf.stack([x1_oc, y1_oc, x2_oc, y2_oc], coord_axis)
    return y_oc


def to_image_centric_hmap(hmap_pred_oc, box_s_raw, box_s_val, im_size):
    ''' Convert object-centric hmap to image-centric hmap.
    Input hmap is assumed to be softmax-ed (i.e., range [0,1]) and foreground only.
    '''
    inv_box_s = geom.crop_inverse(box_s_raw)
    batch_len = tf.shape(hmap_pred_oc)[0]
    # TODO: Set extrapolation_value.
    hmap_pred_ic = tf.image.crop_and_resize(hmap_pred_oc, geom.rect_to_tf_box(inv_box_s),
                                            box_ind=tf.range(batch_len),
                                            crop_size=im_size,
                                            extrapolation_value=None)
    return hmap_pred_ic


def regularize_scale(y_prev, y_curr, y0=None, local_bound=0.01, global_bound=0.1):
    ''' This function regularize (only) scale of new box.
    It is used when `boxreg` option is enabled.
    '''
    w_prev = y_prev[:, 2] - y_prev[:, 0]
    h_prev = y_prev[:, 3] - y_prev[:, 1]
    c_prev = tf.stack([(y_prev[:, 2] + y_prev[:, 0]), (y_prev[:, 3] + y_prev[:, 1])], 1) / 2.0
    w_curr = y_curr[:, 2] - y_curr[:, 0]
    h_curr = y_curr[:, 3] - y_curr[:, 1]
    c_curr = tf.stack([(y_curr[:, 2] + y_curr[:, 0]), (y_curr[:, 3] + y_curr[:, 1])], 1) / 2.0

    # add local bound
    w_reg = tf.clip_by_value(w_curr, w_prev * (1 - local_bound), w_prev * (1 + local_bound))
    h_reg = tf.clip_by_value(h_curr, h_prev * (1 - local_bound), h_prev * (1 + local_bound))

    # add global bound w.r.t. y0
    if y0 is not None:
        w0 = y0[:, 2] - y0[:, 0]
        h0 = y0[:, 3] - y0[:, 1]
        w_reg = tf.clip_by_value(w_reg, w0 * (1 - global_bound), w0 * (1 + global_bound))
        h_reg = tf.clip_by_value(h_reg, h0 * (1 - global_bound), h0 * (1 + global_bound))

    y_reg = tf.stack([c_curr[:, 0] - w_reg / 2.0, c_curr[:, 1] - h_reg / 2.0,
                      c_curr[:, 0] + w_reg / 2.0, c_curr[:, 1] + h_reg / 2.0], 1)
    return y_reg


def get_rectangles_from_hmap(hmap_oc_fg, box_s_raw, box_s_val, y_ref):
    center_oc = util.find_center_in_scoremap(hmap_oc_fg)
    center = to_image_centric_coordinate(center_oc, box_s_raw)
    y_ref_size = tf.stack([y_ref[:, 2] - y_ref[:, 0], y_ref[:, 3] - y_ref[:, 1]], 1)
    y_tmp = tf.concat([center - y_ref_size * 0.5, center + y_ref_size * 0.5], 1)
    y_tmp_oc = to_object_centric_coordinate(y_tmp, box_s_raw, box_s_val)
    return y_tmp_oc, y_tmp


def get_initial_rnn_state(cell_type, cell_size, num_layers=1):
    if cell_type == 'lstm':
        state = [{'h': None, 'c': None} for l in range(num_layers)]
    #elif cell_type == 'gru':
    elif cell_type in ['gru', 'gau']:
        state = [{'h': None} for l in range(num_layers)]
    else:
        assert False, 'Not available cell type.'

    with tf.variable_scope('rnn_state'):
        for l in range(num_layers):
            for key in state[l].keys():
                state[l][key] = tf.fill(cell_size, 0.0, name='layer{}/{}'.format(l, key))
    return state


def pass_rnn(x, state, cell, wd, skip=False):
    ''' Convolutional RNN.
    Currently, `dense` skip type is supported; All hidden states are summed.
    '''
    # TODO:
    # 1. (LSTM) Initialize forget bias 1.
    # 2. (LSTM) Try LSTM architecture as defined in "An Empirical Exploration of-".
    # 2. Try channel-wise convolution.

    # skip state indicating which state will be connected to.
    if skip:
        skip_state = range(state['h'].shape.as_list()[1])  # All states (dense). # TODO: sparse skip. stride?
    else:
        skip_state = [-1]  # only previous state.

    if cell == 'lstm':
        h_prev, c_prev = state['h'], state['c']
        with slim.arg_scope([slim.conv2d],
                            num_outputs=h_prev.shape.as_list()[-1],
                            kernel_size=3,
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(wd)):
            #it = tf.nn.sigmoid(slim.conv2d(x, scope='xi') + slim.conv2d(h_prev, scope='hi'))
            #ft = tf.nn.sigmoid(slim.conv2d(x, scope='xf') + slim.conv2d(h_prev, scope='hf'))
            #ct_tilda = tf.nn.tanh(slim.conv2d(x, scope='xc') + slim.conv2d(h_prev, scope='hc'))
            #ct = (ft * c_prev) + (it * ct_tilda)
            #ot = tf.nn.sigmoid(slim.conv2d(x, scope='xo') + slim.conv2d(h_prev, scope='ho'))
            #ht = ot * tf.nn.tanh(ct)
            #it = tf.nn.sigmoid(slim.conv2d(x, scope='xi') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='hi_{}'.format(s)) for s in skip_state]))
            #ft = tf.nn.sigmoid(slim.conv2d(x, scope='xf') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='hf_{}'.format(s)) for s in skip_state]))
            #ct_tilda = tf.nn.tanh(slim.conv2d(x, scope='xc') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='hc_{}'.format(s)) for s in skip_state]))
            #ct = (tf.reduce_sum(tf.expand_dims(ft, 1) * c_prev, 1)) + (it * ct_tilda)
            #ot = tf.nn.sigmoid(slim.conv2d(x, scope='xo') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='ho_{}'.format(s)) for s in skip_state]))
            #ht = ot * tf.nn.tanh(ct)
            it = tf.nn.sigmoid(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='i'))
            ft = tf.nn.sigmoid(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='f'))
            ct_tilda = tf.nn.tanh(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='c'))
            ct = (tf.reduce_sum(tf.expand_dims(ft, 1) * c_prev, 1)) + (it * ct_tilda)
            ot = tf.nn.sigmoid(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='o'))
            ht = ot * tf.nn.tanh(ct)
        output = ht
        state['h'] = tf.concat([state['h'][:, 1:], tf.expand_dims(ht, 1)], 1)
        state['c'] = tf.concat([state['c'][:, 1:], tf.expand_dims(ct, 1)], 1)
    elif cell == 'gru':
        h_prev = state['h']
        with slim.arg_scope([slim.conv2d],
                            num_outputs=h_prev.shape.as_list()[-1],
                            kernel_size=3,
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(wd)):
            #rt = tf.nn.sigmoid(slim.conv2d(x, scope='xr') + slim.conv2d(h_prev, scope='hr'))
            #zt = tf.nn.sigmoid(slim.conv2d(x, scope='xz') + slim.conv2d(h_prev, scope='hz'))
            #h_tilda = tf.nn.tanh(slim.conv2d(x, scope='xh') + slim.conv2d(rt * h_prev, scope='hh'))
            #ht = zt * h_prev + (1-zt) * h_tilda
            #rt = tf.nn.sigmoid(slim.conv2d(x, scope='xr') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='hr_{}'.format(s)) for s in skip_state]))
            #zt = tf.nn.sigmoid(slim.conv2d(x, scope='xz') +
            #        tf.add_n([slim.conv2d(h_prev[:,s], scope='hz_{}'.format(s)) for s in skip_state]))
            #h_tilda = tf.nn.tanh(slim.conv2d(x, scope='xh') +
            #        tf.add_n([slim.conv2d(rt * h_prev[:,s], scope='hh_{}'.format(s)) for s in skip_state]))
            #ht = tf.reduce_sum(tf.expand_dims(zt, 1) * h_prev, 1) + (1-zt) * h_tilda
            rt = tf.nn.sigmoid(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='r'))
            zt = tf.nn.sigmoid(slim.conv2d(tf.concat([x] + [h_prev[:, s] for s in skip_state], -1), scope='z'))
            h_tilda = tf.nn.tanh(slim.conv2d(tf.concat([x] + [rt * h_prev[:, s] for s in skip_state], -1), scope='h'))
            ht = tf.reduce_sum(tf.expand_dims(zt, 1) * h_prev, 1) + (1 - zt) * h_tilda
        output = ht
        state['h'] = tf.concat([state['h'][:, 1:], tf.expand_dims(ht, 1)], 1)
    elif cell == 'gau':
        h_prev = state['h']
        with slim.arg_scope([slim.conv2d],
                            num_outputs=h_prev.shape.as_list()[-1],
                            kernel_size=3,
                            activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(wd)):
            xh = tf.concat([x] + [h_prev[:, s] for s in skip_state], -1)
            ht = tf.nn.tanh(slim.conv2d(xh, scope='f')) * tf.nn.sigmoid(slim.conv2d(xh, scope='g'))
            ht = slim.conv2d(ht, kernel_size=1, scope='1x1')
            ht = ht + x  # final residual connection.
        output = ht
        state['h'] = tf.concat([state['h'][:, 1:], tf.expand_dims(ht, 1)], 1)
    else:
        assert False, 'Not available cell type.'
    return output, state


def convert_rec_to_heatmap(rec, output_size, min_size=None, mode='box', Gaussian=False, radius_pos=0.1, sigma=0.3):
    '''Create heatmap from rectangle
    Args:
        rec: [batchsz x ntimesteps x 4] ground-truth rectangle labels
    Return:
        heatmap: [batchsz x ntimesteps x o.frmsz x o.frmsz x 2] # fg + bg
    '''
    with tf.name_scope('heatmaps') as scope:
        # JV: This causes a seg-fault in save when two loss functions are constructed?!
        # masks = []
        # for t in range(o.ntimesteps):
        #     masks.append(get_masks_from_rectangles(rec[:,t], o, kind='bg'))
        # return tf.stack(masks, axis=1, name=scope)
        rec, unmerge = merge_dims(rec, 0, 2)
        masks = get_masks_from_rectangles(rec, output_size, kind='bg', min_size=min_size,
                                          mode=mode, Gaussian=Gaussian, radius_pos=radius_pos, sigma=sigma)
        return unmerge(masks, 0)


def get_masks_from_rectangles(rec, output_size, kind='fg', typecast=True, min_size=None, mode='box', Gaussian=False, radius_pos=0.2, sigma=0.3, name='mask'):
    '''
    Args:
        rec: [..., 4]
            Rectangles in standard format (xmin, ymin, xmax, ymax).
        output_size: (height, width)
            If None, then height = width = o.frmsz.

    Returns:
        [..., height, width]
    '''
    with tf.name_scope(name) as scope:
        size_y, size_x = output_size
        # create mask using rec; typically rec=y_prev
        # rec -- [..., 4]
        # JV: Allow different output size.
        # rec *= float(o.frmsz)
        rec = geom.rect_mul(rec, tf.to_float([size_x - 1, size_y - 1]))
        # x1, y1, x2, y2 -- [...]
        x1, y1, x2, y2 = tf.unstack(rec, axis=1)
        if min_size is not None:
            x1, y1, x2, y2 = enforce_min_size(x1, y1, x2, y2, min_size=min_size)
        # grid_x -- [1, width]
        # grid_y -- [height, 1]
        grid_x = tf.expand_dims(tf.cast(tf.range(size_x), tf.float32), 0)
        grid_y = tf.expand_dims(tf.cast(tf.range(size_y), tf.float32), 1)
        # resize tensors so that they can be compared
        # x1, y1, x2, y2 -- [..., 1, 1]
        x1 = tf.expand_dims(tf.expand_dims(x1, -1), -1)
        x2 = tf.expand_dims(tf.expand_dims(x2, -1), -1)
        y1 = tf.expand_dims(tf.expand_dims(y1, -1), -1)
        y2 = tf.expand_dims(tf.expand_dims(y2, -1), -1)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width, height = x2 - x1, y2 - y1
        # masks -- [b, frmsz, frmsz]
        if mode == 'box':
            if not Gaussian:
                masks = tf.logical_and(
                    tf.logical_and(tf.less_equal(x1, grid_x),
                                   tf.less_equal(grid_x, x2)),
                    tf.logical_and(tf.less_equal(y1, grid_y),
                                   tf.less_equal(grid_y, y2)))
            else:  # TODO: Need debug this properly.
                x_sigma = width * sigma  # TODO: can be better..
                y_sigma = height * sigma
                masks = tf.exp(-(tf.square(grid_x - x_center) / (2 * tf.square(x_sigma)) +
                                 tf.square(grid_y - y_center) / (2 * tf.square(y_sigma))))
        elif mode == 'center':
            obj_diam = 0.5 * (width + height)
            r = tf.sqrt(tf.square(grid_x - x_center) + tf.square(grid_y - y_center)) / obj_diam
            if not Gaussian:
                masks = tf.less_equal(r, radius_pos)
            else:
                masks = tf.exp(-0.5 * tf.square(r) / tf.square(sigma))

        if kind == 'fg':  # only foreground mask
            # JV: Make this more general.
            # masks = tf.expand_dims(masks, 3) # to have channel dim
            masks = tf.expand_dims(masks, -1)  # to have channel dim
        elif kind == 'bg':  # add background mask
            if not Gaussian:
                masks_bg = tf.logical_not(masks)
            else:
                masks_bg = 1.0 - masks
            # JV: Make this more general.
            # masks = concat(
            #         (tf.expand_dims(masks,3), tf.expand_dims(masks_bg,3)), 3)
            masks = tf.stack([masks, masks_bg], -1)
        if typecast:  # type cast so that it can be concatenated with x
            masks = tf.cast(masks, tf.float32)
        return masks


def get_loss(example, outputs, search_scale, search_size, losses, perspective, heatmap_params,
             name='loss'):
    with tf.name_scope(name) as scope:
        x_shape = example['x'].shape.as_list()
        ntimesteps = x_shape[1]
        imheight, imwidth = x_shape[2], x_shape[3]

        y_gt = {'ic': None, 'oc': None}
        hmap_gt = {'ic': None, 'oc': None}

        y_gt['ic'] = example['y']
        y_gt['oc'] = to_object_centric_coordinate(example['y'], outputs['box_s_raw'], outputs['box_s_val'])
        hmap_gt['oc'] = convert_rec_to_heatmap(y_gt['oc'], [search_size] * 2, min_size=1.0, **heatmap_params)
        hmap_gt['ic'] = convert_rec_to_heatmap(y_gt['ic'], (imheight, imwidth), min_size=1.0, **heatmap_params)
        if outputs['sc']:
            assert 'sc' in losses
            sc_gt = compute_scale_classification_gt(example, outputs['sc']['scales'])

        # Regress displacement rather than absolute location. Update y_gt.
        if outputs['boxreg_delta']:
            y_gt['ic'] = y_gt['ic'] - tf.concat([tf.expand_dims(example['y0'], 1), y_gt['ic'][:, :ntimesteps - 1]], 1)
            delta0 = y_gt['oc'][:, 0] - tf.stack([0.5 - 1. / search_scale / 2., 0.5 + 1. / search_scale / 2.] * 2)
            y_gt['oc'] = tf.concat([tf.expand_dims(delta0, 1), y_gt['oc'][:, 1:] - y_gt['oc'][:, :ntimesteps - 1]], 1)

        assert(y_gt['ic'].get_shape().as_list()[1] == ntimesteps)

        for key in outputs['hmap_interm']:
            if outputs['hmap_interm'][key] is not None:
                pred_size = outputs['hmap_interm'][key].shape.as_list()[2:4]
                hmap_interm_gt, unmerge = merge_dims(hmap_gt['oc'], 0, 2)
                hmap_interm_gt = tf.image.resize_images(hmap_interm_gt, pred_size,
                                                        method=tf.image.ResizeMethod.BILINEAR,
                                                        align_corners=True)
                hmap_interm_gt = unmerge(hmap_interm_gt, axis=0)
                break

        if 'oc' in outputs['hmap']:
            # Resize GT heatmap to match size of prediction if necessary.
            pred_size = outputs['hmap']['oc'].shape.as_list()[2:4]
            assert all(pred_size)  # Must not be None.
            gt_size = hmap_gt['oc'].shape.as_list()[2:4]
            if gt_size != pred_size:
                hmap_gt['oc'], unmerge = merge_dims(hmap_gt['oc'], 0, 2)
                hmap_gt['oc'] = tf.image.resize_images(hmap_gt['oc'], pred_size,
                                                       method=tf.image.ResizeMethod.BILINEAR,
                                                       align_corners=True)
                hmap_gt['oc'] = unmerge(hmap_gt['oc'], axis=0)

        loss_vars = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in losses or 'l1_relative' in losses:
            y_gt_valid = tf.boolean_mask(y_gt[perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][perspective], example['y_is_valid'])
            loss_l1 = tf.reduce_mean(tf.abs(y_gt_valid - y_pred_valid), axis=-1)
            if 'l1' in losses:
                loss_vars['l1'] = tf.reduce_mean(loss_l1)
            if 'l1_relative' in losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:, 2] - y_gt_valid[:, 0])
                y_size = tf.abs(y_gt_valid[:, 3] - y_gt_valid[:, 1])
                size = tf.stack([x_size, y_size], axis=-1)
                loss_l1_relative = loss_l1 / (tf.reduce_mean(size, axis=-1) + 0.05)
                loss_vars['l1_relative'] = tf.reduce_mean(loss_l1_relative)

        # CLE (center location error). Measured in l2 distance.
        if 'cle' in losses or 'cle_relative' in losses:
            y_gt_valid = tf.boolean_mask(y_gt[perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][perspective], example['y_is_valid'])
            x_center = (y_gt_valid[:, 2] + y_gt_valid[:, 0]) * 0.5
            y_center = (y_gt_valid[:, 3] + y_gt_valid[:, 1]) * 0.5
            center = tf.stack([x_center, y_center], axis=-1)
            x_pred_center = (y_pred_valid[:, 2] + y_pred_valid[:, 0]) * 0.5
            y_pred_center = (y_pred_valid[:, 3] + y_pred_valid[:, 1]) * 0.5
            pred_center = tf.stack([x_pred_center, y_pred_center], axis=-1)
            loss_cle = tf.norm(center - pred_center, axis=-1)
            if 'cle' in losses:
                loss_vars['cle'] = tf.reduce_mean(loss_cle)
            if 'cle_relative' in losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:, 2] - y_gt_valid[:, 0])
                y_size = tf.abs(y_gt_valid[:, 3] - y_gt_valid[:, 1])
                size = tf.stack([x_size, y_size], axis=-1)
                radius = tf.exp(tf.reduce_mean(tf.log(size), axis=-1))
                loss_cle_relative = loss_cle / (radius + 0.05)
                loss_vars['cle_relative'] = tf.reduce_mean(loss_cle_relative)

        # Cross-entropy between probabilty maps (need to change label)
        if 'ce' in losses or 'ce_balanced' in losses:
            hmap_gt_valid = tf.boolean_mask(hmap_gt[perspective], example['y_is_valid'])
            hmap_pred_valid = tf.boolean_mask(outputs['hmap'][perspective], example['y_is_valid'])
            # hmap is [valid_images, height, width, 2]
            count = tf.reduce_sum(hmap_gt_valid, axis=(1, 2), keep_dims=True)
            class_weight = 0.5 / tf.cast(count + 1, tf.float32)
            weight = tf.reduce_sum(hmap_gt_valid * class_weight, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=hmap_gt_valid,
                logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            if 'ce' in losses:
                loss_vars['ce'] = tf.reduce_mean(loss_ce)
            if 'ce_balanced' in losses:
                loss_vars['ce_balanced'] = tf.reduce_mean(
                    tf.reduce_sum(weight * loss_ce, axis=(1, 2)))

        # TODO: Make it neat if things work well.
        if outputs['hmap_A0'] is not None:
            hmap_gt_valid = tf.boolean_mask(hmap_gt[perspective], example['y_is_valid'])
            hmap_pred_valid = tf.boolean_mask(outputs['hmap_A0'], example['y_is_valid'])
            # hmap is [valid_images, height, width, 2]
            count = tf.reduce_sum(hmap_gt_valid, axis=(1, 2), keep_dims=True)
            class_weight = 0.5 / tf.cast(count + 1, tf.float32)
            weight = tf.reduce_sum(hmap_gt_valid * class_weight, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                labels=hmap_gt_valid,
                logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            loss_vars['ce_A0'] = tf.reduce_mean(loss_ce)

        for key in outputs['hmap_interm']:
            if outputs['hmap_interm'][key] is not None:
                hmap_gt_valid = tf.boolean_mask(hmap_interm_gt, example['y_is_valid'])
                hmap_pred_valid = tf.boolean_mask(outputs['hmap_interm'][key], example['y_is_valid'])
                # Flatten to feed into softmax_cross_entropy_with_logits.
                hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
                hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
                loss_ce_interm = tf.nn.softmax_cross_entropy_with_logits(
                    labels=hmap_gt_valid,
                    logits=hmap_pred_valid)
                loss_ce_interm = unmerge(loss_ce_interm, 0)
                loss_vars['ce_{}'.format(key)] = tf.reduce_mean(loss_ce_interm)

        if 'sc' in losses:
            sc_rank = len(outputs['sc']['out'].shape)
            if sc_rank == 5:
                # Use same GT for all spatial positions.
                shape = outputs['sc']['out'].shape.as_list()
                sc_gt = tf.expand_dims(tf.expand_dims(sc_gt, -2), -2)
                sc_gt = tf.tile(sc_gt, [1, 1, shape[2], shape[3], 1])
            sc_gt_valid = tf.boolean_mask(sc_gt, example['y_is_valid'])
            sc_pred_valid = tf.boolean_mask(outputs['sc']['out'], example['y_is_valid'])
            loss_sc = tf.nn.softmax_cross_entropy_with_logits(labels=sc_gt_valid, logits=sc_pred_valid)
            if sc_rank == 5:
                # Reduce over spatial predictions. Use active elements.
                sc_active_valid = tf.boolean_mask(outputs['sc']['active'], example['y_is_valid'])
                loss_sc = (tf.reduce_sum(sc_active_valid * loss_sc, axis=(-2, -1)) /
                           tf.reduce_sum(sc_active_valid, axis=(-2, -1)))
            ##sc_gt_valid = tf.boolean_mask(sc_gt, example['y_is_valid'])
            ##sc_pred_valid = tf.boolean_mask(outputs['sc']['out'], example['y_is_valid'])
            ##sc_gt_valid, unmerge = merge_dims(sc_gt_valid, 0, 1)
            ##sc_pred_valid, _ = merge_dims(sc_pred_valid, 0, 1)
            ##loss_sc = tf.nn.softmax_cross_entropy_with_logits(
            ##        labels=sc_gt_valid,
            ##        logits=sc_pred_valid)
            ##loss_sc = unmerge(loss_sc, 0)
            loss_vars['sc'] = tf.reduce_mean(loss_sc)

        # Reconstruction loss using generalized Charbonnier penalty
        if 'recon' in losses:
            alpha = 0.25
            s_prev_valid = tf.boolean_mask(outputs['s_prev'], example['y_is_valid'])
            s_recon_valid = tf.boolean_mask(outputs['s_recon'], example['y_is_valid'])
            charbonnier_penalty = tf.pow(tf.square(s_prev_valid - s_recon_valid) + 1e-10, alpha)
            loss_vars['recon'] = tf.reduce_mean(charbonnier_penalty)

        # with tf.name_scope('summary'):
        #     for name, loss in loss_vars.iteritems():
        #         tf.summary.scalar(name, loss)

        gt = {}
        #gt['y']    = {'ic': y_gt['ic'],    'oc': y_gt['oc']}
        gt['hmap'] = {'ic': hmap_gt['ic'], 'oc': hmap_gt['oc']}  # for visualization in summary.
        # return tf.reduce_sum(loss_vars.values(), name=scope), gt
        return loss_vars, gt


def compute_scale_classification_gt(example, scales):
    obj_min, obj_max = geom.rect_min_max(tf.concat([tf.expand_dims(example['y0'], 1), example['y']], 1))
    obj_center, obj_size = 0.5 * (obj_min + obj_max), obj_max - obj_min
    diam = tf.reduce_mean(obj_size, axis=-1)  # 0.5*(width+height)
    sc_ratio = tf.divide(diam[:, 1:], diam[:, :-1])
    sc_gt = []
    for i in range(len(scales)):
        if i == 0:
            sc_gt.append(tf.less(sc_ratio, scales[i]))
        elif i < len(scales) / 2:
            sc_gt.append(tf.logical_and(tf.greater_equal(sc_ratio, scales[i - 1]), tf.less(sc_ratio, scales[i])))
        elif i == len(scales) / 2:
            sc_gt.append(tf.logical_and(tf.greater_equal(sc_ratio, scales[i - 1]), tf.less(sc_ratio, scales[i + 1])))
        elif i > len(scales) / 2 and not i == len(scales) - 1:
            sc_gt.append(tf.logical_and(tf.greater_equal(sc_ratio, scales[i]), tf.less(sc_ratio, scales[i + 1])))
        elif i == len(scales) - 1:
            sc_gt.append(tf.greater_equal(sc_ratio, scales[i]))
        else:
            assert False
    return tf.stack(sc_gt, -1)


def add_summaries(example, outputs, gt, image_summaries_collections):
    ntimesteps = example['x'].shape.as_list()[1]
    assert ntimesteps is not None
    # JV: This is done in train.py now.
    # if 'y' in outputs:
    #     boxes = tf.summary.image('box',
    #         _draw_bounding_boxes(example, outputs),
    #         max_outputs=ntimesteps+1, collections=image_summaries_collections)
    # Produce an image summary of the heatmap prediction (ic and oc).
    if 'hmap' in outputs:
        for key in outputs['hmap']:
            hmap = tf.summary.image('hmap_pred_{}'.format(key),
                                    _draw_heatmap(outputs, gt, pred=True, perspective=key),
                                    max_outputs=ntimesteps + 1, collections=image_summaries_collections)
    # Produce an image summary of the heatmap gt (ic and oc).
    if 'hmap' in gt:
        for key in gt['hmap']:
            hmap = tf.summary.image('hmap_gt_{}'.format(key),
                                    _draw_heatmap(outputs, gt, pred=False, perspective=key),
                                    max_outputs=ntimesteps + 1, collections=image_summaries_collections)
    # Produce an image summary of target and search images (input to CNNs).
    for key in ['search', 'target_init', 'target_curr']:
        if key in outputs:
            if outputs[key] is None: continue
            input_image = tf.summary.image('cnn_input_{}'.format(key),
                                           _draw_input_image(outputs, key, name='draw_{}'.format(key)),
                                           max_outputs=ntimesteps + 1, collections=image_summaries_collections)


def _draw_bounding_boxes(example, outputs, time_stride=1, name='draw_box'):
    # Note: This will produce INT_MIN when casting NaN to int.
    with tf.name_scope(name) as scope:
        # example['x']       -- [b, t, h, w, 3]
        # example['y']       -- [b, t, 4]
        # outputs['y'] -- [b, t, 4]
        # Just do the first example in the batch.
        image = (1.0 / 255) * example['x'][0][::time_stride]
        y_gt = example['y'][0][::time_stride]
        y_pred = outputs['y']['ic'][0][::time_stride]
        # image  = tf.concat((tf.expand_dims(model.state_init['x'][0], 0),  image), 0) # add init frame
        # y_gt   = tf.concat((tf.expand_dims(model.state_init['y'][0], 0),   y_gt), 0) # add init y_gt
        # y_pred = tf.concat((tf.expand_dims(model.state_init['y'][0], 0), y_pred), 0) # add init y_gt for pred too
        y = tf.stack([y_gt, y_pred], axis=1)
        coords = tf.unstack(y, axis=2)
        boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=2)
        return tf.image.draw_bounding_boxes(image, boxes, name=scope)


def _draw_heatmap(outputs, gt, pred, perspective, time_stride=1, name='draw_heatmap'):
    with tf.name_scope(name) as scope:
        # outputs['hmap'] -- [b, t, frmsz, frmsz, 2]
        if pred:
            p = outputs['hmap'][perspective][0, ::time_stride]
            if perspective == 'oc':
                p = tf.nn.softmax(p)
        else:
            p = gt['hmap'][perspective][0, ::time_stride]

        # JV: Not sure what model.state['hmap'] is, and...
        # JV: concat() fails when hmap is coarse (lower resolution than input image).
        # hmaps = tf.concat((model.state['hmap'][0][0:1], p[:,:,:,0:1]), 0) # add init hmap
        hmaps = p[:, :, :, 0:1]
        # Convert to uint8 for absolute scale.
        hmaps = tf.image.convert_image_dtype(hmaps, tf.uint8)
        return hmaps

# def _draw_flow_fields(model, key, time_stride=1, name='draw_flow_fields'):
#     with tf.name_scope(name) as scope:
#         if key == 'u':
#             input_image = tf.expand_dims(model.outputs['flow'][0,::time_stride, :, :, 0], -1)
#         elif key =='v':
#             input_image = tf.expand_dims(model.outputs['flow'][0,::time_stride, :, :, 1], -1)
#         else:
#             assert False , 'No available flow fields'
#         return input_image


def _draw_input_image(outputs, key, time_stride=1, name='draw_input_image'):
    with tf.name_scope(name) as scope:
        input_image = outputs[key][0, ::time_stride]
        if key == 'search':
            # if outputs['boxreg_delta']:
            #     y_pred_delta = outputs['y']['oc'][0][::time_stride]
            #     y_pred = y_pred_delta + tf.stack([0.5 - 1./o.search_scale/2., 0.5 + 1./o.search_scale/2.]*2)
            # else:
            #     y_pred = outputs['y']['oc'][0][::time_stride]
            assert not outputs['boxreg_delta']
            y_pred = outputs['y']['oc'][0][::time_stride]
            coords = tf.unstack(y_pred, axis=1)
            boxes = tf.stack([coords[i] for i in [1, 0, 3, 2]], axis=1)
            boxes = tf.expand_dims(boxes, 1)
            return tf.image.draw_bounding_boxes(input_image, boxes, name=scope)
        else:
            return input_image
        return input_image

# def _draw_memory_state(model, mtype, time_stride=1, name='draw_memory_states'):
#     with tf.name_scope(name) as scope:
#         p = tf.nn.softmax(model.memory[mtype][0,::time_stride])
#         return tf.cast(tf.round(255*p[:,:,:,0:1]), tf.uint8)
