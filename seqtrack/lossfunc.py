def get_loss(example, outputs, gt, o, summaries_collections=None, name='loss'):
    with tf.name_scope(name) as scope:
        y_gt    = {'ic': None, 'oc': None}
        hmap_gt = {'ic': None, 'oc': None}

        y_gt['ic'] = example['y']
        y_gt['oc'] = to_object_centric_coordinate(example['y'], outputs['box_s_raw'], outputs['box_s_val'], o)
        hmap_gt['oc'] = convert_rec_to_heatmap(y_gt['oc'], o, min_size=1.0, **o.heatmap_params)
        hmap_gt['ic'] = convert_rec_to_heatmap(y_gt['ic'], o, min_size=1.0, **o.heatmap_params)
        if outputs['sc']:
            assert 'sc' in o.losses
            sc_gt = compute_scale_classification_gt(example, outputs['sc']['scales'])

        # Regress displacement rather than absolute location. Update y_gt.
        if outputs['boxreg_delta']:
            y_gt['ic'] = y_gt['ic'] - tf.concat([tf.expand_dims(example['y0'],1), y_gt['ic'][:,:o.ntimesteps-1]],1) 
            delta0 = y_gt['oc'][:,0] - tf.stack([0.5 - 1./o.search_scale/2., 0.5 + 1./o.search_scale/2.]*2)
            y_gt['oc'] = tf.concat([tf.expand_dims(delta0,1), y_gt['oc'][:,1:]-y_gt['oc'][:,:o.ntimesteps-1]], 1)

        assert(y_gt['ic'].get_shape().as_list()[1] == o.ntimesteps)

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
            assert all(pred_size) # Must not be None.
            gt_size = hmap_gt['oc'].shape.as_list()[2:4]
            if gt_size != pred_size:
                hmap_gt['oc'], unmerge = merge_dims(hmap_gt['oc'], 0, 2)
                hmap_gt['oc'] = tf.image.resize_images(hmap_gt['oc'], pred_size,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True)
                hmap_gt['oc'] = unmerge(hmap_gt['oc'], axis=0)

        losses = dict()

        # l1 distances for left-top and right-bottom
        if 'l1' in o.losses or 'l1_relative' in o.losses:
            y_gt_valid   = tf.boolean_mask(y_gt[o.perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][o.perspective], example['y_is_valid'])
            loss_l1 = tf.reduce_mean(tf.abs(y_gt_valid - y_pred_valid), axis=-1)
            if 'l1' in o.losses:
                losses['l1'] = tf.reduce_mean(loss_l1)
            if 'l1_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:,2] - y_gt_valid[:,0])
                y_size = tf.abs(y_gt_valid[:,3] - y_gt_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                loss_l1_relative = loss_l1 / (tf.reduce_mean(size, axis=-1) + 0.05)
                losses['l1_relative'] = tf.reduce_mean(loss_l1_relative)

        # CLE (center location error). Measured in l2 distance.
        if 'cle' in o.losses or 'cle_relative' in o.losses:
            y_gt_valid   = tf.boolean_mask(y_gt[o.perspective], example['y_is_valid'])
            y_pred_valid = tf.boolean_mask(outputs['y'][o.perspective], example['y_is_valid'])
            x_center = (y_gt_valid[:,2] + y_gt_valid[:,0]) * 0.5
            y_center = (y_gt_valid[:,3] + y_gt_valid[:,1]) * 0.5
            center = tf.stack([x_center, y_center], axis=-1)
            x_pred_center = (y_pred_valid[:,2] + y_pred_valid[:,0]) * 0.5
            y_pred_center = (y_pred_valid[:,3] + y_pred_valid[:,1]) * 0.5
            pred_center = tf.stack([x_pred_center, y_pred_center], axis=-1)
            loss_cle = tf.norm(center - pred_center, axis=-1)
            if 'cle' in o.losses:
                losses['cle'] = tf.reduce_mean(loss_cle)
            if 'cle_relative' in o.losses:
                # TODO: Reduce code duplication?
                x_size = tf.abs(y_gt_valid[:,2] - y_gt_valid[:,0])
                y_size = tf.abs(y_gt_valid[:,3] - y_gt_valid[:,1])
                size = tf.stack([x_size, y_size], axis=-1)
                radius = tf.exp(tf.reduce_mean(tf.log(size), axis=-1))
                loss_cle_relative = loss_cle / (radius + 0.05)
                losses['cle_relative'] = tf.reduce_mean(loss_cle_relative)

        # Cross-entropy between probabilty maps (need to change label)
        if 'ce' in o.losses or 'ce_balanced' in o.losses:
            hmap_gt_valid   = tf.boolean_mask(hmap_gt[o.perspective], example['y_is_valid'])
            hmap_pred_valid = tf.boolean_mask(outputs['hmap'][o.perspective], example['y_is_valid'])
            # hmap is [valid_images, height, width, 2]
            count = tf.reduce_sum(hmap_gt_valid, axis=(1, 2), keep_dims=True)
            class_weight = 0.5 / tf.cast(count+1, tf.float32)
            weight = tf.reduce_sum(hmap_gt_valid * class_weight, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                    labels=hmap_gt_valid,
                    logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            if 'ce' in o.losses:
                losses['ce'] = tf.reduce_mean(loss_ce)
            if 'ce_balanced' in o.losses:
                losses['ce_balanced'] = tf.reduce_mean(
                        tf.reduce_sum(weight * loss_ce, axis=(1, 2)))

        # TODO: Make it neat if things work well.
        if outputs['hmap_A0'] is not None:
            hmap_gt_valid   = tf.boolean_mask(hmap_gt[o.perspective], example['y_is_valid'])
            hmap_pred_valid = tf.boolean_mask(outputs['hmap_A0'], example['y_is_valid'])
            # hmap is [valid_images, height, width, 2]
            count = tf.reduce_sum(hmap_gt_valid, axis=(1, 2), keep_dims=True)
            class_weight = 0.5 / tf.cast(count+1, tf.float32)
            weight = tf.reduce_sum(hmap_gt_valid * class_weight, axis=-1)
            # Flatten to feed into softmax_cross_entropy_with_logits.
            hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
            hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
            loss_ce = tf.nn.softmax_cross_entropy_with_logits(
                    labels=hmap_gt_valid,
                    logits=hmap_pred_valid)
            loss_ce = unmerge(loss_ce, 0)
            losses['ce_A0'] = tf.reduce_mean(loss_ce)

        for key in outputs['hmap_interm']:
            if outputs['hmap_interm'][key] is not None:
                hmap_gt_valid   = tf.boolean_mask(hmap_interm_gt, example['y_is_valid'])
                hmap_pred_valid = tf.boolean_mask(outputs['hmap_interm'][key], example['y_is_valid'])
                # Flatten to feed into softmax_cross_entropy_with_logits.
                hmap_gt_valid, unmerge = merge_dims(hmap_gt_valid, 0, 3)
                hmap_pred_valid, _ = merge_dims(hmap_pred_valid, 0, 3)
                loss_ce_interm = tf.nn.softmax_cross_entropy_with_logits(
                        labels=hmap_gt_valid,
                        logits=hmap_pred_valid)
                loss_ce_interm = unmerge(loss_ce_interm, 0)
                losses['ce_{}'.format(key)] = tf.reduce_mean(loss_ce_interm)

        if 'sc' in o.losses:
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
            losses['sc'] = tf.reduce_mean(loss_sc)

        # Reconstruction loss using generalized Charbonnier penalty
        if 'recon' in o.losses:
            alpha = 0.25
            s_prev_valid  = tf.boolean_mask(outputs['s_prev'],  example['y_is_valid'])
            s_recon_valid = tf.boolean_mask(outputs['s_recon'], example['y_is_valid'])
            charbonnier_penalty = tf.pow(tf.square(s_prev_valid - s_recon_valid) + 1e-10, alpha)
            losses['recon'] = tf.reduce_mean(charbonnier_penalty)

        with tf.name_scope('summary'):
            for name, loss in losses.iteritems():
                tf.summary.scalar(name, loss, collections=summaries_collections)

        #gt['y']    = {'ic': y_gt['ic'],    'oc': y_gt['oc']}
        gt['hmap'] = {'ic': hmap_gt['ic'], 'oc': hmap_gt['oc']} # for visualization in summary.
        return tf.reduce_sum(losses.values(), name=scope), gt
