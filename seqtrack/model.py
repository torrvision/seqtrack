'''This file describes several different models.

A model is a class with the properties::

    model.outputs      # Dictionary of tensors
    model.state        # Dictionary of 2-tuples of tensors
    model.batch_size   # Batch size of instance.
    model.sequence_len # Size of instantiated RNN.

The model constructor should take a dictionary of tensors::

    'x'  # Tensor of images [b, t, h, w, c]
    'x0' # Tensor of initial images [b, h, w, c]
    'y0' # Tensor of initial rectangles [b, 4]

It may also have 'target' if required.
Images input to the model are already normalized (e.g. have dataset mean subtracted).

The `outputs` dictionary should have a key 'y' and may have other keys such as 'hmap'.
'''

import pdb
import functools
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import alexnet, inception, vgg
from transformer.spatial_transformer import transformer
import math
import numpy as np
import os

import cnnutil
import geom
from helpers import merge_dims, diag_conv
from upsample import upsample

concat = tf.concat if hasattr(tf, 'concat') else tf.concat_v2

def convert_rec_to_heatmap(rec, o, min_size=None, mode='box', Gaussian=False, radius_pos=0.1, sigma=0.3):
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
        masks = get_masks_from_rectangles(rec, o, kind='bg', min_size=min_size,
            mode=mode, Gaussian=Gaussian, radius_pos=radius_pos, sigma=sigma)
        return unmerge(masks, 0)

def get_masks_from_rectangles(rec, o, output_size=None, kind='fg', typecast=True, min_size=None, mode='box', Gaussian=False, radius_pos=0.2, sigma=0.3, name='mask'):
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
        if output_size is None:
            output_size = (o.frmsz, o.frmsz)
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
        grid_x = tf.expand_dims(tf.cast(tf.range(size_x), o.dtype), 0)
        grid_y = tf.expand_dims(tf.cast(tf.range(size_y), o.dtype), 1)
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
            else: # TODO: Need debug this properly.
                x_sigma = width * sigma # TODO: can be better..
                y_sigma = height * sigma
                masks = tf.exp(-( tf.square(grid_x - x_center) / (2 * tf.square(x_sigma)) +
                                  tf.square(grid_y - y_center) / (2 * tf.square(y_sigma)) ))
        elif mode == 'center':
            obj_diam = 0.5 * (width + height)
            r = tf.sqrt(tf.square(grid_x - x_center) + tf.square(grid_y - y_center)) / obj_diam
            if not Gaussian:
                masks = tf.less_equal(r, radius_pos)
            else:
                masks = tf.exp(-0.5 * tf.square(r) / tf.square(sigma))

        if kind == 'fg': # only foreground mask
            # JV: Make this more general.
            # masks = tf.expand_dims(masks, 3) # to have channel dim
            masks = tf.expand_dims(masks, -1) # to have channel dim
        elif kind == 'bg': # add background mask
            if not Gaussian:
                masks_bg = tf.logical_not(masks)
            else:
                masks_bg = 1.0 - masks
            # JV: Make this more general.
            # masks = concat(
            #         (tf.expand_dims(masks,3), tf.expand_dims(masks_bg,3)), 3)
            masks = tf.stack([masks, masks_bg], -1)
        if typecast: # type cast so that it can be concatenated with x
            masks = tf.cast(masks, o.dtype)
        return masks

def enforce_min_size(x1, y1, x2, y2, min_size, name='min_size'):
    with tf.name_scope(name) as scope:
        # Ensure that x2-x1 > 1
        xc, xs = 0.5*(x1 + x2), x2-x1
        yc, ys = 0.5*(y1 + y2), y2-y1
        # TODO: Does this propagate NaNs?
        xs = tf.maximum(min_size, xs)
        ys = tf.maximum(min_size, ys)
        x1, x2 = xc-xs/2, xc+xs/2
        y1, y2 = yc-ys/2, yc+ys/2
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
            translate_x = tf.maximum(tf.maximum(y[:,0], y[:,2]) - 1, 0) + \
                          tf.minimum(tf.minimum(y[:,0], y[:,2]) - 0, 0)
            translate_y = tf.maximum(tf.maximum(y[:,1], y[:,3]) - 1, 0) + \
                          tf.minimum(tf.minimum(y[:,1], y[:,3]) - 0, 0)
            y = y - tf.stack([translate_x, translate_y]*2, -1)
            y = tf.reshape(y, dims)
        y = tf.clip_by_value(y, 0.0, 1.0)
    return y

def pass_depth_wise_norm(feature):
    num_channels = feature.shape.as_list()[-1]
    feature_new = []
    for c in range(num_channels):
        mean, var = tf.nn.moments(feature[:,:,:,c], axes=[1,2], keep_dims=True)
        feature_new.append((feature[:,:,:,c] - mean) / (tf.sqrt(var)+1e-5))
    return tf.stack(feature_new, 3)

def process_image_with_box(img, box, o, crop_size, scale, aspect=None):
    ''' Crop image using box and scale.

    crop_size: output size after crop-and-resize.
    scale:     uniform scalar for box.
    '''
    if aspect is not None:
        stretch = tf.stack([tf.pow(aspect, 0.5), tf.pow(aspect, -0.5)], axis=-1)
        box = geom.rect_mul(box, stretch)
    box = modify_aspect_ratio(box, o.aspect_method)
    if aspect is not None:
        box = geom.rect_mul(box, 1./stretch)

    box = scale_rectangle_size(scale, box)
    box_val = geom.rect_intersect(box,geom.unit_rect())

    batch_len = tf.shape(img)[0]
    crop = tf.image.crop_and_resize(img, geom.rect_to_tf_box(box),
                                    box_ind=tf.range(batch_len),
                                    crop_size=[crop_size]*2,
                                    extrapolation_value=128)
    return crop, box, box_val

def modify_aspect_ratio(rect, method='stretch'):
    EPSILON = 1e-3
    if method == 'stretch':
        return rect # No change.
    min_pt, max_pt = geom.rect_min_max(rect)
    center, size = 0.5*(min_pt+max_pt), max_pt-min_pt
    with tf.control_dependencies([tf.assert_greater_equal(size, 0.0)]):
        size = tf.identity(size)
    if method == 'perimeter':
        # Average of dimensions.
        width = tf.reduce_mean(size, axis=-1, keep_dims=True)
        return geom.make_rect(center - 0.5*width, center + 0.5*width)
    if method == 'area':
        # Geometric average of dimensions.
        width = tf.exp(tf.reduce_mean(tf.log(tf.maximum(size, EPSILON)),
                                      axis=-1,
                                      keep_dims=True))
        return geom.make_rect(center - 0.5*width, center + 0.5*width)
    raise ValueError('unknown method: {}'.format(method))

def scale_rectangle_size(alpha, rect):
    min_pt, max_pt = geom.rect_min_max(rect)
    center, size = 0.5*(min_pt+max_pt), max_pt-min_pt
    size *= alpha
    return geom.make_rect(center-0.5*size, center+0.5*size)

def find_center_in_scoremap(scoremap, o):
    assert len(scoremap.shape.as_list()) == 4

    max_val = tf.reduce_max(scoremap, axis=(1,2), keep_dims=True)
    with tf.control_dependencies([tf.assert_greater_equal(scoremap, 0.0)]):
        max_loc = tf.greater_equal(scoremap, max_val*0.95) # values over 95% of max.

    spatial_dim = scoremap.shape.as_list()[1:3]
    assert all(spatial_dim) # Spatial dimension must be static.
    # Compute center of each pixel in [0, 1] in search area.
    dim_y, dim_x = spatial_dim[0], spatial_dim[1]
    centers_x, centers_y = tf.meshgrid(
        tf.to_float(tf.range(dim_x)) / tf.to_float(dim_x - 1),
        tf.to_float(tf.range(dim_y)) / tf.to_float(dim_y - 1))
    centers = tf.stack([centers_x, centers_y], axis=-1)
    max_loc = tf.to_float(max_loc)
    center = tf.divide(
        tf.reduce_sum(centers * max_loc, axis=(1, 2)),
        tf.reduce_sum(max_loc, axis=(1, 2)))

    EPSILON = 1e-4
    with tf.control_dependencies([tf.assert_greater_equal(center, -EPSILON),
                                  tf.assert_less_equal(center, 1.0+EPSILON)]):
        center = tf.identity(center)
    return center

def to_image_centric_coordinate(coord, box_s_raw, o):
    ''' Convert object-centric coordinates to image-centric coordinates.
    Assume that `coord` is either center [x_center,y_center] or box [x1,y1,x2,y2].
    '''
    # scale the size in image-centric and move.
    w_raw = box_s_raw[:,2] - box_s_raw[:,0]
    h_raw = box_s_raw[:,3] - box_s_raw[:,1]
    if coord.shape.as_list()[-1] == 2: # center
        x_center = coord[:,0] * w_raw + box_s_raw[:,0]
        y_center = coord[:,1] * h_raw + box_s_raw[:,1]
        return tf.stack([x_center, y_center], 1)
    elif coord.shape.as_list()[-1] == 4: # box
        x1 = coord[:,0] * w_raw + box_s_raw[:,0]
        y1 = coord[:,1] * h_raw + box_s_raw[:,1]
        x2 = coord[:,2] * w_raw + box_s_raw[:,0]
        y2 = coord[:,3] * h_raw + box_s_raw[:,1]
        return tf.stack([x1, y1, x2, y2], 1)
    else:
        raise ValueError('coord is not expected form.')

def to_object_centric_coordinate(y, box_s_raw, box_s_val, o):
    ''' Convert image-centric coordinates to object-centric coordinates.
    Assume that `coord` is either center [x_center,y_center] or box [x1,y1,x2,y2].
    '''
    #NOTE: currently only assuming input as box.
    coord_axis = len(y.shape.as_list())-1
    x1, y1, x2, y2 = tf.unstack(y, axis=coord_axis)
    x1_raw, y1_raw, x2_raw, y2_raw = tf.unstack(box_s_raw, axis=coord_axis)
    x1_val, y1_val, x2_val, y2_val = tf.unstack(box_s_val, axis=coord_axis)
    s_raw_w = x2_raw - x1_raw # [0,1] range
    s_raw_h = y2_raw - y1_raw # [0,1] range
    # NOTE: Remember that due to the limitation of `enforce_inside_box`,
    # `process_search_with_box` can yield box_s with size of 0.
    #with tf.control_dependencies([tf.assert_greater(s_raw_w, 0.0), tf.assert_greater(s_raw_h, 0.0)]):
    x1_oc = (x1 - x1_raw) / (s_raw_w + 1e-5)
    y1_oc = (y1 - y1_raw) / (s_raw_h + 1e-5)
    x2_oc = (x2 - x1_raw) / (s_raw_w + 1e-5)
    y2_oc = (y2 - y1_raw) / (s_raw_h + 1e-5)
    y_oc = tf.stack([x1_oc, y1_oc, x2_oc, y2_oc], coord_axis)
    return y_oc

def to_image_centric_hmap(hmap_pred_oc, box_s_raw, box_s_val, o):
    ''' Convert object-centric hmap to image-centric hmap.
    Input hmap is assumed to be softmax-ed (i.e., range [0,1]) and foreground only.
    '''
    inv_box_s = geom.crop_inverse(box_s_raw)
    batch_len = tf.shape(hmap_pred_oc)[0]
    # TODO: Set extrapolation_value.
    hmap_pred_ic = tf.image.crop_and_resize(hmap_pred_oc, geom.rect_to_tf_box(inv_box_s),
        box_ind=tf.range(batch_len),
        crop_size=[o.frmsz]*2,
        extrapolation_value=None)
    return hmap_pred_ic

def get_act(act):
    if act == 'relu':
        return tf.nn.relu
    elif act =='tanh':
        return tf.nn.tanh
    elif act == 'linear':
        return None
    else:
        assert False, 'wrong activation type'

def regularize_scale(y_prev, y_curr, y0=None, local_bound=0.01, global_bound=0.1):
    ''' This function regularize (only) scale of new box.
    It is used when `boxreg` option is enabled.
    '''
    w_prev = y_prev[:,2] - y_prev[:,0]
    h_prev = y_prev[:,3] - y_prev[:,1]
    c_prev = tf.stack([(y_prev[:,2] + y_prev[:,0]), (y_prev[:,3] + y_prev[:,1])], 1) / 2.0
    w_curr = y_curr[:,2] - y_curr[:,0]
    h_curr = y_curr[:,3] - y_curr[:,1]
    c_curr = tf.stack([(y_curr[:,2] + y_curr[:,0]), (y_curr[:,3] + y_curr[:,1])], 1) / 2.0

    # add local bound
    w_reg = tf.clip_by_value(w_curr, w_prev*(1-local_bound), w_prev*(1+local_bound))
    h_reg = tf.clip_by_value(h_curr, h_prev*(1-local_bound), h_prev*(1+local_bound))

    # add global bound w.r.t. y0
    if y0 is not None:
        w0 = y0[:,2] - y0[:,0]
        h0 = y0[:,3] - y0[:,1]
        w_reg = tf.clip_by_value(w_reg, w0*(1-global_bound), w0*(1+global_bound))
        h_reg = tf.clip_by_value(h_reg, h0*(1-global_bound), h0*(1+global_bound))

    y_reg = tf.stack([c_curr[:,0]-w_reg/2.0, c_curr[:,1]-h_reg/2.0,
                      c_curr[:,0]+w_reg/2.0, c_curr[:,1]+h_reg/2.0], 1)
    return y_reg

def spatial_transform_by_flow(img2, flow):
    ''' Perform spatial transform where transformation is flow.
    The output image is a reconstructed image I1.
    '''
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        #x = tf.cast(x, 'float32')
        #y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        #x = (x + 1.0)*(width_f) / 2.0
        #y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output

    def _transform_grid(imgsz, flow):
        # grid
        x_range = tf.cast(tf.range(imgsz[2]), tf.float32)
        y_range = tf.cast(tf.range(imgsz[1]), tf.float32)
        grid_x, grid_y = tf.meshgrid(x_range, y_range)
        grid = tf.stack([tf.reshape(grid_x, [-1]), tf.reshape(grid_y, [-1])], 1)
        grid = tf.stack([grid]*imgsz[0])
        # grid transform by flow
        grid_new = grid + tf.reshape(flow, [imgsz[0], -1, flow.shape.as_list()[-1]])
        #grid_new = tf.cast(tf.round(grid_new), tf.int32)
        #grid_new = tf.clip_by_value(grid_new, 0, imgsz[1]-1)
        x_s_flat = tf.reshape(grid_new[:,:,0], [-1])
        y_s_flat = tf.reshape(grid_new[:,:,1], [-1])
        return x_s_flat, y_s_flat

    imgsz = img2.shape.as_list()
    x_s_flat, y_s_flat = _transform_grid(imgsz, flow)

    # interpolate
    input_transformed = _interpolate(img2, x_s_flat, y_s_flat, imgsz[1:3])
    img1_recon = tf.reshape(input_transformed, imgsz)
    return img1_recon

def get_rectangles_from_hmap(hmap_oc_fg, box_s_raw, box_s_val, o, y_ref):
    center_oc = find_center_in_scoremap(hmap_oc_fg, o)
    center = to_image_centric_coordinate(center_oc, box_s_raw, o)
    y_ref_size = tf.stack([y_ref[:,2]-y_ref[:,0], y_ref[:,3]-y_ref[:,1]], 1)
    y_tmp = tf.concat([center-y_ref_size*0.5, center+y_ref_size*0.5], 1)
    y_tmp_oc = to_object_centric_coordinate(y_tmp, box_s_raw, box_s_val, o)
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

def pass_rnn(x, state, cell, o, skip=False):
    ''' Convolutional RNN.
    Currently, `dense` skip type is supported; All hidden states are summed.
    '''
    # TODO:
    # 1. (LSTM) Initialize forget bias 1.
    # 2. (LSTM) Try LSTM architecture as defined in "An Empirical Exploration of-".
    # 2. Try channel-wise convolution.

    # skip state indicating which state will be connected to.
    if skip:
        skip_state = range(state['h'].shape.as_list()[1]) # All states (dense). # TODO: sparse skip. stride?
    else:
        skip_state = [-1] # only previous state.

    if cell == 'lstm':
        h_prev, c_prev = state['h'], state['c']
        with slim.arg_scope([slim.conv2d],
                num_outputs=h_prev.shape.as_list()[-1],
                kernel_size=3,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(o.wd)):
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
            it = tf.nn.sigmoid(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='i'))
            ft = tf.nn.sigmoid(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='f'))
            ct_tilda = tf.nn.tanh(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='c'))
            ct = (tf.reduce_sum(tf.expand_dims(ft, 1) * c_prev, 1)) + (it * ct_tilda)
            ot = tf.nn.sigmoid(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='o'))
            ht = ot * tf.nn.tanh(ct)
        output = ht
        state['h'] = tf.concat([state['h'][:,1:], tf.expand_dims(ht,1)], 1)
        state['c'] = tf.concat([state['c'][:,1:], tf.expand_dims(ct,1)], 1)
    elif cell == 'gru':
        h_prev = state['h']
        with slim.arg_scope([slim.conv2d],
                num_outputs=h_prev.shape.as_list()[-1],
                kernel_size=3,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(o.wd)):
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
            rt = tf.nn.sigmoid(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='r'))
            zt = tf.nn.sigmoid(slim.conv2d(tf.concat([x]+[h_prev[:,s] for s in skip_state],-1), scope='z'))
            h_tilda = tf.nn.tanh(slim.conv2d(tf.concat([x]+[rt * h_prev[:,s] for s in skip_state],-1), scope='h'))
            ht = tf.reduce_sum(tf.expand_dims(zt, 1) * h_prev, 1) + (1-zt) * h_tilda
        output = ht
        state['h'] = tf.concat([state['h'][:,1:], tf.expand_dims(ht, 1)], 1)
    elif cell == 'gau':
        h_prev = state['h']
        with slim.arg_scope([slim.conv2d],
                num_outputs=h_prev.shape.as_list()[-1],
                kernel_size=3,
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(o.wd)):
            xh = tf.concat([x]+[h_prev[:,s] for s in skip_state],-1)
            ht = tf.nn.tanh(slim.conv2d(xh, scope='f')) * tf.nn.sigmoid(slim.conv2d(xh, scope='g'))
            ht = slim.conv2d(ht, kernel_size=1, scope='1x1')
            ht = ht + x # final residual connection.
        output = ht
        state['h'] = tf.concat([state['h'][:,1:], tf.expand_dims(ht, 1)], 1)
    else:
        assert False, 'Not available cell type.'
    return output, state


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
                y_curr_pred = scale_rectangle_size(tf.expand_dims(scale, -1), y_curr_pred)
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


class RNN_dual_mix(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 crop_target=False,
                 mask_search=False,
                 lstm1_nlayers=1,
                 lstm2_nlayers=1,
                 layer_norm=False,
                 residual_lstm=False,
                 feed_examplar=False,
                 dropout_rnn=False,
                 keep_prob=0.2, # following `Recurrent Neural Network Regularization, Zaremba et al.
                 ):
        # model parameters
        self.crop_target   = crop_target
        self.mask_search   = mask_search
        self.lstm1_nlayers = lstm1_nlayers
        self.lstm2_nlayers = lstm2_nlayers
        self.layer_norm    = layer_norm
        self.residual_lstm = residual_lstm
        self.feed_examplar = feed_examplar
        self.dropout_rnn   = dropout_rnn
        self.keep_prob     = keep_prob
        # Ignore summaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.dbg = self._load_model(inputs, o)
        self.image_size   = (o.frmsz, o.frmsz)
        self.sequence_len = o.ntimesteps
        self.batch_size   = o.batchsz

    def _load_model(self, inputs, o):

        def pass_cnn(x, fully_connected):
            out = []
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
                if fully_connected:
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 1024, scope='fc1'); out.append(x)
                    x = slim.fully_connected(x, 1024, scope='fc2'); out.append(x)
            return out

        #def pass_lstm1(x, h_prev, c_prev):
        #    with slim.arg_scope([slim.fully_connected],
        #            num_outputs=o.nunits,
        #            activation_fn=None,
        #            weights_regularizer=slim.l2_regularizer(o.wd)):
        #        # NOTE: `An Empirical Exploration of Recurrent Neural Network Architecture`.
        #        # Initialize forget bias to be 1.
        #        # They also use `tanh` instead of `sigmoid` for input gate. (yet not employed here)
        #        ft = slim.fully_connected(concat((h_prev, x), 1), biases_initializer=tf.ones_initializer(), scope='hf')
        #        it = slim.fully_connected(concat((h_prev, x), 1), scope='hi')
        #        ct_tilda = slim.fully_connected(concat((h_prev, x), 1), scope='hc')
        #        ot = slim.fully_connected(concat((h_prev, x), 1), scope='ho')
        #        ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
        #        ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
        #    return ht, ct

        def pass_lstm1(x, h_prev, c_prev):
            '''
            `forget` bias is initialized to be 1 as in
            `An Empirical Exploration of Recurrent Neural Network Architecture`.
            (with zeros initialization for all gates, training fails!!!)
            As moving to layer normalization, I compute linear functions of
            input and hidden separately (all at once for 4 gates as before).
            '''
            def ln(inputs, epsilon = 1e-5, scope = None):
                mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
                with tf.variable_scope(scope + 'LN'):
                    scale = tf.get_variable('alpha', shape=[inputs.get_shape()[1]],
                                            initializer=tf.constant_initializer(1))
                    shift = tf.get_variable('beta', shape=[inputs.get_shape()[1]],
                                            initializer=tf.constant_initializer(0))
                LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift
                return LN

            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                biases_initializer=None,
                                weights_regularizer=slim.l2_regularizer(o.wd)):
                x_linear = slim.fully_connected(x, 4*o.nunits, scope='x_linear')
                h_linear = slim.fully_connected(h_prev, 4*o.nunits, scope='h_linear')

            if self.layer_norm:
                x_linear = ln(x_linear, scope='x/')
                h_linear = ln(h_linear, scope='h/')

            ft, it, ot, ct_tilda = tf.split(x_linear + h_linear, 4, axis=1)

            with tf.variable_scope('bias'):
                bf = tf.get_variable('bf', shape=[o.nunits], initializer=tf.ones_initializer())
                bi = tf.get_variable('bi', shape=[o.nunits], initializer=tf.zeros_initializer())
                bo = tf.get_variable('bo', shape=[o.nunits], initializer=tf.zeros_initializer())
                bc = tf.get_variable('bc', shape=[o.nunits], initializer=tf.zeros_initializer())

            ft = ft + bf
            it = it + bi
            ot = ot + bo
            ct_tilda = ct_tilda + bc

            ct = (tf.nn.sigmoid(ft) * c_prev) + (tf.nn.sigmoid(it) * tf.nn.tanh(ct_tilda))
            ht = tf.nn.sigmoid(ot) * tf.nn.tanh(ct)
            return ht, ct

        def pass_multi_level_cross_correlation(search, filt):
            ''' Multi-level cross-correlation function producing scoremaps.
            Option 1: depth-wise convolution
            Option 2: similarity score (-> doesn't work well)
            Note that depth-wise convolution with 1x1 filter is actually same as
            channel-wise (and element-wise) multiplication.
            '''
            # TODO: sigmoid or softmax over scoremap?
            # channel-wise l2 normalization as in Universal Correspondence Network?
            scoremap = []
            with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                for i in range(len(search)):
                    depth = search[i].shape.as_list()[-1]
                    scoremap.append(search[i] *
                            tf.expand_dims(tf.expand_dims(slim.fully_connected(filt, depth), 1), 1))
            return scoremap

        def pass_multi_level_deconvolution(x):
            ''' Multi-level deconvolutions.
            This is in a way similar to HourglassNet.
            Using sum.
            '''
            deconv = x[-1]
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

        def pass_lstm2(x, h_prev, c_prev):
            ''' ConvLSTM
            h and c have the same spatial dimension as x.
            '''
            # TODO: increase size of hidden
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

        def pass_out_rectangle(x):
            ''' Regress output rectangle.
            '''
            with slim.arg_scope([slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.flatten(x)
                x = slim.fully_connected(x, 1024, scope='fc1')
                x = slim.fully_connected(x, 1024, scope='fc2')
                x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        def pass_out_heatmap(x):
            ''' Upsample and generate spatial heatmap.
            '''
            with slim.arg_scope([slim.conv2d],
                    #num_outputs=x.shape.as_list()[-1],
                    num_outputs=2,
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                x = slim.conv2d(tf.image.resize_images(x, [241, 241]), kernel_size=[3, 3], scope='deconv')
                x = slim.conv2d(x, kernel_size=[1, 1], scope='conv1')
                x = slim.conv2d(x, kernel_size=[1, 1], activation_fn=None, scope='conv2')
            return x


        x           = inputs['x']  # shape [b, ntimesteps, h, w, 3]
        x0          = inputs['x0'] # shape [b, h, w, 3]
        y0          = inputs['y0'] # shape [b, 4]
        y           = inputs['y']  # shape [b, ntimesteps, 4]
        use_gt      = inputs['use_gt']
        gt_ratio    = inputs['gt_ratio']
        is_training = inputs['is_training']

        if self.feed_examplar:
            examplar = pass_cnn(concat([x0, get_masks_from_rectangles(y0, o)], axis=3), True)[-1]

        h1_init = [None] * self.lstm1_nlayers
        c1_init = [None] * self.lstm1_nlayers
        h2_init = [None] * self.lstm2_nlayers
        c2_init = [None] * self.lstm2_nlayers
        with tf.variable_scope('lstm_init'):
            with slim.arg_scope([slim.model_variable],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    regularizer=slim.l2_regularizer(o.wd)):
                for i in range(self.lstm1_nlayers):
                    h1_init_single = slim.model_variable('h1_{}'.format(i+1), shape=[o.nunits])
                    c1_init_single = slim.model_variable('c1_{}'.format(i+1), shape=[o.nunits])
                    h1_init[i] = tf.stack([h1_init_single] * o.batchsz)
                    c1_init[i] = tf.stack([c1_init_single] * o.batchsz)
                for i in range(self.lstm2_nlayers):
                    h2_init_single = slim.model_variable('h2_{}'.format(i+1), shape=[81, 81, 2])
                    c2_init_single = slim.model_variable('c2_{}'.format(i+1), shape=[81, 81, 2])
                    h2_init[i] = tf.stack([h2_init_single] * o.batchsz)
                    c2_init[i] = tf.stack([c2_init_single] * o.batchsz)


        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0) # for `delta` regression type output.
        hmap_init = tf.identity(get_masks_from_rectangles(y0, o))

        x_prev = x_init
        hmap_prev = hmap_init
        h1_prev, c1_prev = h1_init, c1_init
        h2_prev, c2_prev = h2_init, c2_init

        y_pred = []
        hmap_pred = []

        cnn1in, cnn1area = [], []
        cnn2in, cnn2area = [], []

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr = y[:, t]

            with tf.variable_scope('cnn1', reuse=(t > 0)):
                if self.mask_search:
                    x_search, area = process_image_with_hmap(x_curr, hmap_prev, o, mode='mask')
                cnn1out = pass_cnn(x_search, False)
                cnn1in.append(x_search)
                cnn1area.append(area)

            with tf.variable_scope('cnn2', reuse=(t > 0)):
                #xy = tf.stop_gradient(concat([x_prev, hmap_prev], axis=3))
                if not self.crop_target:
                    xy = concat([x_prev, hmap_prev], axis=3)
                else:
                    xy, area = process_image_with_hmap(x_prev, hmap_prev, o, mode='crop')
                cnn2out = pass_cnn(xy, True)
                cnn2in.append(xy)
                cnn2area.append(area)

            h1_curr = [None] * self.lstm1_nlayers
            c1_curr = [None] * self.lstm1_nlayers
            with tf.variable_scope('lstm1', reuse=(t > 0)):
                xin = tf.identity(cnn2out[-1])
                for i in range(self.lstm1_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h1_curr[i], c1_curr[i] = pass_lstm1(xin, h1_prev[i], c1_prev[i])
                        if self.residual_lstm:
                            xin = h1_curr[i] + slim.fully_connected(xin, o.nunits, scope='proj')
                        else:
                            xin = h1_curr[i]
                    if self.dropout_rnn:
                        xin = slim.dropout(xin, keep_prob=self.keep_prob,
                                           is_training=is_training, scope='dropout')

            if self.feed_examplar:
                with tf.variable_scope('combine_examplar', reuse=(t > 0)):
                    nch_xin = xin.shape.as_list()[-1]
                    nch_examplar = examplar.shape.as_list()[-1]
                    if nch_xin != nch_examplar:
                        xin = xin + slim.fully_connected(examplar, nch_xin, scope='proj')
                    else:
                        xin = xin + examplar

            with tf.variable_scope('multi_level_cross_correlation', reuse=(t > 0)):
                #scoremap = pass_multi_level_cross_correlation(cnn1out, h1_curr[-1]) # multi-layer lstm1
                scoremap = pass_multi_level_cross_correlation(cnn1out, xin)

            with tf.variable_scope('multi_level_deconvolution', reuse=(t > 0)):
                scoremap = pass_multi_level_deconvolution(scoremap)

            with tf.variable_scope('cnn_out_hmap', reuse=(t > 0)):
                hmap_curr_pred = pass_out_heatmap(scoremap)

            h2_curr = [None] * self.lstm2_nlayers
            c2_curr = [None] * self.lstm2_nlayers
            with tf.variable_scope('lstm2', reuse=(t > 0)):
                xin = tf.identity(scoremap)
                for i in range(self.lstm2_nlayers):
                    with tf.variable_scope('layer_{}'.format(i+1), reuse=(t > 0)):
                        h2_curr[i], c2_curr[i] = pass_lstm2(xin, h2_prev[i], c2_prev[i])
                        if self.residual_lstm:
                            xin = h2_curr[i] + slim.conv2d(xin, 2, 1, scope='proj')
                        else:
                            xin = h2_curr[i]
                    if self.dropout_rnn:
                        xin = slim.dropout(h2_curr[i], keep_prob=self.keep_prob,
                                           is_training=is_training, scope='dropout')

            with tf.variable_scope('cnn_out_rec', reuse=(t > 0)):
                #y_curr_pred = pass_out_rectangle(h2_curr[-1]) # multi-layer lstm2
                y_curr_pred = pass_out_rectangle(xin)

            rand_prob = tf.random_uniform([], minval=0, maxval=1)
            gt_condition = tf.logical_and(use_gt, tf.less_equal(rand_prob, gt_ratio))
            hmap_curr_gt = tf.identity(get_masks_from_rectangles(y_curr, o))
            hmap_prev = tf.cond(gt_condition, lambda: hmap_curr_gt,
                                              lambda: tf.expand_dims(tf.nn.softmax(hmap_curr_pred)[:,:,:,0], 3))

            x_prev = x_curr
            h1_prev, c1_prev = h1_curr, c1_curr
            h2_prev, c2_prev = h2_curr, c2_curr

            y_pred.append(y_curr_pred)
            hmap_pred.append(hmap_curr_pred)

        y_pred = tf.stack(y_pred, axis=1) # list to tensor
        hmap_pred = tf.stack(hmap_pred, axis=1)
        y_prev = y_pred[:,-1,:] # for `delta` regression type output.

        outputs = {'y': y_pred, 'hmap': hmap_pred, 'hmap_softmax': tf.nn.softmax(hmap_pred)}
        state = {}
        state.update({'h1_{}'.format(i+1): (h1_init[i], h1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'c1_{}'.format(i+1): (c1_init[i], c1_curr[i]) for i in range(self.lstm1_nlayers)})
        state.update({'h2_{}'.format(i+1): (h2_init[i], h2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'c2_{}'.format(i+1): (c2_init[i], c2_curr[i]) for i in range(self.lstm2_nlayers)})
        state.update({'x': (x_init, x_prev)})
        state.update({'hmap': (hmap_init, hmap_prev)})
        state.update({'y': (y_init, y_prev)})

        #dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        #dbg = {}
        dbg = {'cnn1in': tf.stack(cnn1in, axis=1),
               'cnn2in': tf.stack(cnn2in, axis=1),
               'cnn1area': tf.stack(cnn1area, axis=1),
               'cnn2area': tf.stack(cnn2area, axis=1)}
        return outputs, state, dbg


def rnn_conv_asymm(example, o,
                   summaries_collections=None,
                   # Model parameters:
                   input_num_layers=3,
                   input_kernel_size=[7, 5, 3],
                   input_num_channels=[16, 32, 64],
                   input_stride=[2, 1, 1],
                   input_pool=[True, True, True],
                   input_pool_stride=[2, 2, 2],
                   input_pool_kernel_size=[3, 3, 3],
                   input_batch_norm=False,
                   lstm_num_channels=64,
                   lstm_num_layers=1):
                   # lstm_kernel_size=[3]):

    images = example['x']
    x0     = example['x0']
    y0     = example['y0']
    is_training = example['is_training']
    masks = get_masks_from_rectangles(y0, o)
    if o.debugmode:
        with tf.name_scope('input_preview'):
            tf.summary.image('x', images[0], collections=summaries_collections)
            target = concat([images[0, 0], masks[0]], axis=2)
            tf.summary.image('target', tf.expand_dims(target, axis=0),
                             collections=summaries_collections)
    if o.activ_histogram:
        with tf.name_scope('input_histogram'):
            tf.summary.histogram('x', images, collections=summaries_collections)
    init_input = concat([x0, masks], axis=3)

    assert(len(input_kernel_size)      == input_num_layers)
    assert(len(input_num_channels)     == input_num_layers)
    assert(len(input_stride)           == input_num_layers)
    assert(len(input_pool)             == input_num_layers)
    assert(len(input_pool_stride)      == input_num_layers)
    assert(len(input_pool_kernel_size) == input_num_layers)
    assert(lstm_num_layers >= 1)
    # assert(len(lstm_kernel_size) == lstm_num_layers)

    def input_cnn(x, num_outputs, name='input_cnn'):
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                normalizer_fn = None
                conv2d_params = {'weights_regularizer': slim.l2_regularizer(o.wd)}
                if input_batch_norm:
                    conv2d_params.update({
                        'normalizer_fn': slim.batch_norm,
                        'normalizer_params': {
                            'is_training': is_training,
                        }})
                with slim.arg_scope([slim.conv2d], **conv2d_params):
                    layers = {}
                    for i in range(input_num_layers):
                        conv_name = 'conv{}'.format(i+1)
                        x = slim.conv2d(x, input_num_channels[i],
                                           kernel_size=input_kernel_size[i],
                                           stride=input_stride[i],
                                           scope=conv_name)
                        layers[conv_name] = x
                        if input_pool[i]:
                            pool_name = 'pool{}'.format(i+1)
                            x = slim.max_pool2d(x, kernel_size=input_pool_kernel_size[i],
                                                   stride=input_pool_stride[i],
                                                   scope=pool_name)
                    # if o.activ_histogram:
                    #     with tf.name_scope('summary'):
                    #         for k, v in layers.iteritems():
                    #             tf.summary.histogram(k, v, collections=summaries_collections)
        return x

    def conv_lstm(x, h_prev, c_prev, state_dim, name='conv_lstm'):
        with tf.name_scope(name) as scope:
            with slim.arg_scope([slim.conv2d],
                                num_outputs=state_dim,
                                kernel_size=3,
                                padding='SAME',
                                activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(o.wd)):
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

    def output_cnn(x, name='output_cnn'):
        with tf.name_scope(name):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    weights_regularizer=slim.l2_regularizer(o.wd)):
                    layers = {}
                    x = slim.conv2d(x, 128, kernel_size=3, stride=2, scope='conv1')
                    layers['conv1'] = x
                    x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool1')
                    x = slim.conv2d(x, 256, kernel_size=3, stride=1, scope='conv2')
                    layers['conv2'] = x
                    x = slim.max_pool2d(x, kernel_size=3, stride=2, scope='pool2')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 4, scope='predict')
                    layers['predict'] = x
                    # if o.activ_histogram:
                    #     with tf.name_scope('summary'):
                    #         for k, v in layers.iteritems():
                    #             tf.summary.histogram(k, v, collections=summaries_collections)
        return x

    # At start of sequence, compute hidden state from first example.
    # Feed (h_init, c_init) to resume tracking from previous state.
    # Do NOT feed (h_init, c_init) when starting new sequence.
    # TODO: Share some layers?
    h_init = [None] * lstm_num_layers
    c_init = [None] * lstm_num_layers
    with tf.variable_scope('lstm_init'):
        for j in range(lstm_num_layers):
            with tf.variable_scope('layer_{}'.format(j+1)):
                with tf.variable_scope('h_init'):
                    h_init[j] = input_cnn(init_input, num_outputs=lstm_num_channels)
                with tf.variable_scope('c_init'):
                    c_init[j] = input_cnn(init_input, num_outputs=lstm_num_channels)

    # # TODO: Process all frames together in training (when sequences are equal length)
    # # (because it enables batch-norm to operate on whole sequence)
    # # but not during testing (when sequences are different lengths)
    # x, unmerge = merge_dims(images, 0, 2)
    # with tf.name_scope('frame_cnn') as scope:
    #     with tf.variable_scope('frame_cnn'):
    #         # Pass name scope from above, otherwise makes new name scope
    #         # within name scope created by variable scope.
    #         r = input_cnn(x, num_outputs=lstm_num_channels, name=scope)
    # r = unmerge(r, 0)

    y = []
    ht, ct = h_init, c_init
    for t in range(o.ntimesteps):
        xt = images[:, t]
        with tf.name_scope('frame_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('frame_cnn', reuse=(t > 0)):
                # Pass name scope from above, otherwise makes new name scope
                # within name scope created by variable scope.
                xt = input_cnn(xt, num_outputs=lstm_num_channels, name=scope)
        with tf.name_scope('conv_lstm_{}'.format(t)):
            with tf.variable_scope('conv_lstm', reuse=(t > 0)):
                for j in range(lstm_num_layers):
                    layer_name = 'layer_{}'.format(j+1)
                    with tf.variable_scope(layer_name, reuse=(t > 0)):
                        ht[j], ct[j] = conv_lstm(xt, ht[j], ct[j],
                                                 state_dim=lstm_num_channels)
                    xt = ht[j]
        with tf.name_scope('out_cnn_{}'.format(t)) as scope:
            with tf.variable_scope('out_cnn', reuse=(t > 0)):
                yt = output_cnn(xt, name=scope)
        y.append(yt)
        # tf.get_variable_scope().reuse_variables()
    h_last, c_last = ht, ct
    y = tf.stack(y, axis=1) # list to tensor

    # with tf.name_scope('out_cnn') as scope:
    #     with tf.variable_scope('out_cnn', reuse=(t > 0)):
    #         y = output_cnn(z, name=scope)

    with tf.name_scope('summary'):
        if o.activ_histogram:
            tf.summary.histogram('rect', y, collections=summaries_collections)
        if o.param_histogram:
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                tf.summary.histogram(v.name, v, collections=summaries_collections)

    outputs = {'y': y}
    state = {}
    state.update({'h_{}'.format(j+1): (h_init[j], h_last[j])
                  for j in range(lstm_num_layers)})
    state.update({'c_{}'.format(j+1): (c_init[j], c_last[j])
                  for j in range(lstm_num_layers)})

    class Model:
        pass
    model = Model()
    model.outputs = outputs
    model.state   = state
    # Properties of instantiated model:
    model.image_size   = (o.frmsz, o.frmsz)
    model.sequence_len = o.ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model


def rnn_multi_res(example, o,
                  summaries_collections=None,
                  # Model options:
                  kind='vgg',
                  use_heatmap=False,
                  **model_params):

    images = example['x']
    x0     = example['x0']
    y0     = example['y0']
    is_training = example['is_training']
    masks = get_masks_from_rectangles(y0, o)
    if o.debugmode:
        with tf.name_scope('input_preview'):
            tf.summary.image('x', images[0], collections=summaries_collections)
            target = concat([images[0, 0], masks[0]], axis=2)
            tf.summary.image('target', tf.expand_dims(target, axis=0),
                             collections=summaries_collections)
    if o.activ_histogram:
        with tf.name_scope('input_histogram'):
            tf.summary.histogram('x', images, collections=summaries_collections)
    init_input = concat([x0, masks], axis=3)

    # net_fn(x, None, init=True, ...) returns None, h_init.
    # net_fn(x, h_prev, init=False, ...) returns y, h.
    if kind == 'vgg':
        net_fn = multi_res_vgg
    elif kind == 'resnet':
        raise Exception('not implemented')
    else:
        raise ValueError('unknown net type: {}'.format(kind))

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            # TODO: Share some layers?
            with tf.variable_scope('rnn_init'):
                _, s_init = net_fn(init_input, None, init=True,
                    use_heatmap=use_heatmap, heatmap_stride=o.heatmap_stride,
                    **model_params)

            y, heatmap = [], []
            s_prev = s_init
            for t in range(o.ntimesteps):
                with tf.name_scope('t{}'.format(t)):
                    with tf.variable_scope('frame', reuse=(t > 0)):
                        outputs_t, s = net_fn(images[:, t], s_prev, init=False,
                            use_heatmap=use_heatmap, heatmap_stride=o.heatmap_stride,
                            **model_params)
                        s_prev = s
                        y.append(outputs_t['y'])
                        if use_heatmap:
                            heatmap.append(outputs_t['hmap'])
            y = tf.stack(y, axis=1) # list to tensor
            if use_heatmap:
                heatmap = tf.stack(heatmap, axis=1) # list to tensor

    outputs = {'y': y}
    if use_heatmap:
        outputs['hmap'] = heatmap
    assert(set(s_init.keys()) == set(s.keys()))
    state = {k: (s_init[k], s[k]) for k in s}

    class Model:
        pass
    model = Model()
    model.outputs = outputs
    model.state   = state
    # Properties of instantiated model:
    model.image_size   = (o.frmsz, o.frmsz)
    model.sequence_len = o.ntimesteps # Static length of unrolled RNN.
    model.batch_size   = None # Model accepts variable batch size.
    return model


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


def multi_res_vgg(x, prev, init, use_heatmap, heatmap_stride,
        # Model parameters:
        use_bnorm=False,
        conv_num_groups=5,
        conv_num_layers=[2, 2, 3, 3, 3],
        conv_kernel_size=[3, 3, 3, 3, 3],
        conv_stride=[1, 1, 1, 1, 1],
        conv_rnn_depth=[0, 0, 1, 0, 0],
        conv_dim_first=16,
        conv_dim_last=256,
        fc_num_layers=0,
        fc_dim=256,
        heatmap_input_stride=16,
        ):
    '''
    Layers are conv[1], ..., conv[N], fc[N+1], ..., fc[N+M].
    '''

    assert(len(conv_num_layers) == conv_num_groups)
    assert(len(conv_rnn_depth) == conv_num_groups)
    conv_use_rnn = map(lambda x: x > 0, conv_rnn_depth)

    dims = np.logspace(math.log10(conv_dim_first),
                       math.log10(conv_dim_last),
                       conv_num_groups)
    dims = np.round(dims).astype(np.int)

    with slim.arg_scope([slim.batch_norm], fused=True):
        # Dictionary of state.
        curr = {}
        # Array of (tensor before op with stride, stride) tuples.
        stride_steps = []
        if init and not any(conv_use_rnn):
            return None, curr
        bnorm_params = {'normalizer_fn': slim.batch_norm} if use_bnorm else {}
        for j in range(conv_num_groups):
            # Group of conv plus a pool.
            with slim.arg_scope([slim.conv2d], **bnorm_params):
                conv_name = lambda k: 'conv{}_{}'.format(j+1, k+1)
                for k in range(conv_num_layers[j]):
                    stride = conv_stride[j] if k == conv_num_layers[j]-1 else 1
                    if stride != 1:
                        stride_steps.append((x, stride))
                    x = slim.conv2d(x, dims[j], conv_kernel_size[j], stride=stride,
                                    scope=conv_name(k))
                stride_steps.append((x, 2))
                x = slim.max_pool2d(x, 3, padding='SAME', scope='pool{}'.format(j+1))

            # LSTM at end of group.
            if conv_use_rnn[j]:
                rnn_name = 'rnn{}'.format(j+1)
                if init:
                    # Produce initial state of RNNs.
                    # TODO: Why not variable_scope here? Only expect one instance?
                    for d in range(conv_rnn_depth[j]):
                        layer_name = 'layer{}'.format(d+1)
                        h = '{}_{}_{}'.format(rnn_name, layer_name, 'h')
                        c = '{}_{}_{}'.format(rnn_name, layer_name, 'c')
                        curr[h] = slim.conv2d(x, dims[j], 3, activation_fn=None, scope=h)
                        curr[c] = slim.conv2d(x, dims[j], 3, activation_fn=None, scope=c)
                else:
                    # Different scope for different RNNs.
                    with tf.variable_scope(rnn_name):
                        for d in range(conv_rnn_depth[j]):
                            layer_name = 'layer{}'.format(d+1)
                            with tf.variable_scope(layer_name):
                                h = '{}_{}_{}'.format(rnn_name, layer_name, 'h')
                                c = '{}_{}_{}'.format(rnn_name, layer_name, 'c')
                                curr[h], curr[c] = conv_lstm(x, prev[h], prev[c], state_dim=dims[j])
                                x = curr[h]
            if init and not any(conv_use_rnn[j+1:]):
                # Do not add layers to init network that will not be used.
                return None, curr

        y = x

        outputs = {}
        if use_heatmap:
            # Upsample and convolve to get to heatmap.
            total_stride = np.cumprod([stride for _, stride in stride_steps])
            input_ind = np.asscalar(np.flatnonzero(np.array(total_stride) == heatmap_input_stride))
            output_ind = np.asscalar(np.flatnonzero(np.array(total_stride) == heatmap_stride))
            # TODO: Handle case of using final stride step.
            heatmap, _ = stride_steps[input_ind+1]
            # Now total_stride[output_ind] = (___, heatmap_stride).
            # Our current stride is total_stride[-1].
            # Work back from current stride to restore resolution.
            # Note that we don't take the features from stride_steps[output_ind] because
            # these are the features before the desired stride.
            # TODO: Make sure this works with heatmap_stride = 1.
            for j in range(input_ind, output_ind, -1):
                # Combine current features with features before stride.
                before_stride, stride = stride_steps[j]
                heatmap = upsample(heatmap, stride)
                heatmap = concat([heatmap, before_stride], axis=3)
                num_outputs = 2 if j == output_ind+1 else before_stride.shape.as_list()[3]
                activation_fn = None if j == output_ind+1 else tf.nn.relu
                # TODO: Should the kernel_size be 1, 3, or something else?
                heatmap = slim.conv2d(heatmap, num_outputs, 1, activation_fn=activation_fn,
                                      scope='merge{}'.format(j+1))
            outputs['hmap'] = heatmap

        # Fully-connected stage to get rectangle.
        y = slim.flatten(y)
        fc_name_fn = lambda ind: 'fc{}'.format(conv_num_groups + ind + 1)
        with slim.arg_scope([slim.fully_connected], **bnorm_params):
            for j in range(fc_num_layers):
                y = slim.fully_connected(y, fc_dim, scope=fc_name_fn(j))
        # Make prediction.
        y = slim.fully_connected(y, 4, activation_fn=None, normalizer_fn=None,
                                 scope=fc_name_fn(fc_num_layers))
        outputs['y'] = y
        return outputs, curr


def load_model(o, model_params=None):
    '''
    example is a dictionary that maps strings to Tensors.
    Its keys should include 'inputs', 'labels', 'x0', 'y0'.
    '''
    model_params = model_params or {}
    assert('summaries_collections' not in model_params)
    if o.model == 'RNN_dual_mix':
        model = functools.partial(RNN_dual_mix, o=o, **model_params)
    elif o.model == 'Nornn':
        model = functools.partial(Nornn, o=o, **model_params)
    elif o.model == 'RNN_conv_asymm':
        model = functools.partial(rnn_conv_asymm, o=o, **model_params)
    elif o.model == 'RNN_multi_res':
        model = functools.partial(rnn_multi_res, o=o, **model_params)
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

    o.model = 'RNN_new' # RNN_basic, RNN_a

    # data setting (since the model requires stat, I need this to test)
    import data
    loader = data.load_data(o)

    if o.model == 'RNN_basic':
        o.pass_yinit = True
        m = RNN_basic(o)
    elif o.model == 'RNN_new':
        o.losses = ['ce', 'l2']
        m = RNN_new(o, loader.stat['train'])

    pdb.set_trace()
