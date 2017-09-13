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

def convert_rec_to_heatmap(rec, o, min_size=None, Gaussian=False):
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
        masks = get_masks_from_rectangles(rec, o, kind='bg', min_size=min_size, Gaussian=Gaussian)
        return unmerge(masks, 0)

def get_masks_from_rectangles(rec, o, kind='fg', typecast=True, min_size=None, Gaussian=False, name='mask'):
    with tf.name_scope(name) as scope:
        # create mask using rec; typically rec=y_prev
        # rec -- [b, 4]
        rec *= float(o.frmsz)
        # x1, y1, x2, y2 -- [b]
        x1, y1, x2, y2 = tf.unstack(rec, axis=1)
        if min_size is not None:
            x1, y1, x2, y2 = enforce_min_size(x1, y1, x2, y2, min_size=min_size)
        # grid_x -- [1, frmsz]
        # grid_y -- [frmsz, 1]
        grid_x = tf.expand_dims(tf.cast(tf.range(o.frmsz), o.dtype), 0)
        grid_y = tf.expand_dims(tf.cast(tf.range(o.frmsz), o.dtype), 1)
        # resize tensors so that they can be compared
        # x1, y1, x2, y2 -- [b, 1, 1]
        x1 = tf.expand_dims(tf.expand_dims(x1, -1), -1)
        x2 = tf.expand_dims(tf.expand_dims(x2, -1), -1)
        y1 = tf.expand_dims(tf.expand_dims(y1, -1), -1)
        y2 = tf.expand_dims(tf.expand_dims(y2, -1), -1)
        # masks -- [b, frmsz, frmsz]
        if not Gaussian:
            masks = tf.logical_and(
                tf.logical_and(tf.less_equal(x1, grid_x),
                               tf.less_equal(grid_x, x2)),
                tf.logical_and(tf.less_equal(y1, grid_y),
                               tf.less_equal(grid_y, y2)))
        else: # TODO: Need debug this properly.
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width, height = x2 - x1, y2 - y1
            x_sigma = width * 0.3 # TODO: can be better..
            y_sigma = height * 0.3
            masks = tf.exp(-( tf.square(grid_x - x_center) / (2 * tf.square(x_sigma)) +
                              tf.square(grid_y - y_center) / (2 * tf.square(y_sigma)) ))

        if kind == 'fg': # only foreground mask
            masks = tf.expand_dims(masks, 3) # to have channel dim
        elif kind == 'bg': # add background mask
            if not Gaussian:
                masks_bg = tf.logical_not(masks)
            else:
                masks_bg = 1.0 - masks
            masks = concat(
                    (tf.expand_dims(masks,3), tf.expand_dims(masks_bg,3)), 3)
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

def enforce_inside_box(y, name='inside_box'):
    ''' Force the box to be in range [0,1]
    '''
    assert len(y.shape.as_list()) == 2
    with tf.name_scope(name) as scope:
        # outside [0,1]
        y = tf.maximum(y, 0.0)
        y = tf.minimum(y, 1.0)
        # also enforce no flip
        #y = tf.stack([tf.minimum(y[:,0], y[:,2]), tf.minimum(y[:,1], y[:,3]),
        #              tf.maximum(y[:,2], y[:,2]), tf.maximum(y[:,1], y[:,3])], 1)
    return y

def pass_depth_wise_norm(feature):
    num_channels = feature.shape.as_list()[-1]
    feature_new = []
    for c in range(num_channels):
        mean, var = tf.nn.moments(feature[:,:,:,c], axes=[1,2], keep_dims=True)
        feature_new.append((feature[:,:,:,c] - mean) / (tf.sqrt(var)+1e-5))
    return tf.stack(feature_new, 3)

def process_target_with_box(img, box, o):
    ''' Crop target image x with box y.
    Box y coordinate is in image-centric space.
    '''
    with tf.control_dependencies(
            [tf.assert_greater_equal(box, 0.0), tf.assert_less_equal(box, 1.0)]):
        box = tf.identity(box)
    # JV: Remove dependence on explicit batch size.
    # crop = []
    # for b in range(o.batchsz):
    #     cropbox = tf.expand_dims(tf.stack([box[b,1], box[b,0], box[b,3], box[b,2]], 0), 0)
    #     crop.append(tf.image.crop_and_resize(tf.expand_dims(img[b],0),
    #                                          boxes=cropbox,
    #                                          box_ind=[0],
    #                                          crop_size=[o.frmsz/o.search_scale]*2)) # target size
    # return tf.concat(crop, 0)
    batch_len = tf.shape(img)[0]
    crop_size = (o.frmsz - 1) / o.search_scale + 1
    # Check that search_scale divides (frame_size - 1).
    assert (crop_size - 1) * o.search_scale == o.frmsz - 1
    # TODO: Set extrapolation_value.
    crop = tf.image.crop_and_resize(img, geom.rect_to_tf_box(box),
        box_ind=tf.range(batch_len),
        crop_size=[crop_size]*2,
        extrapolation_value=None)
    return crop

def process_search_with_box(img, box, o):
    ''' Crop search image x with box y.
    '''
    with tf.control_dependencies(
            [tf.assert_greater_equal(box, 0.0), tf.assert_less_equal(box, 1.0)]):
        box = tf.identity(box)

    # JV: Remove dependence on explicit batch size.
    # search = []
    # box_s_raw = []
    # box_s_val = []
    # for b in range(o.batchsz):
    #     x1 = box[b,0]
    #     y1 = box[b,1]
    #     x2 = box[b,2]
    #     y2 = box[b,3]
    #     # search size: twice bigger than object.
    #     w_margin = ((x2 - x1) * float(o.search_scale - 1)) * 0.5
    #     h_margin = ((y2 - y1) * float(o.search_scale - 1)) * 0.5
    #     # search box (raw)
    #     x1_s_raw = x1 - w_margin # can be outside of [0,1]
    #     y1_s_raw = y1 - h_margin
    #     x2_s_raw = x2 + w_margin
    #     y2_s_raw = y2 + h_margin
    #     # search box (valid)
    #     x1_s_val = tf.maximum(x1_s_raw, 0.0)
    #     y1_s_val = tf.maximum(y1_s_raw, 0.0)
    #     x2_s_val = tf.minimum(x2_s_raw, 1.0)
    #     y2_s_val = tf.minimum(y2_s_raw, 1.0)
    #     # crop (only within valid region)
    #     s_val_h = tf.cast((y2_s_val-y1_s_val)*o.frmsz, tf.int32)
    #     s_val_w = tf.cast((x2_s_val-x1_s_val)*o.frmsz, tf.int32)
    #     s_val_h = tf.maximum(s_val_h, 1) # to prevent assertion error.
    #     s_val_w = tf.maximum(s_val_w, 1) # to prevent assertion error.
    #     crop_val = tf.image.crop_to_bounding_box(
    #             image=img[b],
    #             offset_height=tf.cast(y1_s_val*(o.frmsz-1), tf.int32),
    #             offset_width =tf.cast(x1_s_val*(o.frmsz-1), tf.int32),
    #             target_height=s_val_h,
    #             target_width =s_val_w)
    #     # pad crop so that it preserves aspect ratio.
    #     s_raw_h = tf.cast((y2_s_raw-y1_s_raw)*o.frmsz, tf.int32)
    #     s_raw_w = tf.cast((x2_s_raw-x1_s_raw)*o.frmsz, tf.int32)
    #     s_raw_h = tf.maximum(s_raw_h, 1) # to prevent assertion error.
    #     s_raw_w = tf.maximum(s_raw_w, 1) # to prevent assertion error.
    #     crop_pad = tf.image.pad_to_bounding_box(
    #             image=crop_val,
    #             offset_height=tf.cast((y1_s_val-y1_s_raw)*(o.frmsz-1), tf.int32),
    #             offset_width =tf.cast((x1_s_val-x1_s_raw)*(o.frmsz-1), tf.int32),
    #             target_height=s_raw_h,
    #             target_width =s_raw_w)
    #     # resize to 241 x 241
    #     crop_res = tf.image.resize_images(crop_pad, [o.frmsz, o.frmsz])

    #     # outputs we need.
    #     search.append(crop_res)
    #     box_s_raw.append(tf.stack([x1_s_raw, y1_s_raw, x2_s_raw, y2_s_raw], 0))
    #     box_s_val.append(tf.stack([x1_s_val, y1_s_val, x2_s_val, y2_s_val], 0))

    # search = tf.stack(search, 0)
    # box_s_raw = tf.stack(box_s_raw, 0)
    # box_s_val = tf.stack(box_s_val, 0)
    # return search, box_s_raw, box_s_val

    box = scale_rectangle_size(o.search_scale, box)
    box_val = geom.rect_intersect(box, geom.unit_rect())

    batch_len = tf.shape(img)[0]
    # TODO: Set extrapolation_value.
    search = tf.image.crop_and_resize(img, geom.rect_to_tf_box(box),
        box_ind=tf.range(batch_len),
        crop_size=[o.frmsz]*2,
        extrapolation_value=None)
    return search, box, box_val

def scale_rectangle_size(alpha, rect):
    min_pt, max_pt = geom.rect_min_max(rect)
    center, size = 0.5*(min_pt+max_pt), max_pt-min_pt
    size *= alpha
    return geom.make_rect(center-0.5*size, center+0.5*size)

def find_center_in_scoremap(scoremap, o):
    # JV: Keep trailing dimension for broadcasting.
    # scoremap = tf.squeeze(scoremap, axis=-1)
    # assert(len(scoremap.shape.as_list())==3)
    assert len(scoremap.shape.as_list()) == 4

    max_val = tf.reduce_max(scoremap, axis=(1,2), keep_dims=True)
    #max_loc = tf.equal(scoremap, max_val) # only max.
    with tf.control_dependencies([tf.assert_greater_equal(scoremap, 0.0)]):
        # JV: Use greater_equal in case scoremap is all zero.
        max_loc = tf.greater_equal(scoremap, max_val*0.95) # values over 95% of max.
    # NOTE: weighted average based on distance to the center, instead of average.
    # This is to reguarlize object movement, but it's a hack and will not always work.
    # Ideally, learning motion dynamics can solve this without this.
    # Siamese paper seems to put a cosine window on scoremap!

    # JV: Remove dependence on explicit batch size.
    # dims = scoremap.shape.as_list()[1]
    # center = []
    # for b in range(o.batchsz):
    #     maxlocs = tf.cast(tf.where(max_loc[b]), tf.float32)
    #     avg_center = tf.reduce_mean(maxlocs, axis=0)
    #     center.append(tf.cond(tf.less(max_val[b,0,0], o.th_prob_stay), # TODO: test optimal value.
    #                           lambda: tf.stack([dims/2.]*2),
    #                           lambda: avg_center))
    # center = tf.stack(center, 0)
    # center = tf.stack([center[:,1], center[:,0]], 1) # To make it [x,y] format.
    # # JV: Use dims - 1 and resize with align_corners=True to preserve alignment.
    # center = center / (dims - 1) # To keep coordinate in relative scale range [0, 1].
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
    x1_oc = (x1 - x1_raw) / s_raw_w
    y1_oc = (y1 - y1_raw) / s_raw_h
    x2_oc = (x2 - x1_raw) / s_raw_w
    y2_oc = (y2 - y1_raw) / s_raw_h
    y_oc = tf.stack([x1_oc, y1_oc, x2_oc, y2_oc], coord_axis)
    return y_oc

def to_image_centric_hmap(hmap_pred_oc, box_s_raw, box_s_val, o):
    ''' Convert object-centric hmap to image-centric hmap.
    Input hmap is assumed to be softmax-ed (i.e., range [0,1]) and foreground only.
    '''
    # JV: Remove dependence on explicit batch size.
    # hmap_pred = []
    # for b in range(o.batchsz):
    #     # resize to search (raw). It can become bigger or smaller.
    #     s_raw_h = tf.cast((box_s_raw[b][3] - box_s_raw[b][1]) * o.frmsz, tf.int32)
    #     s_raw_w = tf.cast((box_s_raw[b][2] - box_s_raw[b][0]) * o.frmsz, tf.int32)
    #     s_raw_h = tf.maximum(s_raw_h, 1) # to prevent assertion error.
    #     s_raw_w = tf.maximum(s_raw_w, 1) # to prevent assertion error.
    #     s_raw = tf.image.resize_images(hmap_pred_oc[b], [s_raw_h, s_raw_w])
    #     # crop to extract only valid search area.
    #     s_val_h = tf.cast((box_s_val[b][3]-box_s_val[b][1])*o.frmsz, tf.int32)
    #     s_val_w = tf.cast((box_s_val[b][2]-box_s_val[b][0])*o.frmsz, tf.int32)
    #     s_val_h = tf.maximum(s_val_h, 1) # to prevent assertion error.
    #     s_val_w = tf.maximum(s_val_w, 1) # to prevent assertion error.
    #     s_val = tf.image.crop_to_bounding_box(
    #             image=s_raw,
    #             offset_height=tf.cast((box_s_val[b][1]-box_s_raw[b][1])*(o.frmsz-1), tf.int32),
    #             offset_width =tf.cast((box_s_val[b][0]-box_s_raw[b][0])*(o.frmsz-1), tf.int32),
    #             target_height=s_val_h,
    #             target_width =s_val_w)
    #     # pad to reconstruct the original image-centric scale.
    #     hmap = tf.image.pad_to_bounding_box( # zero padded.
    #             image=s_val,
    #             offset_height=tf.cast(box_s_val[b][1]*(o.frmsz-1), tf.int32),
    #             offset_width =tf.cast(box_s_val[b][0]*(o.frmsz-1), tf.int32),
    #             target_height=o.frmsz,
    #             target_width =o.frmsz)
    #     hmap_pred.append(hmap)
    # return tf.stack(hmap_pred, 0)

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
    #w_reg = tf.minimum(tf.maximum(w_curr, w_prev*(1-local_bound)), w_prev*(1+local_bound))
    #h_reg = tf.minimum(tf.maximum(h_curr, h_prev*(1-local_bound)), h_prev*(1+local_bound))
    w_reg = tf.clip_by_value(w_curr, w_prev*(1-local_bound), w_prev*(1+local_bound))
    h_reg = tf.clip_by_value(h_curr, h_prev*(1-local_bound), h_prev*(1+local_bound))

    # add global bound w.r.t. y0
    if y0 is not None:
        w0 = y0[:,2] - y0[:,0]
        h0 = y0[:,3] - y0[:,1]
        #w_reg = tf.minimum(tf.maximum(w_reg, w0*(1-global_bound)), w0*(1+global_bound))
        #h_reg = tf.minimum(tf.maximum(h_reg, h0*(1-global_bound)), h0*(1+global_bound))
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

class Nornn(object):
    '''
    This model has two RNNs (ConvLSTM for Dynamics and LSTM for Appearances).
    '''
    def __init__(self, inputs, o,
                 summaries_collections=None,
                 feat_act='linear', # NOTE: tanh ~ linear >>>>> relu. Do not use relu!
                 depth_wise_norm=False,
                 boxreg=False,
                 boxreg_imgpair=False,
                 boxreg_hmap=False,
                 boxreg_hmap_clean=False,
                 boxreg_mask=False,
                 boxreg_flow=False,
                 resize_target=False,
                 divide_target=False,
                 stn=False,
                 coarse_hmap=False,
                 ):
        # model parameters
        self.feat_act          = feat_act
        self.depth_wise_norm   = depth_wise_norm
        self.boxreg            = boxreg
        self.boxreg_imgpair    = boxreg_imgpair
        self.boxreg_hmap       = boxreg_hmap
        self.boxreg_hmap_clean = boxreg_hmap_clean
        self.boxreg_mask       = boxreg_mask
        self.boxreg_flow       = boxreg_flow
        self.resize_target     = resize_target
        self.divide_target     = divide_target
        self.stn               = stn
        self.coarse_hmap       = coarse_hmap
        # Ignore sumaries_collections - model does not generate any summaries.
        self.outputs, self.state, self.gt, self.dbg = self._load_model(inputs, o)
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

        def pass_cnn(x, o, is_training, act):
            ''' Fully convolutional cnn.
            Either custom or other popular model.
            Note that Slim pre-defined networks can't be directly used, as they
            have some requirements that don't fit my inputs. Thus, for popular
            models I define network on my own following Slim definition.
            '''
            # NOTE: Use padding 'SAME' in convolutions and pooling so that
            # corners of before/after convolutions match! If training example
            # pairs are centered, it may not need to be this way though.
            # TODO: Not really sure if activation should really be gone. TEST.
            if o.cnn_model == 'custom':
                with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training},
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 16, 11, stride=2, scope='conv1')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool1')
                    x = slim.conv2d(x, 32, 5, stride=1, scope='conv2')
                    x = slim.max_pool2d(x, 3, stride=2, padding='SAME', scope='pool2')
                    x = slim.conv2d(x, 64, 3, stride=1, scope='conv3')
                    x = slim.conv2d(x, 128, 3, stride=1, scope='conv4')
                    x = slim.conv2d(x, 256, 3, stride=1, activation_fn=get_act(act), scope='conv5')
            elif o.cnn_model =='siamese': # exactly same as Siamese
                with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(o.wd)):
                    x = slim.conv2d(x, 96, 11, stride=2, scope='conv1')
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
            targets, weights = [], [] # without `weighted` average, training fails.
            targets.append(target); weights.append(1.0)

            if self.resize_target:
                # NOTE: To confirm the feature in this module, there should be
                # enough training pairs different in scale during training -> data augmentation.
                height = target.shape.as_list()[1]
                scales = [0.8, 1.2] # TODO: 1) more scales, 2) conv before/after resize - Relu okay?
                for s in scales:
                    targets.append(tf.image.resize_images(target,
                                                          [int(height*s)]*2,
                                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                          align_corners=True))
                    #weights.append(1.0 - abs(1.0 - s))
                    weights.append(1.0)

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

            # TODO: Need investigate more.
            #weights = [w/sum(weights) for w in weights]
            assert(len(weights)==len(targets))

            # JV: Remove dependence on explicit batch size.
            # scoremap = []
            # for b in range(o.batchsz):
            #     scoremap_each = []
            #     for k in range(len(targets)):
            #         scoremap_each.append(weights[k] *
            #                              tf.nn.depthwise_conv2d(tf.expand_dims(search[b], 0),
            #                                                     tf.expand_dims(targets[k][b], 3),
            #                                                     strides=[1,1,1,1],
            #                                                     padding='SAME'))
            #     scoremap.append(tf.add_n(scoremap_each))
            # scoremap = tf.concat(scoremap, 0)
            scoremap = []
            for k in range(len(targets)):
                scoremap.append(weights[k] * diag_conv(search, targets[k],
                                                       strides=[1, 1, 1, 1],
                                                       padding='SAME'))
            scoremap = tf.add_n(scoremap)

            # After cross-correlation, put some convolutions (separately from deconv).
            with slim.arg_scope([slim.conv2d],
                    num_outputs=scoremap.shape.as_list()[-1],
                    kernel_size=3,
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                scoremap = slim.conv2d(scoremap, scope='conv1')
                scoremap = slim.conv2d(scoremap, scope='conv2')
            return scoremap

        def pass_deconvolution(x, o):
            ''' Upsampling layers.
            The last layer should not have an activation!
            '''
            # NOTE: Not entirely sure yet if `align_corners` option is required.
            with slim.arg_scope([slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                if o.cnn_model in ['custom', 'siamese']:
                    if self.coarse_hmap:
                        # No upsample layers.
                        dim = x.shape.as_list()[-1]
                        x = slim.conv2d(x, num_outputs=dim, kernel_size=3, scope='deconv1')
                        x = slim.conv2d(x, num_outputs=2, kernel_size=1, activation_fn=None, scope='deconv2')
                    else:
                        x = tf.image.resize_images(x, [o.frmsz/2]*2, align_corners=True)
                        x = slim.conv2d(x, num_outputs=x.shape.as_list()[-1]/2, kernel_size=3, scope='deconv1')
                        x = tf.image.resize_images(x, [o.frmsz]*2, align_corners=True)
                        x = slim.conv2d(x, num_outputs=2, kernel_size=3, scope='deconv2')
                        x = slim.conv2d(x, num_outputs=2, kernel_size=1, activation_fn=None, scope='deconv3')

                        # previous shallow
                        #kernel_size=[1,1],
                        #x = slim.conv2d(x, num_outputs=x.shape.as_list()[-1], scope='deconv1')
                        #x = slim.conv2d(x, num_outputs=2, scope='deconv2')
                        #x = tf.image.resize_images(x, [o.frmsz, o.frmsz])
                        #x = slim.conv2d(x, num_outputs=2, activation_fn=None, scope='deconv3')
                        ##x = slim.conv2d(x, num_outputs=2, activation_fn=None, scope='deconv4')
                elif o.cnn_model == 'vgg_16':
                    assert False, 'Please update this better before using it..'
                    x = slim.conv2d(x, num_outputs=512, scope='deconv1')
                    x = tf.image.resize_images(x, [61, 61])
                    x = slim.conv2d(x, num_outputs=256, scope='deconv2')
                    x = tf.image.resize_images(x, [121, 121])
                    x = slim.conv2d(x, num_outputs=2, scope='deconv3')
                    x = tf.image.resize_images(x, [o.frmsz, o.frmsz])
                    x = slim.conv2d(x, num_outputs=2, activation_fn=None, scope='deconv4')
                else:
                    assert False, 'Not available option.'
            return x

        def pass_regress_box(x, is_training):
            ''' Regress output rectangle.
            '''
            # TODO: Batch norm or dropout.
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(o.wd)):
                with slim.arg_scope([slim.conv2d],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training}):
                    x = slim.conv2d(x, 32, 5, 2, scope='conv1')
                    x = slim.max_pool2d(x, 2, 2, scope='pool1')
                    x = slim.conv2d(x, 64, 5, 2, scope='conv2')
                    x = slim.max_pool2d(x, 2, 2, scope='pool2')
                    x = slim.conv2d(x, 128, 5, 2, scope='conv3')
                    x = slim.flatten(x)
                    x = slim.fully_connected(x, 4096, scope='fc1')
                    x = slim.fully_connected(x, 4096, scope='fc2')
                    x = slim.fully_connected(x, 4, activation_fn=None, scope='fc3')
            return x

        x           = inputs['x']  # shape [b, ntimesteps, h, w, 3]
        x0          = inputs['x0'] # shape [b, h, w, 3]
        y0          = inputs['y0'] # shape [b, 4]
        y           = inputs['y']  # shape [b, ntimesteps, 4]
        use_gt      = inputs['use_gt']
        gt_ratio    = inputs['gt_ratio']
        is_training = inputs['is_training']

        y0_size = tf.stack([y0[:,2]-y0[:,0], y0[:,3]-y0[:,1]], 1)
        y0 = enforce_inside_box(y0) # In OTB, Panda has GT error (i.e., > 1.0).

        # Add identity op to ensure that we can feed state here.
        x_init = tf.identity(x0)
        y_init = tf.identity(y0) # for `delta` regression type output.
        hmap_init = tf.identity(get_masks_from_rectangles(y0, o, Gaussian=True))

        x_prev = x_init
        y_prev = y_init
        #hmap_prev = hmap_init # Not being used anymore.

        y_pred = []
        hmap_pred = []
        # Case of object-centric approach.
        #y_gt_oc = []
        y_pred_oc = []
        hmap_gt_oc = []
        hmap_pred_oc = []
        box_s_raw = []
        box_s_val = []
        target = []
        search = []
        s_prev = []
        s_recon = []
        flow = []

        for t in range(o.ntimesteps):
            x_curr = x[:, t]
            y_curr_gt = y[:, t]

            # target and search images
            #target_curr = process_target_with_box(x_prev, y_prev, o)
            target_curr = tf.cond(is_training, # TODO: check the condition being passed.
                                  lambda: process_target_with_box(x_prev, y_prev, o),
                                  lambda: process_target_with_box(x0, y0, o))
            search_curr, box_s_raw_curr, box_s_val_curr = process_search_with_box(x_curr, y_prev, o)

            if self.stn:
                with tf.variable_scope('stn_localization', reuse=(t > 0)):
                    theta = pass_stn_localization(target_curr)
                    size = target_curr.shape.as_list()
                    target_curr = transformer(target_curr, theta, size[1:3])
                    target_curr.set_shape(size)

            with tf.variable_scope(o.cnn_model, reuse=(t > 0)) as scope:
                # TODO: Perform multi-scale resize for target here.
                target_feat = pass_cnn(target_curr, o, is_training, self.feat_act)
                if t == 0:
                    scope.reuse_variables() # two Siamese CNNs shared.
                search_feat = pass_cnn(search_curr, o, is_training, self.feat_act)

            if self.depth_wise_norm: # For NCC. It has some speed issue. # TODO: test this properly.
                search_feat = pass_depth_wise_norm(search_feat)
                target_feat = pass_depth_wise_norm(target_feat)

            with tf.variable_scope('cross_correlation', reuse=(t > 0)):
                scoremap = pass_cross_correlation(search_feat, target_feat, o)

            with tf.variable_scope('deconvolution', reuse=(t > 0)):
                hmap_curr_pred_oc = pass_deconvolution(scoremap, o)
                hmap_curr_pred_oc_fg = tf.expand_dims(tf.nn.softmax(hmap_curr_pred_oc)[:,:,:,0], -1)

            if self.boxreg: # regress box from `scoremap`.
                # Create inputs to regression network.
                boxreg_inputs = []
                if self.boxreg_imgpair: # Add raw image pair.
                    # TODO: image pair used to regress box can be different from image pair for flow.
                    # i.e., whether A0 is used or not. Note that this will make difference for sequence.
                    search_prev, _, _ = process_search_with_box(x_prev, y_prev, o)
                    img_pair = tf.concat([search_prev, search_curr], 3)
                    boxreg_inputs.append(img_pair)
                if self.boxreg_hmap: # Add hmap estimate.
                    boxreg_inputs.append(hmap_curr_pred_oc_fg) # TODO: never tested fg only yet.
                if self.boxreg_hmap_clean: # Add hmap clean. Use argmax & y0_size.
                    c_curr_pred_oc = find_center_in_scoremap(hmap_curr_pred_oc_fg, o)
                    c_curr_pred    = to_image_centric_coordinate(c_curr_pred_oc, box_s_raw_curr, o)
                    y_curr_pred    = tf.concat([c_curr_pred-y0_size*0.5, c_curr_pred+y0_size*0.5], 1)
                    #y_prev_size = tf.stack([y_prev[:,2]-y_prev[:,0], y_prev[:,3]-y_prev[:,1]], 1)
                    #y_curr_pred    = tf.concat([c_curr_pred-y_prev_size*0.5, c_curr_pred+y_prev_size*0.5], 1)
                    y_curr_pred_oc = to_object_centric_coordinate(y_curr_pred, box_s_raw_curr, box_s_val_curr, o)
                    new_hmap = get_masks_from_rectangles(y_curr_pred_oc, o, Gaussian=True)
                    boxreg_inputs.append(new_hmap)
                if self.boxreg_mask: # Add target_mask to inputs.
                    target_mask, _, _ = process_search_with_box(
                        get_masks_from_rectangles(y_prev, o), y_prev, o)
                    boxreg_inputs.append(target_mask)
                if self.boxreg_flow: # Add optical flow to inputs.
                    with tf.variable_scope('cnn_flow', reuse=(t > 0)):
                        flow_feat = pass_cnn(img_pair, o, is_training, 'relu')
                    with tf.variable_scope('deconvolution_flow', reuse=(t > 0)):
                        flow_curr_pred_oc = pass_deconvolution(flow_feat, o)
                        search_recon = spatial_transform_by_flow(search_curr, flow_curr_pred_oc)
                    # NOTE: Consider passing only flow of target (perhaps by masking).
                    boxreg_inputs.append(flow_curr_pred_oc)
                boxreg_inputs = tf.concat(boxreg_inputs, 3)
                boxreg_inputs = tf.stop_gradient(boxreg_inputs)
                # Box regression.
                with tf.variable_scope('regress_box', reuse=(t > 0)):
                    y_curr_pred_oc = pass_regress_box(boxreg_inputs, is_training)
                    y_curr_pred = to_image_centric_coordinate(y_curr_pred_oc, box_s_raw_curr, o)
            else: # argmax to find center (then use x0's box or regress box size - the latter didn't work)
                c_curr_pred_oc = find_center_in_scoremap(hmap_curr_pred_oc_fg, o)
                c_curr_pred    = to_image_centric_coordinate(c_curr_pred_oc, box_s_raw_curr, o)
                y_curr_pred    = tf.concat([c_curr_pred-y0_size*0.5, c_curr_pred+y0_size*0.5], 1)
                y_curr_pred_oc = to_object_centric_coordinate(y_curr_pred, box_s_raw_curr, box_s_val_curr, o)

            # Get image-centric outputs. Some are used for visualization purpose.
            y_curr_pred    = regularize_scale(y_prev, y_curr_pred, y0)
            y_curr_pred    = enforce_inside_box(y_curr_pred)
            hmap_curr_pred = to_image_centric_hmap(hmap_curr_pred_oc_fg, box_s_raw_curr, box_s_val_curr, o)

            y_pred.append(y_curr_pred)
            y_pred_oc.append(y_curr_pred_oc)
            hmap_pred.append(hmap_curr_pred)
            hmap_pred_oc.append(hmap_curr_pred_oc)
            box_s_raw.append(box_s_raw_curr)
            box_s_val.append(box_s_val_curr)
            target.append(target_curr) # To visualize what network sees.
            search.append(search_curr) # To visualize what network sees.
            s_prev.append(search_prev if self.boxreg_flow else None)
            s_recon.append(search_recon if self.boxreg_flow else None)
            flow.append(flow_curr_pred_oc if self.boxreg_flow else None)

            # for next time-step
            x_prev    = x_curr
            y_prev    = y_curr_pred # NOTE: For sequence learning, use scheduled sampling during training.
            hmap_prev = hmap_curr_pred

        y_pred        = tf.stack(y_pred, axis=1)
        hmap_pred     = tf.stack(hmap_pred, axis=1)
        y_pred_oc     = tf.stack(y_pred_oc, axis=1)
        hmap_pred_oc  = tf.stack(hmap_pred_oc, axis=1)
        #y_gt_oc      = tf.stack(y_gt_oc, axis=1)
        box_s_raw     = tf.stack(box_s_raw, axis=1)
        box_s_val     = tf.stack(box_s_val, axis=1)
        target        = tf.stack(target, axis=1)
        search        = tf.stack(search, axis=1)
        s_prev        = tf.stack(s_prev, axis=1) if self.boxreg_flow else None
        s_recon       = tf.stack(s_recon, axis=1) if self.boxreg_flow else None
        flow          = tf.stack(flow, axis=1) if self.boxreg_flow else None

        outputs = {'y':         {'ic': y_pred,    'oc': y_pred_oc}, # NOTE: Do not use 'ic' to compute loss.
                   'hmap':      {'ic': hmap_pred, 'oc': hmap_pred_oc}, # NOTE: hmap_pred_oc is no softmax yet.
                   'box_s_raw': box_s_raw,
                   'box_s_val': box_s_val,
                   'target':    target,
                   'search':    search,
                   's_prev':    s_prev,
                   's_recon':   s_recon,
                   'flow':      flow,
                   }
        state = {}
        state.update({'x': (x_init, x_prev)})
        state.update({'hmap': (hmap_init, hmap_prev)})
        state.update({'y': (y_init, y_prev)})
        gt = {}
        dbg = {} # dbg = {'h2': tf.stack(h2, axis=1), 'y_pred': y_pred}
        return outputs, state, gt, dbg


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
        # Ignore sumaries_collections - model does not generate any summaries.
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

