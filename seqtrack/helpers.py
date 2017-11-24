import pdb
import datetime
import errno
import json
import numpy as np
import os
from PIL import Image
import tensorflow as tf

import geom_np

def get_time():
    dt = datetime.datetime.now()
    time = '{0:04d}{1:02d}{2:02d}_h{3:02d}m{4:02d}s{5:02d}'.format(
            dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
    return time


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def getColormap(N,cmapname):
    '''Returns a function that maps each index in 0, 1, ..., N-1 to a distinct RGB color.'''
    colornorm = colors.Normalize(vmin=0, vmax=N-1)
    scalarmap = cmx.ScalarMappable(norm=colornorm, cmap=cmapname)
    def mapIndexToRgbColor(index):
        return scalarmap.to_rgba(index)
    return mapIndexToRgbColor

def createScalarMap(name='hot', vmin=-10, vmax=10):
    cm = plt.get_cmap(name)
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)

def load_image(fname, size=None, resize=False):
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    if size is not None:
        size = tuple(size)
        if im.size != size:
            if resize:
                im = im.resize(size)
            else:
                pdb.set_trace()
                raise ValueError('size does not match')
    return im

def crop_and_resize(src, box, size, pad_value):
    '''
    Args:
        src: PIL image
        size: (width, height)
    '''
    assert len(pad_value) == 3
    # Valid region in original image.
    src_valid = geom_np.rect_intersect(box, geom_np.unit_rect())
    # Valid region in box.
    box_valid = geom_np.crop_rect(src_valid, box)
    src_valid_pix = np.rint(geom_np.rect_mul(src_valid, src.size)).astype(np.int)
    box_valid_pix = np.rint(geom_np.rect_mul(box_valid, size)).astype(np.int)

    out = Image.new('RGB', size, pad_value)
    src_valid_pix_size = geom_np.rect_size(src_valid_pix)
    box_valid_pix_size = geom_np.rect_size(box_valid_pix)
    if all(src_valid_pix_size >= 1) and all(box_valid_pix_size >= 1):
        # Resize to final size in output image.
        src_valid_im = src.crop(src_valid_pix)
        out_valid_im = src_valid_im.resize(box_valid_pix_size)
        out.paste(out_valid_im, box_valid_pix)
    return out

def load_image_viewport(fname, viewport, size, pad_value=None):
    '''
    Args:
        size: (width, height)
    '''
    if pad_value is None:
        pad_value = (128, 128, 128)
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return crop_and_resize(im, viewport, size, pad_value)

def im_to_arr(x, dtype=np.float32):
    return np.array(x, dtype=dtype)

def pad_to(x, n, axis=0, mode='constant'):
    x = np.asarray(x)
    width = [(0, 0) for s in x.shape]
    width[axis] = (0, n - x.shape[axis])
    return np.pad(x, width, mode=mode)

def cache_json(filename, func, makedir=False):
    '''Caches the result of a function in a file.

    Args:
        func -- Function with zero arguments.
    '''
    if os.path.exists(filename):
        with open(filename, 'r') as r:
            result = json.load(r)
    else:
        if makedir:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        result = func()
        with open(filename, 'w') as w:
            json.dump(result, w)
    return result


def merge_dims(x, a, b):
    '''Merges dimensions a to b-1

    Returns:
        Reshaped tensor and a function to restore the shape.
    '''
    # Group dimensions of x into a, b-a, n-b:
    #     [0, ..., a-1 | a, ..., b-1 | b, ..., n-1]
    # Then y has dimensions grouped in: a, 1, n-b:
    #     [0, ..., a-1 | a | a+1, ..., a+n-b]
    # giving a total length of m = n-b+a+1.
    x_dynamic = tf.shape(x)
    x_static = x.shape.as_list()
    n = len(x_static)

    def restore(v, axis):
        '''Restores dimensions [axis] to dimensions [a, ..., b-1].'''
        v_dynamic = tf.shape(v)
        v_static = v.shape.as_list()
        m = len(v_static)
        # Substitute the static size where possible.
        u_dynamic = ([v_static[i] or v_dynamic[i] for i in range(0, axis)] +
                     [x_static[i] or x_dynamic[i] for i in range(a, b)] +
                     [v_static[i] or v_dynamic[i] for i in range(axis+1, m)])
        u = tf.reshape(v, u_dynamic)
        return u

    # Substitute the static size where possible.
    y_dynamic = ([x_static[i] or x_dynamic[i] for i in range(0, a)] +
                 [tf.reduce_prod(x_dynamic[a:b])] +
                 [x_static[i] or x_dynamic[i] for i in range(b, n)])
    y = tf.reshape(x, y_dynamic)
    return y, restore


def diag_conv(x, f, strides, padding, **kwargs):
    '''
    Args:
        x: [b, hx, wx, c]
        f: [b, hf, wf, c]

    Returns:
        [b, ho, wo, c]
    '''
    assert len(x.shape) == 4
    assert len(f.shape) == 4
    # x.shape is [b, hx, wx, c]
    # f.shape is [b, hf, wf, c]
    # [b, hx, wx, c] -> [hx, wx, b, c] -> [hx, wx, b*c]
    x, restore = merge_dims(tf.transpose(x, [1, 2, 0, 3]), 2, 4)
    # [b, hf, wf, c] -> [hf, wf, b, c] -> [hf, wf, b*c]
    f, _ = merge_dims(tf.transpose(f, [1, 2, 0, 3]), 2, 4)
    x = tf.expand_dims(x, axis=0) # [1, hx, wx, b*c]
    f = tf.expand_dims(f, axis=3) # [hf, wf, b*c, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding=padding, **kwargs)
    x = tf.squeeze(x, axis=0) # [ho, wo, b*c]
    # [ho, wo, b*c] -> [ho, wo, b, c] -> [b, ho, wo, c]
    x = tf.transpose(restore(x, axis=2), [2, 0, 1, 3])
    return x


def to_nested_tuple(tensor, value):
    if isinstance(tensor, dict) or isinstance(tensor, list) or isinstance(tensor, tuple):
        # Recurse on collection.
        if isinstance(tensor, dict):
            assert isinstance(value, dict)
            assert set(tensor.keys()) == set(value.keys())
            keys = sorted(tensor.keys()) # Not necessary but may as well.
            pairs = [(tensor[k], value[k]) for k in keys]
        else:
            assert isinstance(value, list) or isinstance(value, tuple)
            assert len(tensor) == len(value)
            pairs = zip(tensor, value)
        pairs = map(lambda pair: to_nested_tuple(*pair), pairs)
        # Remove pairs that are empty.
        pairs = filter(lambda x: x != (None, None), pairs)
        if len(pairs) == 0:
            return None, None
        # Convert from list of tuples to tuple of lists.
        tensor, value = zip(*pairs)
        # Convert from lists to tuples.
        return tuple(tensor), tuple(value)
    else:
        # TODO: Assert tensor is tf.Tensor?
        # TODO: Assert value is np.array?
        return tensor, value


def escape_filename(s):
    return s.replace('/', '_')


class LazyDict:

    '''
    Dictionary that delays evaluation of its values until required.

    Example:
        d = LazyDict()
        d['x'] = lambda: loadData() # assign lambda
        x1 = d['x'] # loadData() is called here
        x2 = d['x'] # loadData() is not called again
    '''

    def __init__(self):
        self._fn = {}
        self._cache = {}

    def __getitem__(self, k):
        if k in self._cache:
            return self._cache[k]
        v = self._fn[k]()
        self._cache[k] = v
        return v

    def __setitem__(self, k, v):
        if k in self._cache:
            del self._cache[k]
        self._fn[k] = v

    def __delitem__(self, k):
        if k in self._cache:
            del self._cache[k]
        del self._fn[k]


def map_nested(f, xs):
    if isinstance(xs, dict):
        return {k: map_nested(f, x) for k, x in xs.items()}
    if isinstance(xs, list):
        return [map_nested(f, x) for x in xs]
    if isinstance(xs, tuple):
        return tuple([map_nested(f, x) for x in xs])
    return f(xs)


# def softmax_cross_entropy_with_logits(
#         _sentinel=None,
#         labels=None,
#         logits=None,
#         dim=-1,
#         name=None):
#     sc_gt_valid, unmerge = merge_dims(sc_gt_valid, 0, 1)
#     sc_pred_valid, _ = merge_dims(sc_pred_valid, 0, 1)
#     loss_sc = tf.nn.softmax_cross_entropy_with_logits(labels=sc_gt, logits=outputs['sc']['out'])
