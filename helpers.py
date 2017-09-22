import pdb
import datetime
import errno
import json
import numpy as np
import os
from PIL import Image
import tensorflow as tf

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

def load_image(fname, size=None, method=None, resize=False):
    '''
    Args:
        size: (width, height)
        method: stretch or fit
    '''
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')

    if size is not None:
        size = tuple(size)
        if method == 'stretch':
            if im.size != size:
                if resize:
                    im = im.resize(size)
                else:
                    raise ValueError('wrong size: expect {} == {}'.format(im.size, size))
        elif method == 'fit':
            if not all(np.array(im.size) <= np.array(size)):
                if resize:
                    actual_size = np.asfarray(im.size)
                    max_size = np.asfarray(size)
                    scale = np.amin(max_size / actual_size)
                    im = im.resize(tuple(np.round(scale * max_size).astype(np.int)))
                else:
                    raise ValueError('wrong size: expect {} <= {}'.format(im.size, size))
        else:
            raise ValueError('unknown method: {}'.format(method))
    return im

def im_to_arr(x, dtype=np.float32):
    return np.array(x, dtype=dtype)

def pad_image_to(images, size, pad_value=0):
    '''
    Args:
        images: [..., height, width, channels]
        size: (width, height)
    '''
    width, height = size
    im_shape = list(np.shape(images))
    pad_shape = im_shape[:-3] + [height, width] + im_shape[-1:]
    return np.pad(images,
                  [(0, pad-im) for im, pad in zip(im_shape, pad_shape)],
                  mode='constant',
                  constant_values=pad_value)

def pad_to(x, n, axis=0, mode='constant'):
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
        # Convert from list of tuples to tuple of lists.
        if len(pairs) == 0:
            import pdb ; pdb.set_trace()
        tensor, value = zip(*pairs)
        # Convert from lists to tuples.
        return tuple(tensor), tuple(value)
    else:
        # TODO: Assert tensor is tf.Tensor?
        # TODO: Assert value is np.array?
        return tensor, value


def most_specific_shape(x):
    static = x.shape.as_list()
    dynamic = tf.unstack(tf.shape(x))
    assert len(static) == len(dynamic)
    return [s or d for s, d in zip(static, dynamic)]
