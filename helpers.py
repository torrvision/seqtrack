import pdb
import datetime
import errno
import simplejson as json
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
    '''src is a PIL image'''
    assert len(pad_value) == 3
    # Valid region in original image.
    valid_src = geom_np.rect_intersect(box, geom_np.rect_identity())
    # Valid region in box.
    valid_box = geom_np.crop_rect(valid_src, box)

    valid_src_abs = np.rint(geom_np.rect_mul(valid_src, src.size)).astype(np.int)
    valid_box_abs = np.rint(geom_np.rect_mul(valid_box, size)).astype(np.int)

    im_valid_src = src.crop(valid_src_abs)
    # Resize to final size in output image.
    im_valid_out = im_valid_src.resize(geom_np.rect_size(valid_box_abs))
    out = Image.new('RGB', size, pad_value)
    out.paste(im_valid_out, valid_box_abs)
    import pdb ; pdb.set_trace()
    return out

def load_image_viewport(fname, viewport, size, pad_value=None):
    if pad_value is None:
        pad_value = (128, 128, 128)
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return crop_and_resize(im, viewport, size, pad_value)

def im_to_arr(x, dtype=np.float32):
    return (1.0/255) * np.array(x, dtype=dtype)

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


def merge_dims(x, a, b, name='merge'):
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

    def restore(v, axis, name='restore'):
        '''Restores dimensions [axis] to dimensions [a, ..., b-1].'''
        with tf.name_scope(name):
            v_dynamic = tf.shape(v)
            v_static = v.shape.as_list()
            m = len(v_static)
            # Substitute the static size where possible.
            u_dynamic = ([v_static[i] or v_dynamic[i] for i in range(0, axis)] +
                         [x_static[i] or x_dynamic[i] for i in range(a, b)] +
                         [v_static[i] or v_dynamic[i] for i in range(axis+1, m)])
            u = tf.reshape(v, u_dynamic)
        return u

    with tf.name_scope(name):
        # Substitute the static size where possible.
        y_dynamic = ([x_static[i] or x_dynamic[i] for i in range(0, a)] +
                     [tf.reduce_prod(x_dynamic[a:b])] +
                     [x_static[i] or x_dynamic[i] for i in range(b, n)])
        y = tf.reshape(x, y_dynamic)
    return y, restore


def most_static_shape(x):
    return [s or d for s, d in zip(x.shape.as_list(), tf.unstack(tf.shape(x)))]
