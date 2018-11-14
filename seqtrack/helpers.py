from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import errno
import functools
import itertools
import json
import msgpack
import numpy as np
import os
import sys
import tempfile
import tensorflow as tf
import time
from PIL import Image
from contextlib import contextmanager

from tensorflow.contrib.layers.python.layers import utils as layer_utils

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom
from seqtrack import geom_np


Codec = collections.namedtuple('Codec', ['module', 'ext', 'binary'])

CODECS = {
    'json': Codec(module=json, ext='.json', binary=False),
    'msgpack': Codec(module=msgpack, ext='.msgpack', binary=True),
}


def get_time():
    dt = datetime.datetime.now()
    time = '{0:04d}{1:02d}{2:02d}_h{3:02d}m{4:02d}s{5:02d}'.format(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return time


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


def getColormap(N, cmapname):
    '''Returns a function that maps each index in 0, 1, ..., N-1 to a distinct RGB color.'''
    colornorm = colors.Normalize(vmin=0, vmax=N - 1)
    scalarmap = cmx.ScalarMappable(norm=colornorm, cmap=cmapname)

    def mapIndexToRgbColor(index):
        return scalarmap.to_rgba(index)
    return mapIndexToRgbColor


def createScalarMap(name='hot', vmin=-10, vmax=10):
    cm = plt.get_cmap(name)
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)


def load_image(fname, size_hw=None, resize=False):
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    if size_hw is not None:
        height, width = size_hw
        size_wh = (width, height)
        if im.size != size_wh:
            if resize:
                im = im.resize(size_wh)
            else:
                pdb.set_trace()
                raise ValueError('size does not match')
    return im


def crop_and_resize(src, box, size_hw, pad_value):
    '''
    Args:
        src: PIL image
    '''
    assert len(pad_value) == 3
    height, width = size_hw
    size_wh = (width, height)
    # Valid region in original image.
    src_valid = geom_np.rect_intersect(box, geom_np.unit_rect())
    # Valid region in box.
    box_valid = geom_np.crop_rect(src_valid, box)
    src_valid_pix = np.rint(geom_np.rect_mul(src_valid, src.size)).astype(np.int)
    box_valid_pix = np.rint(geom_np.rect_mul(box_valid, size_wh)).astype(np.int)

    out = Image.new('RGB', size_wh, pad_value)
    src_valid_pix_size = geom_np.rect_size(src_valid_pix)
    box_valid_pix_size = geom_np.rect_size(box_valid_pix)
    if all(src_valid_pix_size >= 1) and all(box_valid_pix_size >= 1):
        # Resize to final size in output image.
        src_valid_im = src.crop(src_valid_pix)
        out_valid_im = src_valid_im.resize(box_valid_pix_size)
        out.paste(out_valid_im, box_valid_pix)
    return out


def load_image_viewport(fname, viewport, size_hw, pad_value=None):
    if pad_value is None:
        pad_value = (128, 128, 128)
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return crop_and_resize(im, viewport, size_hw, pad_value)


def im_to_arr(x, dtype=np.float32):
    # return np.array(x, dtype=dtype)
    return (1. / 255) * np.array(x, dtype=dtype)


def pad_to(x, n, axis=0, mode='constant'):
    x = np.asarray(x)
    width = [(0, 0) for s in x.shape]
    width[axis] = (0, n - x.shape[axis])
    return np.pad(x, width, mode=mode)


def cache(codec_module, filename, func, makedir=False, perm=0o644, binary=False):
    '''Caches the result of a function in a file.

    Args:
        codec_module -- Object with methods dump() and load().
        func -- Function with zero arguments.
    '''
    if os.path.exists(filename):
        logger.info('load from cache file: "%s"', filename)
        start = time.clock()
        with open(filename, 'r') as f:
            result = codec_module.load(f)
        dur = time.clock() - start
        logger.info('time to load cache: %.3g sec "%s"', dur, filename)
    else:
        logger.info('no cache found: "%s"', filename)
        dir, basename = os.path.split(filename)
        if not os.path.exists(dir):
            if makedir:
                os.makedirs(dir)
            else:
                raise RuntimeError('makedir is false and dir does not exist: {}'.format(dir))
        # Create temporary file in same directory.
        result = func()
        # TODO: Clean up tmp file on exception.
        start = time.clock()
        with tempfile.NamedTemporaryFile(delete=False, dir=dir, suffix=basename) as f:
            codec_module.dump(result, f)
        dur = time.clock() - start
        os.chmod(f.name, perm)  # Default permissions are 600.
        os.rename(f.name, filename)
        logger.info('time to dump cache: %.3g sec "%s"', dur, filename)
    return result


cache_json = functools.partial(cache, json)


def merge_dims(x, a, b, name='merge_dims'):
    '''Merges dimensions a to b-1

    Returns:
        Reshaped tensor and a function to restore the shape.
    '''
    n = len(x.shape)
    a, b = _array_interval(a, b, n)

    def restore(v, axis, x_static, x_dynamic, name='restore'):
        with tf.name_scope(name) as scope:
            '''Restores dimensions [axis] to dimensions [a, ..., b-1].'''
            v_dynamic = tf.unstack(tf.shape(v))
            v_static = v.shape.as_list()
            m = len(v_static)
            # Substitute the static size where possible.
            u_dynamic = ([v_static[i] or v_dynamic[i] for i in range(0, axis)] +
                         [x_static[i] or x_dynamic[i] for i in range(a, b)] +
                         [v_static[i] or v_dynamic[i] for i in range(axis + 1, m)])
            u = tf.reshape(v, u_dynamic)
            return u

    with tf.name_scope(name) as scope:
        # Group dimensions of x into a, b-a, n-b:
        #     [0, ..., a-1 | a, ..., b-1 | b, ..., n-1]
        # Then y has dimensions grouped in: a, 1, n-b:
        #     [0, ..., a-1 | a | a+1, ..., a+n-b]
        # giving a total length of m = n-b+a+1.
        x_dynamic = tf.unstack(tf.shape(x))
        x_static = x.shape.as_list()

        def prod(xs):
            return functools.reduce(lambda x, y: x * y, xs)
        # Substitute the static size where possible.
        y_dynamic = ([x_static[i] or x_dynamic[i] for i in range(0, a)] +
                     [prod([x_static[i] or x_dynamic[i] for i in range(a, b)])] +
                     [x_static[i] or x_dynamic[i] for i in range(b, n)])
        y = tf.reshape(x, y_dynamic)
        restore_fn = functools.partial(restore, x_static=x_static, x_dynamic=x_dynamic)
        return y, restore_fn


def expand_dims_n(input, axis=None, n=1, name=None):
    for i in range(n):
        input = tf.expand_dims(input, axis, name=name)
    return input


def diag_xcorr(x, f, stride=1, padding='VALID', name='diag_xcorr', **kwargs):
    '''
    Args:
        x: [b, ..., hx, wx, c]
        f: [b, hf, wf, c]

    Returns:
        [b, ..., ho, wo, c]
    '''
    with tf.name_scope(name) as scope:
        assert len(f.shape) == 4
        if len(x.shape) == 4:
            x = tf.expand_dims(x, 1)
            x = diag_xcorr(x, f, stride, padding, name=name, **kwargs)
            x = tf.squeeze(x, 1)
            return x
        if len(x.shape) > 5:
            # Merge dims 0, (1, ..., n-4), n-3, n-2, n-1
            x, restore = merge_dims(x, 0, len(x.shape) - 3)
            x = diag_xcorr(x, f, stride, padding, name=name, **kwargs)
            x = restore(x, 1)
            return x
        assert len(x.shape) == 5
        stride = layer_utils.n_positive_integers(2, stride)

        # x.shape is [b, n, hx, wx, c]
        # f.shape is [b, hf, wf, c]
        # [b, n, hx, wx, c] -> [n, hx, wx, b, c] -> [n, hx, wx, b*c]
        x, restore = merge_dims(tf.transpose(x, [1, 2, 3, 0, 4]), 3, 5)
        # [b, hf, wf, c] -> [hf, wf, b, c] -> [hf, wf, b*c]
        f, _ = merge_dims(tf.transpose(f, [1, 2, 0, 3]), 2, 4)
        f = tf.expand_dims(f, axis=3)  # [hf, wf, b*c, 1]
        strides = [1, stride[0], stride[1], 1]
        x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding=padding, **kwargs)
        # [n, ho, wo, b*c] -> [n, ho, wo, b, c] -> [b, n, ho, wo, c]
        x = tf.transpose(restore(x, axis=3), [3, 0, 1, 2, 4])
        return x


def to_nested_tuple(tensor, value):
    if isinstance(tensor, dict) or isinstance(tensor, list) or isinstance(tensor, tuple):
        # Recurse on collection.
        if isinstance(tensor, dict):
            assert isinstance(value, dict)
            assert set(tensor.keys()) == set(value.keys())
            keys = sorted(tensor.keys())  # Not necessary but may as well.
            pairs = [(tensor[k], value[k]) for k in keys]
        else:
            assert isinstance(value, list) or isinstance(value, tuple)
            assert len(tensor) == len(value)
            pairs = zip(tensor, value)
        pairs = list(map(lambda pair: to_nested_tuple(*pair), pairs))
        # Remove pairs that are empty.
        pairs = list(filter(lambda x: x != (None, None), pairs))
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


def modify_aspect_ratio(rect, method='stretch', name='modify_aspect_ratio'):
    if method == 'stretch':
        return rect  # No change.
    with tf.name_scope(name) as scope:
        EPSILON = 1e-3
        min_pt, max_pt = geom.rect_min_max(rect)
        center, size = 0.5 * (min_pt + max_pt), max_pt - min_pt
        with tf.control_dependencies([tf.assert_greater_equal(size, 0.0)]):
            size = tf.identity(size)
        if method == 'perimeter':
            # Average of dimensions.
            width = tf.reduce_mean(size, axis=-1, keepdims=True)
            return geom.make_rect(center - 0.5 * width, center + 0.5 * width)
        if method == 'area':
            # Geometric average of dimensions.
            width = tf.exp(tf.reduce_mean(tf.log(tf.maximum(size, EPSILON)),
                                          axis=-1,
                                          keepdims=True))
            return geom.make_rect(center - 0.5 * width, center + 0.5 * width)
        raise ValueError('unknown method: {}'.format(method))


def get_act(act):
    if act == 'relu':
        return tf.nn.relu
    elif act == 'tanh':
        return tf.nn.tanh
    elif act == 'leaky':
        return leaky_relu
    elif act == 'linear':
        return None
    else:
        raise ValueError('wrong activation type: {}'.format(act))


def leaky_relu(x, name='leaky_relu'):
    with tf.name_scope(name) as scope:
        return tf.maximum(0.1 * x, x, name=scope)


def weighted_mean(x, w, axis=None, keepdims=False, name='weighted_mean'):
    with tf.name_scope(name) as scope:
        w = tf.to_float(w)
        p = normalize_prob(w * tf.ones_like(x), axis=axis)
        return tf.reduce_sum(p * x, axis=axis, keepdims=keepdims)


def normalize_prob(x, axis=None, name='normalize'):
    with tf.name_scope(name) as scope:
        with tf.control_dependencies([tf.assert_non_negative(x)]):
            x = tf.identity(x)
        z = tf.reduce_sum(x, axis=axis, keepdims=True)
        with tf.control_dependencies([tf.assert_positive(z)]):
            p = (1. / z) * x
        return p


def most_static_shape(x):
    return [s or d for s, d in zip(x.shape.as_list(), tf.unstack(tf.shape(x)))]


def stack_dict(frames, axis=0, keys=None):
    '''Converts list of dictionaries to dictionary of tensors.'''
    if keys is None:
        keys = frames[0].keys()
    return {
        k: tf.stack([frame[k] for frame in frames], axis=axis)
        for k in keys
    }


def quote(s):
    return "'" + s + "'"


def quote_list(x):
    return ', '.join(map(quote, x))


# def unstack_dict(d, keys, axis):
#     '''Converts dictionary of tensors to list of dictionaries.'''
#     # Gather lists of all elements at same index.
#     # {'x': [x0, x1], 'y': [y0, y1]} => [[x0, y0], [x1, y1]]
#     value_lists = zip(*[tf.unstack(d[k], axis=axis) for k in keys])
#     # Create a dictionary from each.
#     # [[x0, y0], [x1, y1]] => [{'x': x0, 'y': y0}, {'x': x1, 'y': y1}]
#     return [dict(zip(keys, vals)) for vals in value_lists]


class ProgressMeter(object):

    def __init__(self, interval_num=0, interval_time=0, num_to_print=0, print_func=None):
        self._interval_num = interval_num
        self._interval_time = interval_time
        self._num_to_print = num_to_print
        self._print_func = print_func or default_print_func

    def __call__(self, elems):
        self._n = len(elems)
        self._iter = iter(elems)
        self._i = 0
        self._start = time.time()
        self._prev_num = None
        self._prev_time = None
        self._prev_progress = None
        return self

    def __iter__(self):
        return self

    def __next__(self):
        try:
            elem = next(self._iter)
            self._increment()
            return elem
        except StopIteration:
            self._increment()
            raise

    next = __next__  # For Python 2

    def _progress(self):
        return int(self._i * self._num_to_print / self._n)

    def _to_print(self):
        if self._i == self._n:
            return True
        if self._interval_num:
            if self._prev_num is None or self._i - self._prev_num >= self._interval_num:
                return True
        if self._interval_time:
            if self._prev_time is None or time.time() - self._prev_time >= self._interval_time:
                return True
        if self._num_to_print:
            if self._prev_progress is None or self._progress() - self._prev_progress >= 1:
                return True
        return False

    def _increment(self):
        if self._to_print():
            now = time.time()
            self._print_func(self._i, self._n, now - self._start)
            self._prev_num = self._i
            self._prev_time = now
            self._prev_progress = self._progress()
        self._i += 1


def default_print_func(i, n, time_elapsed):
    if i == 0:
        return
    time_per_elem = float(time_elapsed) / i
    if n is None:
        progress_str = '{:d}'.format(i)
    else:
        percent = float(i) / n * 100
        time_total = float(n) * time_per_elem
        progress_str = '{:3.0f}% ({:d}/{:d})'.format(percent, i, n)
    progress_str += '; time elapsed {} ({:.3g} sec each)'.format(
        str(datetime.timedelta(seconds=round(time_elapsed))), time_per_elem)
    if n is not None and i < n:
        time_rem = time_total - time_elapsed
        progress_str += '; remaining {} of {}'.format(
            str(datetime.timedelta(seconds=round(time_rem))),
            str(datetime.timedelta(seconds=round(time_total))))
    print(progress_str, file=sys.stderr)


class MapContext(object):

    def __init__(self, name):
        self.name = name

    def tmp_dir(self):
        '''Returns None to mean that a tmp_dir must be created if needed.'''
        return None


def map_dict(func, items):
    for k, v in items:
        yield k, func(MapContext(k), v)


def map_dict_list(func, items):
    return list(map_dict(func, items))


def filter_dict(func, items):
    for k, v in items:
        if func(k, v):
            yield k, v


class CachedDictMapper(object):

    def __init__(self, dir, codec_name='json', mapper=None):
        # Default to simple mapper.
        mapper = mapper or map_dict
        self._dir = dir
        self._codec_name = codec_name
        self._mapper = mapper

    def __call__(self, func, items):
        cache_filter = CacheFilter(self._dir, codec_name=self._codec_name)
        uncached_items = cache_filter.filter(items)
        uncached_results = self._mapper(func, uncached_items)
        # Construct list here to deliberately not be lazy (write when available).
        uncached_results = list(_dump_cache_for_each(uncached_results, self._dir,
                                                     codec_name=self._codec_name))
        # Combine cached and uncached results.
        return itertools.chain(cache_filter.results, uncached_results)


def _dump_cache_for_each(items, dir, codec_name='json'):
    codec = CODECS[codec_name]
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    for k, v in items:
        fname = os.path.join(dir, k + codec.ext)
        with open_atomic_write(fname, 'wb' if codec.binary else 'w') as f:
            codec.module.dump(v, f)
        yield k, v


@contextmanager
def open_atomic_write(filename, mode='w+b', perm=0o644):
    dir, basename = os.path.split(filename)
    with tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=dir, suffix=basename) as f:
        yield f
    os.chmod(f.name, perm)  # Default permissions are 600.
    os.rename(f.name, filename)


class CacheFilter(object):
    '''Keeps only items that do not have a cached value.

    Items that do have a cache are loaded and stored in results.
    '''

    def __init__(self, dir, codec_name='json'):
        self.results = []
        self._dir = dir
        self._codec_name = codec_name

    def filter(self, items):
        codec = CODECS[self._codec_name]
        for k, v in items:
            fname = os.path.join(self._dir, k + codec.ext)
            if os.path.exists(fname):
                with open(fname, 'rb' if codec.binary else 'r') as f:
                    result = codec.module.load(f)
                self.results.append((k, result))
            else:
                yield k, v


def assert_partition(x, subsets):
    assert isinstance(x, set)
    counts = {}
    for subset in subsets:
        for elem in subset:
            counts[elem] = counts.get(elem, 0) + 1
    union = set(counts.keys())
    missing = x.difference(union)
    if len(missing) > 0:
        raise RuntimeError('missing from partition: {}'.format(helpers.quote_list(missing)))
    extra = union.difference(x)
    if len(extra) > 0:
        raise RuntimeError('extra in partition: {}'.format(helpers.quote_list(extra)))
    multiple = [elem for elem, count in counts.items() if count > 1]
    if len(multiple) > 0:
        raise RuntimeError('repeated in partition: {}'.format(helpers.quote_list(multiple)))
    # Unnecessary but just for sanity.
    assert len(x) == sum(map(len, subsets))


def assert_key_subset(lhs, rhs):
    extra = set(lhs.keys()).difference(set(rhs.keys()))
    if len(extra) > 0:
        raise RuntimeError('extra keys: {}'.format(str(list(extra))))


def unique_value(elems):
    '''Returns the single element which is repeated in elems.

    Raises an exception if elems is empty or contains diverse elements.
    '''
    first = None
    i = 0
    for x in elems:
        if i == 0:
            first = x
        else:
            assert x == first, 'element {} not equal: {} != {}'.format(i, x, first)
        i += 1
    if i == 0:
        raise ValueError('empty collection')
    return first


def round_lattice(size, stride, x):
    i = int(round(max(x - size, 0) / stride))
    return i * stride + size


class DictAccumulator(object):

    def __init__(self):
        self._totals = {}
        self._counts = {}

    def add(self, metrics):
        for key, value in metrics.items():
            self._totals[key] = self._totals.get(key, 0) + value
            self._counts[key] = self._counts.get(key, 0) + 1

    def mean(self):
        mean = {}
        for key in self._totals:
            mean[key] = self._totals[key] / self._counts[key]
        return mean

    def reset(self):
        self._totals = {}
        self._counts = {}

    def flush(self):
        '''Return mean and then reset.'''
        mean = self.mean()
        self.reset()
        return mean


def _array_interval(a, b, n):
    if a is None:
        a = 0
    elif a < 0:
        a += n
    if b is None:
        b = n
    elif b < 0:
        b += n
    return a, b


def known_spatial_dim(x):
    size = x.shape[-3:-1].as_list()
    assert all(n is not None for n in size), 'unknown spatial dim: {:s}'.format(x.shape.as_list())
    return size
