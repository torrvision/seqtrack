from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import csv
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

nest = tf.contrib.framework.nest
from tensorflow.contrib.layers.python.layers import utils as layer_utils

import matplotlib
# Assume that user will use('Agg') before importing this module if necessary.
# matplotlib.use('Agg')

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom
from seqtrack import geom_np


def json_default(x):
    if isinstance(x, np.generic):
        return x.tolist()
    else:
        return x


Codec = collections.namedtuple('Codec', ['dump', 'load', 'ext', 'binary'])

CODECS = {
    'json': Codec(
        dump=functools.partial(json.dump, sort_keys=True, default=json_default),
        load=json.load,
        ext='.json',
        binary=False,
    ),
    'msgpack': Codec(
        # The newer version of msgpack now supports a binary type.
        # msgpack.dump(..., use_bin_type=False) will pack {bytes, string} -> "raw".
        # msgpack.dump(..., use_bin_type=True) will pack bytes -> "bin" and string -> "raw".
        # msgpack.load(..., raw=True) will unpack "raw" -> bytes.
        # msgpack.load(..., raw=False) will unpack "raw" -> string and "bin" -> bytes.
        # Therefore, to decode strings, we need raw=False.
        # The value of use_bin_type does not matter since we never encode bytes.
        # However, it's probably best to use use_bin_type=True with raw=False.
        dump=functools.partial(msgpack.dump, use_bin_type=True),
        load=functools.partial(msgpack.load, raw=False),
        ext='.msgpack',
        binary=True,
    ),
}


def get_time():
    dt = datetime.datetime.now()
    time = '{0:04d}{1:02d}{2:02d}_h{3:02d}m{4:02d}s{5:02d}'.format(
        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return time


def getColormap(N, cmapname):
    '''Returns a function that maps each index in 0, 1, ..., N-1 to a distinct RGB color.'''
    colornorm = matplotlib.colors.Normalize(vmin=0, vmax=N - 1)
    scalarmap = matplotlib.cmx.ScalarMappable(norm=colornorm, cmap=cmapname)

    def mapIndexToRgbColor(index):
        return scalarmap.to_rgba(index)
    return mapIndexToRgbColor


def createScalarMap(name='hot', vmin=-10, vmax=10):
    # Import pyplot here to avoid need to call matplotlib.use().
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(name)
    cNorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return matplotlib.cmx.ScalarMappable(norm=cNorm, cmap=cm)


def pad_to(x, n, axis=0, mode='constant'):
    x = np.asarray(x)
    width = [(0, 0) for s in x.shape]
    width[axis] = (0, n - x.shape[axis])
    return np.pad(x, width, mode=mode)


def cache(codec_name, filename, func, makedir=False):
    '''Caches the result of a function in a file.

    Args:
        codec_name -- String
        func -- Function with zero arguments.
    '''
    codec = CODECS[codec_name]
    if os.path.exists(filename):
        logger.info('load from cache file: "%s"', filename)
        start = time.clock()
        with open(filename, 'rb' if codec.binary else 'r') as f:
            result = codec.load(f)
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
        with open_atomic_write(filename, 'wb' if codec.binary else 'w') as f:
            codec.dump(result, f)
        dur = time.clock() - start
        logger.info('time to dump cache: %.3g sec "%s"', dur, filename)
    return result


cache_json = functools.partial(cache, 'json')


def merge_dims(x, a, b, name='merge_dims'):
    '''Merges dimensions a to b-1

    Returns:
        Reshaped tensor and a function to restore the shape.
    '''
    with tf.name_scope(name) as scope:
        n = len(x.shape)
        a, b = _array_interval(a, b, n)
        x_dynamic = tf.unstack(tf.shape(x))
        x_static = x.shape.as_list()
        # Substitute the static size where possible.
        x_shape = [x_static[i] or x_dynamic[i] for i in range(n)]
        def prod(xs):
            return functools.reduce(lambda x, y: x * y, xs, 1)
        y_shape = x_shape[:a] + [prod(x_shape[a:b])] + x_shape[b:]
        y = tf.reshape(x, y_shape)
        restore_fn = functools.partial(split_dims, shape=x_shape[a:b])
        return y, restore_fn


def split_dims(v, axis, shape, name='split'):
    '''Split single dimension `axis` into `shape`.'''
    with tf.name_scope(name) as scope:
        m = len(v.shape)
        shape = list(shape)
        v_dynamic = tf.unstack(tf.shape(v))
        v_static = v.shape.as_list()
        v_shape = [v_static[i] or v_dynamic[i] for i in range(m)]
        u_shape = v_shape[:axis] + list(shape) + v_shape[axis:][1:]
        u = tf.reshape(v, u_shape)
        return u


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


# def to_nested_tuple(tensor, value):
#     if isinstance(tensor, dict) or isinstance(tensor, list) or isinstance(tensor, tuple):
#         # Recurse on collection.
#         if isinstance(tensor, dict):
#             assert isinstance(value, dict)
#             assert set(tensor.keys()) == set(value.keys())
#             keys = sorted(tensor.keys())  # Not necessary but may as well.
#             pairs = [(tensor[k], value[k]) for k in keys]
#         else:
#             assert isinstance(value, list) or isinstance(value, tuple)
#             assert len(tensor) == len(value)
#             pairs = zip(tensor, value)
#         pairs = list(map(lambda pair: to_nested_tuple(*pair), pairs))
#         # Remove pairs that are empty.
#         pairs = list(filter(lambda x: x != (None, None), pairs))
#         if len(pairs) == 0:
#             return None, None
#         # Convert from list of tuples to tuple of lists.
#         tensor, value = zip(*pairs)
#         # Convert from lists to tuples.
#         return tuple(tensor), tuple(value)
#     else:
#         # TODO: Assert tensor is tf.Tensor?
#         # TODO: Assert value is np.array?
#         return tensor, value


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


def scalar_size(size, method, axis=-1, keepdims=False, name='rect_magnitude'):
    with tf.name_scope(name) as scope:
        if method == 'perimeter':
            return tf.reduce_mean(size, axis=axis, keepdims=keepdims)
        elif method == 'area':
            return tf.exp(tf.reduce_mean(tf.log(size), axis=axis, keepdims=keepdims))
        else:
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


def stack_structure(elems, axis=0):
    '''
    Args:
        elems: List of (identically) structured elements.
    '''
    assert len(elems) > 0
    structure = elems[0]
    for i in range(1, len(elems)):
        nest.assert_same_structure(structure, elems[i])
    elems = [nest.flatten(x) for x in elems]
    fields = [tf.stack(field, axis=axis) for field in zip(*elems)]
    fields = nest.pack_sequence_as(structure, fields)
    return fields


def unstack_structure(fields, axis=0):
    '''
    Args:
        elems: List of (identically) structured elements.
    '''
    structure = fields
    fields = nest.flatten(fields)
    elems = list(zip(*[tf.unstack(field, axis=axis) for field in fields]))
    elems = [nest.pack_sequence_as(structure, elem) for elem in elems]
    return elems


def quote(s):
    return "'" + s + "'"


def quote_list(x):
    return ', '.join(map(quote, x))


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


def map_dict_list(func, items):
    return list(map_dict(func, items))


def filter_dict(func, items):
    for k, v in items:
        if func(k, v):
            yield k, v


@contextlib.contextmanager
def open_atomic_write(filename, mode='w+b', perm=0o644):
    dir, basename = os.path.split(filename)
    with tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=dir, suffix=basename) as f:
        yield f
    os.chmod(f.name, perm)  # Default permissions are 600.
    os.rename(f.name, filename)


def assert_partition(x, subsets):
    assert isinstance(x, set)
    counts = {}
    for subset in subsets:
        for elem in subset:
            counts[elem] = counts.get(elem, 0) + 1
    union = set(counts.keys())
    missing = x.difference(union)
    if len(missing) > 0:
        raise RuntimeError('missing from partition: {}'.format(quote_list(missing)))
    extra = union.difference(x)
    if len(extra) > 0:
        raise RuntimeError('extra in partition: {}'.format(quote_list(extra)))
    multiple = [elem for elem, count in counts.items() if count > 1]
    if len(multiple) > 0:
        raise RuntimeError('repeated in partition: {}'.format(quote_list(multiple)))
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


def set_xscale_log(ax):
    ax.set_xscale('log')
    major_subs = np.array([1, 2, 5])
    minor_subs = np.array(sorted(set(range(1, 10)) - set(major_subs)))
    ax.xaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=major_subs, numticks=12))
    ax.xaxis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=minor_subs, numticks=12))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def wrap_merge_batch_dims_and_call(n, fn, x, *args, **kwargs):
    '''
    If `x` has `N` dimensions, then the first `N - n` are merged before calling `fn`.
    The structure is then restored.
    '''
    assert n >= 0
    num_batch_dims = len(x.shape) - n
    if num_batch_dims > 1:
        # We must merge the multiple batch dimensions into one.
        x, restore_fn = merge_dims(x, None, -n)
        y = fn(x, *args, **kwargs)
        y = restore_fn(y, 0)
    else:
        # No merge necessary.
        y = fn(x, *args, **kwargs)
    return y


def wrap_merge_batch_dims(n, fn):
    return functools.partial(wrap_merge_batch_dims_and_call, n, fn)


def merge_batch_dims_decorator(n):
    return functools.partial(wrap_merge_batch_dims, n)


def wrap_merge_batch_and_map(n, fn, x, *args, **kwargs):
    return wrap_merge_batch_dims_and_call(
        n,
        functools.partial(tf.map_fn, fn),
        x,
        *args,
        **kwargs)


def merge_dicts(*args):
    return dict(itertools.chain.from_iterable(x.items() for x in args))


def flatten_dict(keys, values):
    # return flatten_items(zip(keys, values))
    nest.assert_shallow_structure(keys, values)
    return dict(zip(nest.flatten(keys),
                    nest.flatten_up_to(keys, values)))


def flatten_items(items):
    '''Maps list of (key, value) pairs to list of flattened (key, value) pairs.

    This is useful for constructing a `feed_dict` when the key is a dictionary of tensors.

    >>> flatten_items([(1, 2), ([3, 4], [5, 6]), ({'a': 7, 'b': 8}, {'a': 9, 'b': 10})])
    {1: 2, 3: 5, 4: 6, 7: 9, 8: 10}
    '''
    for k, v in items:
        nest.assert_shallow_structure(k, v)
        yield (nest.flatten(k), nest.flatten_up_to(k, v))


def dump_csv(f, series, sort_keys=True, sort_fields=True):
    assert len(series) > 0
    keys = list(series.keys())
    if sort_keys:
        keys = sorted(keys)
    fields = list(series[keys[0]])
    # TODO: Assert all the same or take (order-preserving?) union?
    if sort_fields:
        fields = sorted(fields)
    writer = csv.DictWriter(f, fieldnames=['key'] + fields)
    writer.writeheader()
    for key in keys:
        row = dict(series[key])
        assert 'key' not in row
        row['key'] = key
        writer.writerow(row)


def mkdir_p(*args, **kwargs):
    try:
        os.makedirs(*args, **kwargs)
    except OSError as ex:
        if ex.errno == errno.EEXIST:
            pass
        else:
            raise


def update_existing_keys(target, modifications):
    '''Modifies target in-place.'''
    new_keys = set(modifications.keys()) - set(target.keys())
    if new_keys:
        raise ValueError('new keys: {}'.format(list(new_keys)))
    target.update(modifications)
    return target


def partial_apply_kwargs(func):
    return functools.partial(apply_kwargs, func)


def apply_kwargs(func, kwargs):
    return func(**kwargs)


def get_unique_value(elems):
    '''Maps element or collection of repeated elements to single element.

    Intended as inverse of n_positive_integers().
    '''
    try:
        iterator = iter(elems)
    except TypeError as ex:
        return elems
    values = set(iterator)
    value, = values
    return value


def assert_no_keys_in_common(a, b):
    intersection = set(a.keys()).intersection(set(b.keys()))
    if intersection:
        raise ValueError('keys in common: {}'.format(str(intersection)))
