from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import tempfile

from seqtrack import helpers


class MapContext(object):

    def __init__(self, name):
        self.name = name

    def tmp_dir(self):
        '''Returns None to mean that a tmp_dir must be created if needed.'''
        return tempfile.mkdtemp()


def map(func, items):
    '''
    Args:
        func: Maps (context, input_value) to output_value.
        items: Iterable collection of (key, input_value) pairs.

    Returns:
        Yields (key, output_value) pairs.
    '''
    return ((k, func(MapContext(k), v)) for k, v in items)


def ignore_context(func):
    '''
    Args:
        func: Maps input value to output value.

    Returns:
        Func that map (context, input value) to output value.
    '''
    return functools.partial(call_ignore_context, func)


def call_ignore_context(func, context, x):
    return func(x)


# def map_dict_list(func, items):
#     return list(map_dict(func, items))


# def filter_dict(func, items):
#     for k, v in items:
#         if func(k, v):
#             yield k, v


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
    codec = helpers.CODECS[codec_name]
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    for k, v in items:
        fname = os.path.join(dir, k + codec.ext)
        with helpers.open_atomic_write(fname, 'wb' if codec.binary else 'w') as f:
            codec.dump(v, f)
        yield k, v


class CacheFilter(object):
    '''Keeps only items that do not have a cached value.

    Items that do have a cache are loaded and stored in results.
    '''

    def __init__(self, dir, codec_name='json'):
        self.results = []
        self._dir = dir
        self._codec_name = codec_name

    def filter(self, items):
        codec = helpers.CODECS[self._codec_name]
        for k, v in items:
            fname = os.path.join(self._dir, k + codec.ext)
            if os.path.exists(fname):
                with open(fname, 'rb' if codec.binary else 'r') as f:
                    result = codec.load(f)
                self.results.append((k, result))
            else:
                yield k, v
