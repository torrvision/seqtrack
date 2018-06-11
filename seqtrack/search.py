from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future_builtins import map
from future_builtins import filter

import csv
import hashlib
import itertools
import json
import math
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers


# class FunctionInterface(object):
#     '''Describes the function interface.
# 
#     The important thing is to provide the args and kwargs of each input vector.
#     '''
# 
#     def sample_input_vectors(self):
#         pass
# 
#     def input_from_vector(self, input_vec):
#         args = []
#         kwargs = {}
#         return args, kwargs
# 
#     def output_to_vector(self, output):
#         output_vec = {}
#         return output_vec


def main(func, input_stream, kwargs_fn=None, postproc_fn=None,
         use_existing_inputs=True, max_num_configs=None, report_only=False,
         cache_dir='cache', input_codec='json', kwargs_codec='json', output_codec='json',
         use_slurm=True, slurm_flags=None):
    '''Evaluates an expensive function on named inputs and saves the outputs.

    Args:
        func: Function that maps (name=name, **kwargs) to output.
        kwargs_fn: Maps flat dict from sampler to kwargs for func().
            If none then identity function is used.
            The result of kwargs_fn() will be cached in cache_dir/kwargs.
        postproc_fn: Maps result of func() to flat dict of scalar values.
            If none then identity function is used.
            The result of func() is cached but not the result of postproc_fn().
        input_stream: Collection of dicts with fields 'vector', 'args', 'kwargs'.
    '''
    slurm_flags = slurm_flags or []

    if report_only:
        inputs = discover_dict(os.path.join(cache_dir, 'inputs'), codec_name=input_codec)
        outputs = discover_dict(os.path.join(cache_dir, 'outputs'), codec_name=output_codec)
        if len(outputs) == 0:
            raise RuntimeError('zero outputs found')
        logger.info('found %d outputs', len(outputs))
        output_stream = outputs.items()
    else:
        if use_existing_inputs:
            existing_inputs = discover_dict(os.path.join(cache_dir, 'inputs'), codec_name=input_codec)
            input_stream = helpers.filter_dict(lambda k, v: k not in existing_inputs, input_stream)
            # Concatenate existing and new inputs.
            input_stream = itertools.chain(existing_inputs.items(), input_stream)
        # Take first n inputs.
        if max_num_configs and max_num_configs > 0:
            input_stream = itertools.islice(input_stream, max_num_configs)
        inputs = dict(input_stream)
        input_stream = inputs.items()
        # Write inputs as they are accessed.
        input_stream = dump_dict_stream(input_stream, dir=os.path.join(cache_dir, 'inputs'),
                                        codec_name=input_codec)

        # Map inputs to kwargs and save kwargs to file.
        # If kwargs have already been generated and written, use existing result.
        kwargs_mapper = helpers.CachedDictMapper(dir=os.path.join(cache_dir, 'kwargs'),
                                                 codec_name=kwargs_codec)
        kwargs_fn = kwargs_fn or _identity
        kwargs_stream = kwargs_mapper(functools.partial(_apply_value, kwargs_fn), input_stream)

        # Map stream of named kwargs to stream of named outputs (order may be different).
        if use_slurm:
            func_mapper = slurm.SlurmDictMapper(tempdir='tmp',
                                                opts=['--' + x for x in slurm_flags])
        else:
            func_mapper = helpers.map_dict
        # Cache the outputs and use slurm mapper to evaluate those without cache.
        func_mapper = helpers.CachedDictMapper(dir=os.path.join(cache_dir, 'outputs'),
                                               codec_name=output_codec, mapper=func_mapper)
        output_stream = func_mapper(functools.partial(_apply_kwargs, func), kwargs_stream)

    if postproc_fn:
        output_stream = helpers.map_dict(functools.partial(_apply_value, postproc_fn),
                                         output_stream)
    # Construct dictionary from stream.
    outputs = dict(output_stream)

    logger.info('write report to report.csv')
    with open('report.csv', 'w') as f:
        write_summary(f, inputs, outputs)


def _apply_value(func, key, value):
    return func(value)


def _identity(x):
    return x


def _apply_kwargs(func, name, kwargs):
    return func(name=name, **kwargs)


def _update_default_kwargs(default, kwargs_fn, key, vector):
    kwargs = {}
    if default is not None:
        kwargs.update(default)
    if kwargs_fn is not None:
        override = kwargs_fn(vector)
        kwargs.update(override)
    return kwargs


def dump_dict_stream(items, dir, codec_name):
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    codec = helpers.CODECS[codec_name]
    for key, value in items:
        fname = os.path.join(dir, key + codec.ext)
        with helpers.open_atomic_write(fname, 'wb' if codec.binary else 'w') as f:
            codec.module.dump(value, f)
        yield key, value


def discover_dict(dir, codec_name):
    codec = helpers.CODECS[codec_name]

    names = os.listdir(dir) if os.path.isdir(dir) else []
    names = [name for name in names if name.endswith(codec.ext)]
    names = [name for name in names if os.path.isfile(os.path.join(dir, name))]

    x = {}
    for name in names:
        key = name[:-len(codec.ext)]
        with open(os.path.join(dir, name), 'rb' if codec.binary else 'r') as f:
            value = codec.module.load(f)
        x[key] = value
    return x


# def make_sample(vector, kwargs, metrics):
#     '''Describes an evaluation.
# 
#     Store kwargs for compatibility with different parameterizations.
#     '''
#     return dict(vector=vector, kwargs=kwargs, metrics=metrics)


# def search(vectors, func, results_dir, to_kwargs=None):
#     '''Evaluates the function at each vector and saves results to results_dir.
#
#     Evaluates func(**to_kwargs(vector)).
#
#     Args:
#         func: Function that accepts kwargs and returns a dict.
#
#     Returns:
#         Dictionary of evaluated samples.
#     '''
#     samples = {}
#     for vector in vectors:
#         # TODO: Should name be derived from vector or kwargs?
#         name = _hash(vector)
#         if to_kwargs is not None:
#             kwargs = to_kwargs(vector)
#         else:
#             kwargs = vector
#         # TODO: Use parallel, cached dict-mapper here?
#         samples[name] = helpers.cache(
#             os.path.join(results_dir, name + '.json'),
#             lambda: make_sample(vector, kwargs, metrics=func(**kwargs)))


# def load(results_dir):
#     files = os.listdir(results_dir)
#     json_files = [x for x in files if x.endswith('.json')]
#     samples = {}
#     for basename in json_files:
#         name, _ = os.path.splitext(basename)
#         try:
#             with open(os.path.join(results_dir, basename), 'r') as f:
#                 samples[name] = json.load(f)
#         except (IOError, ValueError) as ex:
#             logger.warning('could not load results from "{}": {}'.format(
#                 os.path.join(results_dir, basename), str(ex)))
#     return samples


def write_summary(f, inputs, outputs):
    names = outputs.keys()
    keys_input = set(key for name in names for key in inputs[name].keys())
    keys_output = set(key for name in names for key in outputs[name].keys())
    keys = list(itertools.chain(
        ['name'],
        ['input_' + key for key in sorted(keys_input)],
        ['output_' + key for key in sorted(keys_output)]))

    writer = csv.DictWriter(f, keys)
    writer.writeheader()
    for name in names:
        record = dict(itertools.chain(
            [('name', name)],
            [('input_' + key, val) for key, val in inputs[name].items()],
            [('output_' + key, val) for key, val in outputs[name].items()]))
        writer.writerow(record)


# Could use hyperopt for this? At least to define the space?
# However, it doesn't easily support conditionals without nesting (i.e. choice).
# (May be possible using scope.switch().)
# The parallel execution could be useful although it requires mongodb.


def sample_param(rand, method, *params):
    funcs = {
        'const': lambda rand, x: x,
        'choice': _choice,
        'uniform': lambda rand, a, b: rand.uniform(a, b),
        'log_uniform': _log_uniform,
        'one_minus_log_uniform': _one_minus_log_uniform,
        'uniform_format': _uniform_format,
        'log_uniform_format': _log_uniform_format,
        'one_minus_log_uniform_format': _one_minus_log_uniform_format,
    }
    if method not in funcs:
        raise RuntimeError('method not found: "{}"'.format(method))

    try:
        value = funcs[method](rand, *params)
    except Exception as ex:
        raise RuntimeError('sample "{}" with args {}: {}'.format(
            method, repr(params), str(ex)))
    return value


def _choice(rand, items):
    # The function rand.choice() performs some sort of type coercion.
    # rand.choice([True, False]) returns type np.bool_ not bool.
    i = rand.choice(len(items))
    return items[i]


def _log_uniform(rand, a, b):
    return math.exp(rand.uniform(math.log(a), math.log(b)))


def _one_minus_log_uniform(rand, a, b):
    return 1 - _log_uniform(rand, 1 - b, 1 - a)


def _uniform_format(rand, a, b, spec):
    return _round(rand.uniform(a, b), spec)


def _log_uniform_format(rand, a, b, spec):
    return _round(_log_uniform(rand, a, b), spec)


def _one_minus_log_uniform_format(rand, a, b, spec):
    x = 1 - _log_uniform_format(rand, 1 - b, 1 - a, spec)
    # Round again to avoid e.g. 1 - 0.941 == 0.05900000000000005
    return round(x, 10)


def _round(x, spec):
    return float(format(x, spec))


def hash_vector(obj):
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]
