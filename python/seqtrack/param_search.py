# Could use hyperopt for this? At least to define the space?
# However, it doesn't easily support conditionals without nesting (i.e. choice).
# (May be possible using scope.switch().)
# The parallel execution could be useful although it requires mongodb.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import functools
import hashlib
import itertools
import json
import math
import os

import tensorflow as tf
nest = tf.contrib.framework.nest

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers
from seqtrack import mapdict
from seqtrack import slurm


def main(func, input_stream, postproc_fn=None,
         use_existing_inputs=True, max_num_configs=None, report_only=False,
         cache_dir='cache', input_codec='json', kwargs_codec='json', output_codec='json',
         use_slurm=True, slurm_kwargs=None):
         # slurm_group_size=None):
    '''Evaluates an expensive function on named inputs and saves the outputs.

    Args:
        func: Function that maps (context, kwargs) to output.
        postproc_fn: Maps result of func() to flat dict of scalar values.
            If none then identity function is used.
            The result of func() is cached but not the result of postproc_fn().
        input_stream: Collection of dicts with fields 'vector', 'args', 'kwargs'.
        report_only: Report existing results, do not perform any more function evaluations.
    '''

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

        # # Map inputs to kwargs and save kwargs to file.
        # # If kwargs have already been generated and written, use existing result.
        # kwargs_mapper = helpers.CachedDictMapper(dir=os.path.join(cache_dir, 'kwargs'),
        #                                          codec_name=kwargs_codec)
        # kwargs_fn = kwargs_fn or _identity
        # kwargs_stream = kwargs_mapper(functools.partial(_apply_value, kwargs_fn), input_stream)

        # Map stream of named kwargs to stream of named outputs (order may be different).
        if use_slurm:
            # if slurm_group_size and slurm_group_size > 1:
            #     func_mapper = slurm.SlurmDictGroupMapper(tempdir='tmp', opts=slurm_flags,
            #                                              group_size=slurm_group_size)
            # else:
            #     func_mapper = slurm.SlurmDictMapper(tempdir='tmp', opts=slurm_flags)
            func_mapper = slurm.SlurmDictMapper(**(slurm_kwargs or {}))
        else:
            func_mapper = mapdict.map
        # Cache the outputs and use slurm mapper to evaluate those without cache.
        func_mapper = mapdict.CachedDictMapper(dir=os.path.join(cache_dir, 'outputs'),
                                               codec_name=output_codec, mapper=func_mapper)
        output_stream = func_mapper(func, input_stream)

    if postproc_fn:
        output_stream = ((k, postproc_fn(v)) for k, v in output_stream)
    # Construct dictionary from stream.
    outputs = dict(output_stream)

    logger.info('write report to report.csv')
    with open('report.csv', 'w') as f:
        write_summary(f, inputs, outputs)


def _identity(x):
    return x


def apply_kwargs(func, context, kwargs):
    return func(context, **kwargs)


def dump_dict_stream(items, dir, codec_name):
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    codec = helpers.CODECS[codec_name]
    for key, value in items:
        fname = os.path.join(dir, key + codec.ext)
        with helpers.open_atomic_write(fname, 'wb' if codec.binary else 'w') as f:
            codec.dump(value, f)
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
            value = codec.load(f)
        x[key] = value
    return x


def write_summary(f, inputs, outputs):
    names = sorted(outputs.keys())
    # Flatten nested structure.
    inputs = {k: dict(nest.flatten_with_joined_string_paths(v)) for k, v in inputs.items()}
    outputs = {k: dict(nest.flatten_with_joined_string_paths(v)) for k, v in outputs.items()}
    # Take union of all fields.
    input_fields = sorted(set(itertools.chain.from_iterable(v.keys() for v in inputs.values())))
    output_fields = sorted(set(itertools.chain.from_iterable(v.keys() for v in outputs.values())))
    fields = list(itertools.chain(
        ['name'],
        ['input/' + key for key in input_fields],
        ['output/' + key for key in output_fields]))
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for name in names:
        record = dict(itertools.chain(
            [('name', name)],
            [('input/' + key, val) for key, val in inputs[name].items()],
            [('output/' + key, val) for key, val in outputs[name].items()]))
        writer.writerow(record)


def hash(obj):
    s = json.dumps(obj, sort_keys=True)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]
