from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import hashlib
import itertools
import json
import math
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import helpers


def make_sample(vector, kwargs, metrics):
    '''Describes an evaluation.

    Store kwargs for compatibility with different parameterizations.
    '''
    return dict(vector=vector, kwargs=kwargs, metrics=metrics)


def search(vectors, func, results_dir, to_kwargs=None):
    '''Evaluates the function at each vector and saves results to results_dir.

    Evaluates func(**to_kwargs(vector)).

    Args:
        func: Function that accepts kwargs and returns a dict.

    Returns:
        Dictionary of evaluated samples.
    '''
    samples = {}
    for vector in vectors:
        # TODO: Should name be derived from vector or kwargs?
        name = _hash(vector)
        if to_kwargs is not None:
            kwargs = to_kwargs(vector)
        else:
            kwargs = vector
        # TODO: Use parallel, cached dict-mapper here?
        samples[name] = helpers.cache(
            os.path.join(results_dir, name + '.json'),
            lambda: make_sample(vector, kwargs, metrics=func(**kwargs)))


def load(results_dir):
    files = os.listdir(results_dir)
    json_files = [x for x in files if x.endswith('.json')]
    samples = {}
    for basename in json_files:
        name, _ = os.path.splitext(basename)
        try:
            with open(os.path.join(results_dir, basename), 'r') as f:
                samples[name] = json.load(f)
        except (IOError, ValueError) as ex:
            logger.warning('could not load results from "{}": {}'.format(
                os.path.join(results_dir, basename), str(ex)))
    return samples


def write_summary(f, vectors, outputs):
    names = outputs.keys()
    keys_vector = set(key for name in names for key in vectors[name].keys())
    keys_output = set(key for name in names for key in outputs[name].keys())
    keys = list(itertools.chain(
        ['name'],
        ['vector_' + key for key in sorted(keys_vector)],
        ['output_' + key for key in sorted(keys_output)]))

    writer = csv.DictWriter(f, keys)
    for name in names:
        record = dict(itertools.chain(
            [('name', name)],
            [('vector_' + key, val) for key, val in vectors[name].items()],
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
