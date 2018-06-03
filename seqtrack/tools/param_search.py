from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from future_builtins import filter

import argparse
import functools
import itertools
import json
import msgpack
import numpy as np
import os
import pprint

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import cnnutil
from seqtrack import helpers
from seqtrack import search
from seqtrack import slurm
from seqtrack import train
from seqtrack.models import util

# The pickled object must be imported to unpickle in a different package (slurmproc.worker).
from seqtrack.tools import param_search_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    if args.report_only:
        vectors = _discover_dict(os.path.join('cache', 'vectors'), codec=json, ext='.json')
        results = _discover_dict(os.path.join('cache', 'results'), codec=msgpack, ext='.msgpack')
        if len(results) == 0:
            raise RuntimeError('zero results found')
        logger.info('found %d results', len(results))
        _report(args, vectors, results)
        return

    vectors_existing = _discover_dict(os.path.join('cache', 'vectors'), codec=json, ext='.json')
    # Make stream of (named) vectors.
    vector_stream = _make_vector_stream(np.random.RandomState(0))
    vector_stream = filter(lambda pair: pair[0] not in vectors_existing, vector_stream)
    # Concatenate existing and new vectors.
    vector_stream = itertools.chain(vectors_existing.items(), vector_stream)

    # Take first n vectors.
    vectors = dict(itertools.islice(vector_stream, args.num_configs))
    vector_stream = vectors.items()

    # Write vectors as they are mapped.
    vector_stream = _dump_dict_stream(vector_stream, dir=os.path.join('cache', 'vectors'),
                                      codec=json, ext='.json')
    # Map stream of named vectors to stream of named results (order may be different).
    if args.slurm:
        mapper = slurm.SlurmDictMapper(tempdir='tmp', opts=['--' + x for x in args.slurm_flags])
    else:
        mapper = helpers.map_dict
    # Cache the results and use SLURM mapper to evaluate those without cache.
    mapper = helpers.CachedDictMapper(dir=os.path.join('cache', 'results'),
                                      codec=msgpack, ext='.msgpack', mapper=mapper)
    result_stream = mapper(functools.partial(work._train, args), vector_stream)
    # Construct dictionary from stream.
    results = dict(result_stream)

    _report(args, vectors, results)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_configs', type=int, default=10,
                        help='number of configurations')
    parser.add_argument('--report_only', action='store_true')

    parser.add_argument('--loglevel', default='info', help='debug, info, warning')
    parser.add_argument('--verbose_train', action='store_true')

    parser.add_argument('--no_slurm', dest='slurm', action='store_false',
                        help='Submit jobs to slurm or run directly?')
    parser.add_argument('--slurm_flags', nargs='+', help='flags for sbatch (without "--")')

    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='TRE_3_iou_seq_mean',
                        help='metric to optimize for')
    app.add_setup_data_args(parser)
    app.add_tracker_config_args(parser)
    app.add_eval_args(parser)
    app.add_parallel_args(parser)
    # Keep image resolution fixed across trials.
    parser.add_argument('--imwidth', type=int, default=360, help='image resolution')
    parser.add_argument('--imheight', type=int, default=360, help='image resolution')

    parser.add_argument('--train_dataset', type=json.loads, default='"ilsvrc_train"',
                        help='JSON to specify the training distribution')
    parser.add_argument('--val_dataset', type=json.loads, default='"ilsvrc_val"',
                        help='JSON to specify the validation distribution')
    parser.add_argument('--num_steps', type=int, default=200000,
                        help='number of gradient steps')

    return parser.parse_args()


def _make_vector_stream(rand):
    while True:
        vector = {}
        vector.update(work.sample_vector_train(rand, work.DEFAULT_DISTRIBUTION_TRAIN))
        vector.update(work.sample_vector_siamfc(rand, work.DEFAULT_DISTRIBUTION_SIAMFC))
        name = search.hash_vector(vector)
        yield name, vector


def _dump_dict_stream(items, dir, codec, ext):
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)
    for key, value in items:
        fname = os.path.join(dir, key + ext)
        with helpers.open_atomic_write(fname) as f:
            codec.dump(value, f)
        yield key, value


def _discover_dict(dir, codec, ext):
    names = os.listdir(dir) if os.path.isdir(dir) else []
    names = [name for name in names if name.endswith(ext)]
    names = [name for name in names if os.path.isfile(os.path.join(dir, name))]

    x = {}
    for name in names:
        key = name[:-len(ext)]
        with open(os.path.join(dir, name), 'r') as f:
            value = codec.load(f)
        x[key] = value
    return x


def _report(args, vectors, results):
    # Use optimize_dataset to choose checkpoint of each experiment.
    # If there were multiple trials, take mean over trials.
    results_best = {
        name: train.summarize_trials([results[name]], val_dataset=args.optimize_dataset,
                                     sort_key=lambda metrics: metrics[args.optimize_metric])
        for name in results.keys()}

    # TODO: Use of summary name for two things is really bad!
    logger.info('write report to report.csv')
    with open('report.csv', 'w') as f:
        search.write_summary(f, vectors, results_best)


if __name__ == '__main__':
    main()
