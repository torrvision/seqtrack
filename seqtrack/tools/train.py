from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import errno
import functools
import itertools
import json
import numpy as np
import os
import pprint
import tensorflow as tf
nest = tf.contrib.framework.nest

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import slurm
from seqtrack import train


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict(make_kwargs(args, seed) for seed in range(args.num_trials))
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    # Dump time-series for training and tracking for each seed.
    for name in kwargs.keys():
        report_dir = os.path.join('reports', 'trials', name)
        helpers.mkdir_p(report_dir, 0o755)
        with open(os.path.join(report_dir, 'train_series.csv'), 'w') as f:
            helpers.dump_csv(f, flatten_dicts(results[name]['train_series']))
        with open(os.path.join(report_dir, 'track_series.csv'), 'w') as f:
            helpers.dump_csv(f, flatten_dicts(results[name]['track_series']))

    # Find best checkpoint for each seed and write metrics to file.
    best = {}
    for name in sorted(kwargs.keys()):
        best[name] = find_max(
            results[name]['track_series'],
            lambda x: x[args.optimize_dataset][args.optimize_metric])
        print('max for "{:s}" at {:d} ({:.3g})'.format(
            name, best[name]['arg'], best[name][args.optimize_dataset][args.optimize_metric]))
    report_dir = os.path.join('reports')
    helpers.mkdir_p(report_dir, 0o755)
    with open(os.path.join(report_dir, 'best.csv'), 'w') as f:
        helpers.dump_csv(f, flatten_dicts(best))

    # Compute statistics across trials.
    summary = train.summarize_trials(results.values(), val_dataset=args.optimize_dataset,
                                     sort_key=lambda metrics: metrics[args.optimize_metric])
    print()
    print('statistics over all trials:')
    pprint.pprint(summary)


def parse_arguments():
    parser = argparse.ArgumentParser()

    app.add_setup_data_args(parser)
    app.add_instance_arguments(parser)
    app.add_train_args(parser)
    app.add_tracker_config_args(parser)
    app.add_eval_args(parser)
    app.add_slurm_args(parser)

    parser.add_argument('--loglevel', default='info', help='debug, info, warning')
    parser.add_argument('--verbose_train', action='store_true')

    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of repetitions')
    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='TRE_3_iou_seq_mean',
                        help='metric to optimize for')

    parser.add_argument('--model_params', type=json.loads, default={},
                        help='JSON string specifying model')
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


def make_mapper(args):
    if args.slurm:
        mapper = slurm.SlurmDictMapper(tempdir='tmp',
                                       max_submit=args.slurm_max_submit,
                                       opts=['--' + x for x in args.slurm_flags])
    else:
        mapper = helpers.map_dict
    # Cache the results and use SLURM mapper to evaluate those without cache.
    mapper = helpers.CachedDictMapper(dir=os.path.join('cache', 'train'),
                                      codec_name='msgpack', mapper=mapper)
    return mapper


def make_kwargs(args, seed):
    name = 'seed_{}'.format(seed)
    kwargs = app.train_kwargs(args, name)
    kwargs.update(
        seed=seed,
        model_params=args.model_params,
        resume=args.resume,
    )
    return name, kwargs


def flatten_dicts(series):
    return {k: dict(nest.flatten_with_joined_string_paths(v)) for k, v in series.items()}


def find_max(series, key):
    keys = list(series.keys())
    arg = max(keys, key=lambda x: key(series[x]))
    # TODO: Use OrderedDict and put 'arg' first?
    result = dict(series[arg])
    assert 'arg' not in result
    result['arg'] = arg
    return result


if __name__ == '__main__':
    main()
