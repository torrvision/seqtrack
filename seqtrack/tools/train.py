import argparse
import functools
import itertools
import json
import msgpack
import numpy as np
import os

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
from seqtrack.tools import train_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # Map stream of named vectors to stream of named results (order may be different).
    seeds = {'seed_{}'.format(seed): seed for seed in range(args.num_trials)}
    if args.slurm:
        mapper = slurm.SlurmDictMapper(tempdir='tmp', opts=['--' + x for x in args.slurm_flags])
    else:
        mapper = helpers.map_dict
    # Cache the results and use SLURM mapper to evaluate those without cache.
    mapper = helpers.CachedDictMapper(dir=os.path.join('cache', 'trials'),
                                      codec=msgpack, ext='.msgpack', mapper=mapper)
    result_stream = mapper(functools.partial(work._train, args), seeds.items())
    # Construct dictionary from stream.
    results = dict(result_stream)

    names = sorted(seeds.keys())
    for name in names:
        print '-' * 40
        print 'name:', name
        print 'result:', results[name]

    summary = train.summarize_trials(results.values(), val_dataset=args.optimize_dataset,
                                     sort_key=lambda metrics: metrics[args.optimize_metric])

    print
    print 'summary:', summary


def parse_arguments():
    parser = argparse.ArgumentParser()

    app.add_setup_data_args(parser)
    app.add_tracker_config_args(parser)
    app.add_eval_args(parser)
    app.add_parallel_args(parser)

    parser.add_argument('--loglevel', default='info', help='debug, info, warning')
    parser.add_argument('--verbose_train', action='store_true')

    parser.add_argument('--no_slurm', dest='slurm', action='store_false',
                        help='Submit jobs to slurm or run directly?')
    parser.add_argument('--slurm_flags', nargs='+', help='flags for sbatch (without "--")')

    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of repetitions')
    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='TRE_3_iou_seq_mean',
                        help='metric to optimize for')

    parser.add_argument('--resume', action='store_true')

    app.add_instance_arguments(parser)

    app.add_train_args(parser)
    # parser.add_argument('--train_dataset', type=json.loads, default='"ilsvrc_train"',
    #                     help='JSON to specify the training distribution')
    # parser.add_argument('--val_dataset', type=json.loads, default='"ilsvrc_val"',
    #                     help='JSON to specify the validation distribution')
    # parser.add_argument('--sampler_params', type=json.loads,
    #                     default={'kind': 'regular', 'freq': 10},
    #                     help='JSON to specify frame sampler')
    # parser.add_argument('--num_steps', type=int, default=200000,
    #                     help='number of gradient steps')

    parser.add_argument('--model_params', type=json.loads, default={},
                        help='JSON string specifying model')

    return parser.parse_args()


if __name__ == '__main__':
    main()
