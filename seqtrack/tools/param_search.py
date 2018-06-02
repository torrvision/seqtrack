from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from seqtrack.tools import param_search_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # Make infinite stream of (named) vectors.
    vector_stream = _make_vector_stream(np.random.RandomState(0))
    vectors = dict(itertools.islice(vector_stream, args.num_configs))
    # Map stream of named vectors to stream of named results (order may be different).
    if args.slurm:
        mapper = slurm.SlurmDictMapper(tempdir='tmp', opts=['--' + x for x in args.slurm_flags])
    else:
        mapper = helpers.map_dict
    # Cache the results and use SLURM mapper to evaluate those without cache.
    mapper = helpers.CachedDictMapper(dir=os.path.join('cache', 'trials'),
                                      codec=msgpack, ext='.msgpack', mapper=mapper)
    result_stream = mapper(functools.partial(work._train, args), vectors.items())
    # Construct dictionary from stream.
    results = dict(result_stream)

    # Print a table of vectors and results.
    names = sorted(vectors.keys())
    for name in names:
        print('-' * 40)
        print('name:', name)
        print('vector:', vectors[name])
        print('result:', results[name])

    summaries = {
        name: train.summarize_trials([results], val_dataset=args.optimize_dataset,
                                     sort_key=lambda metrics: metrics[args.optimize_metric])
        for name, results in results.items()}

    # Print a table of vectors and results.
    print()
    for name in names:
        print('-' * 40)
        print('name:', name)
        print('summary:', summaries[name])
        # search.write_summary(summaries[name])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_configs', type=int, default=10,
                        help='number of configurations')

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


if __name__ == '__main__':
    main()
