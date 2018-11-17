'''
Compare convergence with different xcorr settings.
(Mean, batch-norm, learned weights.)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import slurm
from seqtrack import train

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # feature_init_methods = [
    #     ('xavier', ),
    #     ('kaiming', ),
    # ]

    join_configs = [
        ('sum', dict(arch='xcorr', params=dict())),
        ('mean', dict(arch='xcorr', params=dict(use_mean=True))),
        ('bnorm_sum', dict(arch='xcorr', params=dict(use_batch_norm=True))),
        # This should be about the same as above:
        ('bnorm_mean', dict(arch='xcorr', params=dict(
            use_mean=True,
            use_batch_norm=True))),
        ('sum_weights', dict(arch='xcorr', params=dict(
            learn_spatial_weight=True))),
        # ('mean_weights_init_ones', dict(arch='xcorr', params=dict(
        #     learn_spatial_weight=True,
        #     use_mean=True))),
        ('mean_weights', dict(arch='xcorr', params=dict(
            learn_spatial_weight=True,
            weight_init_method='mean',
            use_mean=True))),
        ('bnorm_sum_weights', dict(arch='xcorr', params=dict(
            learn_spatial_weight=True,
            use_batch_norm=True))),
        ('bnorm_mean_weights', dict(arch='xcorr', params=dict(
            learn_spatial_weight=True,
            weight_init_method='mean',
            use_mean=True,
            use_batch_norm=True))),
    ]

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args, seed=seed, join=join, join_config=join_config)
        for seed in range(args.num_trials)
        for join, join_config in join_configs
    ])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    for metric_name in args.train_metrics:
        fig, ax = plt.subplots()
        plt.yscale('log')
        plt.xlabel('Gradient steps (k)')
        plt.ylabel(metric_name)
        for i, (join, _) in enumerate(join_configs):
            name_fn = lambda seed: make_name(seed=seed, join=join)
            steps = sorted(results[name_fn(seed=0)]['train_series'].keys())
            for subset in ['train', 'val']:
                # Take mean across trials.
                quality = np.mean([[
                    results[name_fn(seed)]['train_series'][t][subset + '/' + metric_name]
                    for t in steps] for seed in range(args.num_trials)], axis=0)
                # TODO: Shade colors by learning rate?
                # TODO: Add variance across trials?
                plt.plot(np.asfarray(steps) / 1000, quality, color=COLORS[i],
                         linestyle=('solid' if subset == 'train' else 'dashed'),
                         label=(join if subset == 'train' else None))
        ax.legend()
        plt.savefig('plot_train_{}.pdf'.format(metric_name))


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
    parser.add_argument('--track_metrics', nargs='+', default=['TRE_3_iou_seq_mean'])
    parser.add_argument('--train_metrics', nargs='+', default=['loss', 'dist', 'iou'])

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


def make_kwargs(args, seed, join, join_config):
    name = make_name(seed=seed, join=join)
    kwargs = app.train_kwargs(args, name)
    # Note: Overrides the learning rate specified by args.
    kwargs.update(
        seed=seed,
        model_params=dict(
            use_desired_size=True,
            target_size=64,
            desired_template_scale=2.0,
            desired_search_radius=1.0,
            feature_arch='alexnet',
            feature_arch_params=None,
            join_type='single',
            join_arch=join_config['arch'],
            join_params=join_config['params'],
            window_params=dict(
                normalize_method='mean',
                window_profile='hann',
                combine_method='mul',
            ),
            window_radius=1.0,
            arg_max_eps=0.01,
            # TODO: Study weight decay and loss config.
            wd=1e-4,
            loss_params=dict(
                method='sigmoid',
                params=dict(
                    balanced=True,
                    pos_weight=1,
                    label_method='hard',
                    label_params=dict(positive_radius=0.3, negative_radius=0.3),
                ),
            ),
        ),
    )
    return name, kwargs


def make_name(seed=None, **kwargs):
    return '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())])


if __name__ == '__main__':
    main()
