'''
Varies a radius parameter in several loss functions.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import functools
import itertools
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

ERRORBAR_SIZE = 1.64485
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
# https://matplotlib.org/api/markers_api.html
MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd']


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # Given radius, return label_params dict.
    labels = [
        ('hard_binary', lambda r: dict(radius=r)),
        ('gaussian', lambda r: dict(sigma=r)),
    ]
    labels = [x for x in labels if x[0] in args.labels]

    # Given radius return `params` dict for siamfc.compute_loss.
    sigmoid_losses = [
        ('sigmoid_{}_{}'.format(pos_weight, label_name), functools.partial(
            lambda label_name, label_fn, r: dict(
                method='sigmoid',
                params=dict(balanced=True,
                            pos_weight=pos_weight,
                            label_method=label_name,
                            label_params=label_fn(r))),
            label_name, label_fn))
        for pos_weight in args.sigmoid_pos_weights
        for label_name, label_fn in labels]
    softmax_losses = [
        ('softmax_{}'.format(label_name), functools.partial(
            lambda label_name, label_fn, r: dict(
                method='softmax',
                params=dict(label_method=label_name,
                            label_params=label_fn(r))),
            label_name, label_fn))
        for label_name, label_fn in labels]
    max_margin_losses = [
        ('max_margin_dist', lambda r: dict(
            method='max_margin', params=dict(
                cost_method='distance_greater',
                cost_params=dict(threshold=r))))]
    losses = (
        (sigmoid_losses if 'sigmoid' in args.loss_families else []) +
        (softmax_losses if 'softmax' in args.loss_families else []) +
        (max_margin_losses if 'max_margin' in args.loss_families else []))

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args, seed=seed, loss=loss, loss_params_fn=loss_params_fn, radius=radius)
        for seed in range(args.num_trials)
        for loss, loss_params_fn in losses
        for radius in args.radius])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    # To obtain one number per training process, we use one dataset as validation.
    summaries = {}
    for loss, _ in losses:
        for radius in args.radius:
            summary_name = make_name(loss=loss, radius=radius)
            trial_names = [make_name(loss=loss, radius=radius, seed=seed)
                           for seed in range(args.num_trials)]
            summaries[summary_name] = train.summarize_trials(
                [results[name]['track_series'] for name in trial_names],
                val_dataset=args.optimize_dataset,
                sort_key=lambda metrics: metrics[args.optimize_metric])

    quality_metric = args.optimize_dataset + '_' + args.optimize_metric

    fig, ax = plt.subplots()
    plt.xlabel('Radius (relative to object size)')
    plt.ylabel(quality_metric)
    for loss_ind, (loss, _) in enumerate(losses):
        name_fn = lambda radius: make_name(loss=loss, radius=radius)
        quality = [summaries[name_fn(radius)][quality_metric] for radius in args.radius]
        variance = [summaries[name_fn(radius)][quality_metric + '_var'] for radius in args.radius]
        error = ERRORBAR_SIZE * np.sqrt(variance)
        plt.fill_between(x=args.radius, y1=quality - error, y2=quality + error,
                         color=COLORS[loss_ind], alpha=0.2, label=None)
        plt.plot(args.radius, quality, label=loss,
                 color=COLORS[loss_ind], marker=MARKERS[loss_ind])
    ax.legend()
    plt.savefig('plot.pdf')


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

    parser.add_argument('--radius', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3, 0.5],
                        help='Radius of loss function')
    parser.add_argument('--loss_families', nargs='+',
                        default=['sigmoid', 'softmax', 'max_margin'])
    parser.add_argument('--sigmoid_pos_weights', type=float, nargs='+', default=[0.1, 1.0])
    parser.add_argument('--labels', nargs='+', default=['hard_binary', 'gaussian'])
    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of repetitions')

    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='TRE_3_iou_seq_mean',
                        help='metric to optimize for')

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


def make_kwargs(args, seed, loss, loss_params_fn, radius):
    name = make_name(seed=seed, loss=loss, radius=radius)
    kwargs = app.train_kwargs(args, name)
    kwargs.update(
        seed=seed,
        model_params=dict(
            use_desired_size=True,
            target_size=64,
            desired_template_scale=2.0,
            desired_search_radius=1.0,
            feature_arch='alexnet',
            feature_arch_params=None,
            feature_extra_conv_enable=False,
            join_type='single',
            join_arch='xcorr',
            join_params=dict(use_batch_norm=True),
            window_params=dict(normalize_method='mean',
                               window_profile='hann',
                               combine_method='mul'),
            window_radius=1.0,
            arg_max_eps=0.01,
            wd=1e-4,
            loss_params=loss_params_fn(radius),
        ),
    )
    return name, kwargs


def make_name(seed=None, **kwargs):
    parts = [key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())]
    if seed is not None:
        parts.append('seed_' + str(seed))
    return '_'.join(parts)


if __name__ == '__main__':
    main()
