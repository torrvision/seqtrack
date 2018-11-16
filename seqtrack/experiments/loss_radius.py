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


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    radii = [0.1, 0.3]  # [0.05, 0.1, 0.2, 0.3, 0.5]

    # Parameterize losses.
    sigmoid_pos_weights = [1.0]  # [0.1, 1.0, 10.0]
    # Given radius, return label_params dict.
    labels = [('hard_binary', lambda r: dict(radius=r)),
              ('gaussian', lambda r: dict(sigma=r))]
    # Given radius return `params` dict for siamfc.compute_loss.
    losses = list(itertools.chain(
        [('sigmoid_{}_{}'.format(pos_weight, label_name), functools.partial(
             lambda label_name, label_fn, r: dict(
                 method='sigmoid',
                 params=dict(balanced=True,
                             pos_weight=pos_weight,
                             label_method=label_name,
                             label_params=label_fn(r))),
             label_name, label_fn))
         for pos_weight in sigmoid_pos_weights
         for label_name, label_fn in labels],
        [('softmax_{}'.format(label_name), functools.partial(
             lambda label_name, label_fn, r: dict(
                 method='softmax',
                 params=dict(label_method=label_name,
                             label_params=label_fn(r))),
             label_name, label_fn))
         for label_name, label_fn in labels],
        [('max_margin_dist', lambda r: dict(
            method='max_margin', params=dict(cost_method='distance_greater',
                                             cost_params=dict(threshold=r))))],
    ))

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args, seed=seed, loss=loss, loss_params_fn=loss_params_fn, radius=radius)
        for seed in range(args.num_trials)
        for loss, loss_params_fn in losses
        for radius in radii])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    # # To obtain one number per configuration, we use one dataset as validation.
    # summaries = {}
    # for balanced in balanced_range:
    #     for pos_weight in pos_weight_range:
    #         for pos_radius, neg_radius in pos_neg_radius_range:
    #             summary_name = make_name(balanced=balanced, pos_weight=pos_weight,
    #                                      pos_radius=pos_radius, neg_radius=neg_radius)
    #             trial_names = [make_name(balanced=balanced, pos_weight=pos_weight,
    #                                      pos_radius=pos_radius, neg_radius=neg_radius, seed=seed)
    #                            for seed in range(args.num_trials)]
    #             summaries[summary_name] = train.summarize_trials(
    #                 [results[name]['track_series'] for name in trial_names],
    #                 val_dataset=args.optimize_dataset,
    #                 sort_key=lambda metrics: metrics[args.optimize_metric])

    # quality_metric = args.optimize_dataset + '_' + args.optimize_metric
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # # https://matplotlib.org/api/markers_api.html
    # markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd']

    # for balanced in balanced_range:
    #     fig, ax = plt.subplots()
    #     helpers.set_xscale_log(ax)
    #     plt.xlabel('negative_radius')
    #     plt.ylabel(quality_metric)

    #     # Join points of equal positive and negative radius.
    #     for j, pos_weight in enumerate(pos_weight_range):
    #         name_fn = lambda pos_radius, neg_radius: make_name(
    #             balanced=balanced, pos_weight=pos_weight,
    #             pos_radius=pos_radius, neg_radius=neg_radius)
    #         quality = [summaries[name_fn(radius, radius)][quality_metric]
    #                    for radius in radius_range]
    #         plt.plot(radius_range, quality, label=None, color='black', linestyle='dotted')

    #     for i, pos_radius in enumerate(radius_range):
    #         for j, pos_weight in enumerate(pos_weight_range):
    #             name_fn = lambda neg_radius: make_name(
    #                 balanced=balanced, pos_weight=pos_weight,
    #                 pos_radius=pos_radius, neg_radius=neg_radius)
    #             # Consider negative radius >= positive radius.
    #             neg_radii = radius_range[i:]
    #             quality = [summaries[name_fn(neg_radius)][quality_metric]
    #                        for neg_radius in neg_radii]
    #             variance = [summaries[name_fn(neg_radius)].get(quality_metric + '_var', np.nan)
    #                         for neg_radius in neg_radii]
    #             error = 1.64485 * np.sqrt(variance)
    #             plt.fill_between(x=neg_radii, y1=quality - error, y2=quality + error,
    #                              color=colors[i], label=None, alpha=0.2)
    #             plt.plot(neg_radii, quality,
    #                      label='pos_radius {}, pos_weight {}'.format(pos_radius, pos_weight),
    #                      color=colors[i], marker=markers[j])

    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, 0.55 * box.width, box.height])
    #     ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    #     plt.savefig('plot_balanced_{}.pdf'.format(balanced))


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