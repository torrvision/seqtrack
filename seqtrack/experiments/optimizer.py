'''
The goal is to determine the best optimizer configuration.

There is some dependence between the optimizer and the loss.
However, it is too intensive to do this for all possible loss configurations.
Maybe we just do it once for each loss type (cross entropy, max margin, ...).
Hopefully the optimizer will work well for other loss configurations.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import argparse
import collections
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

    opt_configs = [
        ('sgd', dict(optimizer='sgd', optimizer_params=None)),
        ('momentum_0.9', dict(
            optimizer='momentum',
            optimizer_params=dict(momentum=0.9))),
        ('momentum_0.9_nesterov', dict(
            optimizer='momentum',
            optimizer_params=dict(momentum=0.9, use_nesterov=True))),
        ('rmsprop', dict(optimizer='rmsprop', optimizer_params=None)),
        ('adam', dict(optimizer='adam', optimizer_params=None)),
        ('adam_eps_1e-4', dict(
            optimizer='adam',
            optimizer_params=dict(epsilon=1e-4))),
    ]
    schedule_configs = [
        ('constant', dict(lr_schedule='constant', lr_params=None)),
        ('decay_0.1_remain_0.5', dict(
            lr_schedule='remain',
            lr_params=dict(decay_rate=0.1, remain_rate=0.5, max_power=3))),
    ]
    inits = [1e-2, 1e-3, 1e-4]
    # TODO: Add batch size!

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args, seed=seed,
                    opt=opt, opt_config=opt_config,
                    schedule=schedule, schedule_config=schedule_config,
                    init=init)
        for seed in range(args.num_trials)
        for opt, opt_config in opt_configs
        for schedule, schedule_config in schedule_configs
        for init in inits
    ])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    for metric_name in args.train_metrics:
        # Plot all curves. Use one color per optimizer.
        fig, ax = plt.subplots()
        plt.xlabel('Gradient steps (k)')
        plt.ylabel(metric_name)
        for opt_index, (opt, _) in enumerate(opt_configs):
            done_label = False
            for (init_index, init), schedule in (
                    ((init_index, init), schedule)
                    for init_index, init in enumerate(inits)
                    for _, (schedule, _) in enumerate(schedule_configs)):
                name_fn = lambda seed: make_name(seed=seed, opt=opt, schedule=schedule, init=init)
                steps = sorted(results[name_fn(seed=0)]['train_series'].keys())
                for subset in ['train', 'val']:
                    # Take mean across trials.
                    quality = np.mean([[
                        results[name_fn(seed)]['train_series'][t][subset + '/' + metric_name]
                        for t in steps] for seed in range(args.num_trials)], axis=0)
                    plt.plot(np.asfarray(steps) / 1000, quality,
                             color=COLORS[opt_index],
                             linestyle=('solid' if subset == 'train' else 'dotted'),
                             label=(opt if not done_label else None))
                    done_label = True
        ax.legend()
        plt.savefig('plot_train_{}.pdf'.format(metric_name))
        plt.close()

    for dataset in args.eval_datasets:
        for metric_name in args.track_metrics:
            fig, ax = plt.subplots()
            plt.xlabel('Gradient steps (k)')
            plt.ylabel(metric_name)
            for opt_index, (opt, _) in enumerate(opt_configs):
                done_label = False
                for (init_index, init), schedule in (
                        ((init_index, init), schedule)
                        for init_index, init in enumerate(inits)
                        for _, (schedule, _) in enumerate(schedule_configs)):
                    name_fn = lambda seed: make_name(seed=seed, opt=opt, schedule=schedule, init=init)
                    steps = sorted(results[name_fn(seed=0)]['track_series'].keys())
                    steps_k = np.asfarray(steps) / 1000

                    quality = np.mean([[
                        results[name_fn(seed)]['track_series'][t][dataset + '-full'][metric_name]
                        for t in steps] for seed in range(args.num_trials)], axis=0)
                    variance_name = metric_name + '_var'
                    variance_test = np.mean([[
                        results[name_fn(seed)]['track_series'][t][dataset + '-full'][variance_name]
                        for t in steps] for seed in range(args.num_trials)], axis=0)
                    variance_train = np.var([[
                        results[name_fn(seed)]['track_series'][t][dataset + '-full'][metric_name]
                        for t in steps] for seed in range(args.num_trials)], axis=0)
                    variance = variance_train + variance_test
                    error = np.sqrt(variance)

                    plt.fill_between(steps_k, quality - error, quality + error,
                                     color=COLORS[opt_index], label=None, alpha=0.2)
                    plt.plot(steps_k, quality, color=COLORS[opt_index],
                             label=(opt if not done_label else None))
                    done_label = True
            ax.legend()
            plt.savefig('plot_track_{}_{}.pdf'.format(dataset, metric_name))


    # Plot different learning-rate schedules for each optimizer.
    for opt, _ in opt_configs:
        for metric_name in args.train_metrics:
            fig, ax = plt.subplots()
            plt.xlabel('Gradient steps (k)')
            plt.ylabel(metric_name)
            for (init_index, init), (schedule_index, schedule) in (
                    ((init_index, init), (schedule_index, schedule))
                    for schedule_index, (schedule, _) in enumerate(schedule_configs)
                    for init_index, init in enumerate(inits)):
                name_fn = lambda seed: make_name(seed=seed, opt=opt, schedule=schedule, init=init)
                steps = sorted(results[name_fn(seed=0)]['train_series'].keys())
                for subset in ['train', 'val']:
                    # Take mean across trials.
                    quality = np.mean([[
                        results[name_fn(seed)]['train_series'][t][subset + '/' + metric_name]
                        for t in steps] for seed in range(args.num_trials)], axis=0)
                    label = '{} {}'.format(init, schedule)
                    plt.plot(np.asfarray(steps) / 1000, quality,
                             color=make_color(schedule_index, init_index / (len(inits) - 1)),
                             linestyle=('solid' if subset == 'train' else 'dotted'),
                             label=(label if subset == 'train' else None))
            ax.legend()
            plt.savefig('plot_optimizer_{}_train_{}.pdf'.format(opt, metric_name))
            plt.close()


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

    default_loss_params = dict(
        method='sigmoid',
        params=dict(
            balanced=True,
            pos_weight=1,
            label_method='hard',
            label_params=dict(positive_radius=0.3, negative_radius=0.3)))
    parser.add_argument('--loss_params', type=json.loads, help='Loss function',
                        default=json.dumps(default_loss_params))

    parser.add_argument('-n', '--num_trials', type=int, default=1,
                        help='number of repetitions')
    parser.add_argument('--track_metrics', nargs='+',
                        default=['OPE_iou_seq_mean', 'TRE_3_iou_seq_mean'])
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


def make_kwargs(args, seed, opt, opt_config, schedule, schedule_config, init):
    name = make_name(seed=seed, opt=opt, schedule=schedule, init=init)
    kwargs = app.train_kwargs(args, name)
    # Note: Overrides the optimizer specified by args.
    kwargs.update(opt_config)
    # Note: Overrides the learning rate specified by args.
    kwargs.update(lr_init=init, **schedule_config)
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
            join_arch='xcorr',
            join_params=dict(use_batch_norm=True),
            window_params=dict(
                normalize_method='mean',
                window_profile='hann',
                combine_method='mul'),
            window_radius=1.0,
            arg_max_eps=0.01,
            # TODO: Study weight decay and loss config.
            wd=1e-4,
            loss_params=args.loss_params,
        ),
    )
    return name, kwargs


def make_name(seed=None, **kwargs):
    return '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())])


def make_color(i, degree):
    degree = float(degree)
    base = colors.rgb_to_hsv(colors.to_rgb(COLORS[i]))
    # TODO: This is not ideal because it does not include the original color.
    def interpolate(a, b, p):
        return (1 - p) * a + p * b
    # The color `low` is darker. Decrease `value`.
    low = np.array(base)
    low[2] = 0.7 * low[2]
    # The color `high` is lighter. Decrease saturation.
    high = np.array(base)
    high[1] = 0.7 * high[1]
    return colors.hsv_to_rgb(interpolate(low, high, degree))


if __name__ == '__main__':
    main()
