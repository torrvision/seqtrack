'''
Vary the amount of context with and without spatial weights.
Do this for several different feature architectures.
It may be good to vary the resolution of the template too;
this can be another experiment.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import argparse
import collections
import itertools
import json
import os
import pprint

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import slurm
from seqtrack import train
from seqtrack.models import siamfc

ERRORBAR_SIZE = 1.64485
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
# https://matplotlib.org/api/markers_api.html
MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd']


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    use_spatial_weights = [False, True]

    FeatureConfig = collections.namedtuple(
        'FeatureConfig',
        ['arch', 'arch_params', 'extra_conv_enable', 'extra_conv_params'])
    alexnet_configs = [
        ('alexnet_conv2', FeatureConfig(
            arch='alexnet',
            arch_params=dict(output_layer='conv2'),
            extra_conv_enable=False,
            extra_conv_params=None)),
        ('alexnet_conv3', FeatureConfig(
            arch='alexnet',
            arch_params=dict(output_layer='conv3'),
            extra_conv_enable=False,
            extra_conv_params=None)),
        ('alexnet_conv5', FeatureConfig(
            arch='alexnet',
            arch_params=dict(output_layer='conv5'),
            extra_conv_enable=False,
            extra_conv_params=None)),
    ]
    resnet_configs = [
        ('resnet_block1', FeatureConfig(
            arch='slim_resnet_v1_50',
            arch_params=dict(num_blocks=1),
            extra_conv_enable=True,
            extra_conv_params=None)),
        ('resnet_block2', FeatureConfig(
            arch='slim_resnet_v1_50',
            arch_params=dict(num_blocks=2),
            extra_conv_enable=True,
            extra_conv_params=None)),
        ('resnet_block3', FeatureConfig(
            arch='slim_resnet_v1_50',
            arch_params=dict(num_blocks=3),
            extra_conv_enable=True,
            extra_conv_params=None)),
    ]
    feat_families = [alexnet_configs, resnet_configs]

    # Obtain a set of feature dimensions to consider for each feature.
    # If there are multiple elements with the same key, only one will remain.
    # This occurs when an architecture has a minimum context (due to the receptive field).
    # For example, if the minimum possible context is 2.5
    # and the desired contexts are 1, 2, 3,
    # this will result in contexts [2.5, 2.5, 3].
    # We remove duplicates here to avoid wasting computation.
    image_dims = {
        feat: collections.OrderedDict(
            make_dimensions_pair(find_dimensions(
                target_size=args.target_size,
                desired_search_radius=args.desired_search_radius,
                feat_config=feat_config,
                context=context))
            for context in args.desired_contexts)
        for feat, feat_config in itertools.chain(*feat_families)}
    print('image_dims:')
    pprint.pprint(image_dims)

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args,
                    feat=feat,
                    feat_config=feat_config,
                    weight=weight,
                    dims=dims,
                    dims_config=dims_config,
                    seed=seed)
        for feat, feat_config in itertools.chain(*feat_families)
        for weight in use_spatial_weights
        for dims, dims_config in image_dims[feat].items()
        for seed in range(args.num_trials)
    ])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    # To obtain one number per training process, we use one dataset as validation.
    summaries = {}
    for feat, feat_config in itertools.chain(*feat_families):
        for weight in use_spatial_weights:
            for context in args.desired_contexts:
                summary_name = make_name(feat=feat, weight=weight, context=context)
                trial_names = [make_name(feat=feat, weight=weight, context=context, seed=seed)
                               for seed in range(args.num_trials)]
                summary = train.summarize_trials(
                    [results[name]['track_series'] for name in trial_names],
                    val_dataset=args.optimize_dataset,
                    sort_key=lambda metrics: metrics[args.optimize_metric])
                # Add model parameters.
                # TODO: Move into summarize_trials?
                model_properties = helpers.unique_value(
                    [results[name]['model_properties'] for name in trial_names])
                summary.update({'model/' + k: v for k, v in model_properties.items()})
                summaries[summary_name] = summary

    quality_metric = args.optimize_dataset + '_' + args.optimize_metric

    # Make one plot with weight and one plot without.
    for weight in use_spatial_weights:
        fig, ax = plt.subplots()
        plt.xlabel('template_scale')
        plt.ylabel(quality_metric)
        plt.title('learn_spatial_weight={}'.format(weight))

        i = 0
        for family_ind, feat_configs in enumerate(feat_families):
            for feat_ind, (feat, feat_config) in enumerate(feat_configs):
                color = make_color(family_ind, len(feat_families), feat_ind, len(feat_configs))
                name_fn = lambda context: make_name(feat=feat, weight=weight, context=context)
                contexts = [summaries[name_fn(context)]['model/template_scale']
                            for context in args.desired_contexts]
                quality = [summaries[name_fn(context)][quality_metric]
                           for context in args.desired_contexts]
                # variance = [summaries[name_fn(context)].get(quality_metric + '_var', np.nan)
                variance = [summaries[name_fn(context)][quality_metric + '_var']
                            for context in args.desired_contexts]
                error = ERRORBAR_SIZE * np.sqrt(variance)
                # TODO: Plot all fill_betweens then all lines?
                plt.fill_between(x=contexts, y1=quality - error, y2=quality + error,
                                 color=color, alpha=0.1, label=None)
                plt.plot(contexts, quality, label=feat, color=color, marker=MARKERS[i])
                i += 1

        ax.legend()
        plt.savefig('plot_weight_{}.pdf'.format(weight))


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
    parser.add_argument('--desired_contexts', type=float, nargs='+', default=[1.0, 2.0, 4.0])

    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--desired_search_radius', type=float, default=1)

    default_loss_params = dict(method='sigmoid',
                               params=dict(balanced=True,
                                           label_method='gaussian',
                                           label_params=dict(sigma=0.2)))
    parser.add_argument('--loss_params', type=json.loads,
                        default=json.dumps(default_loss_params))

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


def make_kwargs(args, feat, feat_config, weight, dims, dims_config, seed):
    name = make_name(feat=feat, weight=weight, dims=dims, seed=seed)
    kwargs = app.train_kwargs(args, name)
    kwargs.update(
        seed=seed,
        model_params=dict(
            target_size=args.target_size,
            use_desired_size=False,
            template_size=dims_config['template_size'],
            search_size=dims_config['search_size'],
            feature_arch=feat_config.arch,
            feature_arch_params=feat_config.arch_params,
            feature_extra_conv_enable=feat_config.extra_conv_enable,
            feature_extra_conv_params=feat_config.extra_conv_params,
            join_type='single',
            join_arch='xcorr',
            join_params=dict(
                learn_spatial_weight=weight,
                use_batch_norm=True,
            ),
            window_params=dict(
                normalize_method='mean',
                window_profile='hann',
                combine_method='mul',
            ),
            window_radius=1.0,
            arg_max_eps=0,
            # TODO: Study weight decay and loss config.
            wd=1e-4,
            loss_params=args.loss_params,
        ),
    )
    return name, kwargs


def make_name(seed=None, **kwargs):
    name = '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())])
    if seed is not None:
        name += '_seed_' + str(seed)
    return name


def make_color(color_ind, color_num, level_ind, level_num, max_step=0.1):
    step = min(max_step, 0.5 / (level_num - 1))
    u = level_ind * step
    curr_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(COLORS[color_ind]))
    next_hsv = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(COLORS[color_ind + 1]))
    return matplotlib.colors.hsv_to_rgb((1 - u) * curr_hsv + u * next_hsv)


def make_dimensions_name(dims):
    return 'template_{}_search_{}'.format(dims['template_size'], dims['search_size'])


def make_dimensions_pair(dims):
    return (make_dimensions_name(dims), dims)


def find_dimensions(target_size,
                    desired_search_radius,
                    feat_config,
                    context):
    return siamfc.dimensions(
        target_size=target_size,
        desired_template_scale=context,
        desired_search_radius=desired_search_radius,
        # Must be same as constructor:
        feature_arch=feat_config.arch,
        feature_arch_params=feat_config.arch_params,
        feature_extra_conv_enable=feat_config.extra_conv_enable,
        feature_extra_conv_params=feat_config.extra_conv_params)


if __name__ == '__main__':
    main()
