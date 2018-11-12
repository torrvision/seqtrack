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
import pprint

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import cnn
from seqtrack import helpers
from seqtrack import slurm
from seqtrack import train
# from seqtrack.models import util

# # The pickled object must be imported to unpickle in a different package (slurmproc.worker).
# from seqtrack.tools import train_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # Vary the amount of context with and without spatial weights.
    # Do this for several different feature architectures.
    # It may be good to vary the resolution of the template too;
    # this can be another experiment.

    FeatureConfig = collections.namedtuple('FeatureConfig', ['arch', 'arch_params'])
    feature_configs = {
        'alexnet_conv2': FeatureConfig(
            arch='alexnet',
            arch_params=dict(
                output_layer='conv2')),
        'alexnet_conv3': FeatureConfig(
            arch='alexnet',
            arch_params=dict(
                output_layer='conv3')),
        'alexnet_conv5': FeatureConfig(
            arch='alexnet',
            arch_params=dict(
                output_layer='conv5')),
    }
    use_spatial_weights = [False, True]
    desired_context_amounts = [1.5, 2.0, 3.0]

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args,
                    feat=feat,
                    feat_config=feat_config,
                    weight=weight,
                    context=context,
                    seed=seed)
        for feat, feat_config in feature_configs.items()
        for weight in use_spatial_weights
        for context in desired_context_amounts
        for seed in range(args.num_trials)
    ])

    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    names = sorted(kwargs.keys())
    for name in names:
        print('-' * 40)
        print('name:', name)
        print('result:')
        pprint.pprint(results[name])

    summary = train.summarize_trials(results.values(), val_dataset=args.optimize_dataset,
                                     sort_key=lambda metrics: metrics[args.optimize_metric])

    print()
    print('summary:')
    pprint.pprint(summary)


def parse_arguments():
    parser = argparse.ArgumentParser()

    app.add_setup_data_args(parser)
    app.add_tracker_config_args(parser)
    app.add_eval_args(parser)

    parser.add_argument('--loglevel', default='info', help='debug, info, warning')
    parser.add_argument('--verbose_train', action='store_true')

    parser.add_argument('--slurm', action='store_true',
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


def make_mapper(args):
    if args.slurm:
        mapper = slurm.SlurmDictMapper(tempdir='tmp', opts=['--' + x for x in args.slurm_flags])
    else:
        mapper = helpers.map_dict
    # Cache the results and use SLURM mapper to evaluate those without cache.
    mapper = helpers.CachedDictMapper(dir=os.path.join('cache', 'train'),
                                      codec_name='msgpack', mapper=mapper)
    return mapper


def make_kwargs(args, feat, feat_config, weight, context, seed):
    name = make_name(feat=feat, weight=weight, context=context, seed=seed)
    return name, dict(
        dir=os.path.join('train', name),
        model_params=dict(
            use_desired_size=True,
            target_size=64,
            desired_template_scale=context,
            desired_search_radius=1.0,
            feature_arch=feat_config.arch,
            feature_arch_params=feat_config.arch_params,
            join_type='single',
            join_arch='xcorr',
            join_params=dict(
                learn_spatial_weight=weight,
                use_batch_norm=True,
            ),
            hann_method='none',  # none, mul_prob, add_logit
            hann_coeff=1.0,
            # TODO: Study weight decay and loss config.
            wd=1e-4,
            balance_classes=True,
            enable_ce_loss=True,
            enable_margin_loss=False,
            ce_label='gaussian_distance',
        ),
        seed=seed,
        # From app.add_setup_data_args():
        untar=args.untar,
        data_dir=args.data_dir,
        tar_dir=args.tar_dir,
        tmp_data_dir=args.tmp_data_dir,
        preproc_id=args.preproc,
        data_cache_dir=args.data_cache_dir,
        pool_datasets=args.pool_datasets,
        pool_split=args.pool_split,
        # From app.add_instance_arguments():
        ntimesteps=args.ntimesteps,
        batchsz=args.batchsz,
        imwidth=args.imwidth,
        imheight=args.imheight,
        # From app.add_train_args():
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        num_steps=args.num_steps,
        lr_init=args.lr_init,
        lr_params=args.lr_params,
        optimizer=args.optimizer,
        optimizer_params=args.optimizer_params,
        grad_clip=args.grad_clip,
        grad_clip_params=args.grad_clip_params,
        use_gt_train=args.use_gt_train,
        gt_decay_rate=args.gt_decay_rate,
        min_gt_ratio=args.min_gt_ratio,
        sampler_params=args.sampler_params,
        augment_motion=args.augment_motion,
        motion_params=args.motion_params,
        # From app.add_eval_args():
        eval_datasets=args.eval_datasets,
        eval_tre_num=args.eval_tre_num,
        max_eval_videos=args.max_eval_videos,
        # From add_tracker_config_args(parser)
        use_queues=args.use_queues,
        nosave=args.nosave,
        period_ckpt=args.period_ckpt,
        period_assess=args.period_assess,
        period_skip=args.period_skip,
        period_summary=args.period_summary,
        period_preview=args.period_preview,
        visualize=args.visualize,
        keep_frames=args.keep_frames,
        session_config_kwargs=dict(
            gpu_manctrl=args.gpu_manctrl,
            gpu_frac=args.gpu_frac,
            log_device_placement=args.log_device_placement),
        # Other arguments:
        verbose_train=args.verbose_train,
        summary_dir='summary',
        summary_name=name,
    )


def make_name(**kwargs):
    return '_'.join([key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())])


if __name__ == '__main__':
    main()
