'''
Compares tracking results with different windows.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import slurm
from seqtrack import train


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    # TODO: Avoid re-training!!
    profiles = ['rect', 'linear', 'quadratic', 'hann', 'cosine']
    radii = [0.5, 1.0, 2.0]

    # Map stream of named vectors to stream of named results (order may be different).
    kwargs = dict([
        make_kwargs(args, seed=seed, radius=radius, profile=profile)
        for seed in range(args.num_trials)
        for profile in profiles
        for radius in radii
    ])
    mapper = make_mapper(args)
    result_stream = mapper(slurm.partial_apply_kwargs(train.train_worker), kwargs.items())
    results = dict(result_stream)

    import pdb; pdb.set_trace()

    # # To obtain one number per training process, we use one dataset as validation.
    # summaries = {}
    # for feat, feat_config in feature_configs.items():
    #     for weight in use_spatial_weights:
    #         for context in desired_context_amounts:
    #             summary_name = make_name(feat=feat, weight=weight, context=context)
    #             trial_names = [make_name(feat=feat, weight=weight, context=context, seed=seed)
    #                            for seed in range(args.num_trials)]
    #             summary = train.summarize_trials(
    #                 [results[name]['track_series'] for name in trial_names],
    #                 val_dataset=args.optimize_dataset,
    #                 sort_key=lambda metrics: metrics[args.optimize_metric])
    #             # Add model parameters.
    #             # TODO: Move into summarize_trials?
    #             model_properties = helpers.unique_value(
    #                 [results[name]['model_properties'] for name in trial_names])
    #             summary.update({'model/' + k: v for k, v in model_properties.items()})
    #             summaries[summary_name] = summary

    # import matplotlib
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(4, 3))
    # fig, ax = plt.subplots()
    # plt.xlabel('Template context')
    # plt.ylabel('Mean IOU')
    # for feat, feat_config in feature_configs.items():
    #     for weight in use_spatial_weights:
    #         name_fn = lambda context: make_name(feat=feat, weight=weight, context=context)
    #         contexts = [summaries[name_fn(context)]['model/template_scale']
    #                     for context in desired_context_amounts]
    #         quality_metric = args.optimize_dataset + '_' + args.optimize_metric
    #         quality = [summaries[name_fn(context)][quality_metric]
    #                    for context in desired_context_amounts]
    #         plt.scatter(contexts, quality, label=feat + (' (weight)' if weight else ''))
    # ax.legend()
    # plt.savefig('plot.pdf')


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


def make_kwargs(args, seed, profile, radius):
    name = make_name(seed=seed, profile=profile, radius=radius)
    return name, dict(
        dir=os.path.join('train', name),
        model_params=dict(
            use_desired_size=True,
            target_size=64,
            desired_template_scale=2.0,
            desired_search_radius=1.0,
            feature_arch='alexnet',
            feature_arch_params=None,
            join_type='single',
            join_arch='xcorr',
            join_params=dict(
                use_batch_norm=True,
            ),
            window_radius=radius,
            window_params=dict(
                normalize_method='mean',
                window_profile=profile,
                combine_method='mul',
            ),
            arg_max_eps=0.01,
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


def make_name(seed=None, **kwargs):
    parts = [key + '_' + str(kwargs[key]) for key in sorted(kwargs.keys())]
    if seed is not None:
        parts.append('seed_' + str(seed))
    return '_'.join(parts)


if __name__ == '__main__':
    main()
