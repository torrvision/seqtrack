from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import numpy as np

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import helpers
from seqtrack import param_search
from seqtrack import train
from seqtrack import slurm

# The pickled object must be imported to unpickle in a different package (slurmproc.worker).
from seqtrack.tools import param_search_train_work as work


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))

    input_stream = make_input_stream(args, np.random.RandomState(args.seed))

    train_fn = functools.partial(
        train.train_worker_params,
        kwargs=app.train_kwargs(args),
        train_dir='train',
    )

    param_search.main(
        train_fn,
        input_stream,
        # kwargs_fn=functools.partial(to_kwargs, args),
        postproc_fn=functools.partial(select_best_epoch, args),
        use_existing_inputs=True,
        max_num_configs=args.num_configs,
        report_only=args.report_only,
        cache_dir='cache',
        input_codec='json', kwargs_codec='json', output_codec='msgpack',
        use_slurm=args.use_slurm,
        slurm_kwargs=dict(
            tempdir='tmp',
            max_submit=args.slurm_max_submit,
            opts=['--' + x for x in args.slurm_flags],
        ),
        # slurm_group_size=args.slurm_group_size,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default='info', help='debug, info, warning')

    parser.add_argument('-n', '--num_configs', type=int, default=8,
                        help='number of configurations')
    parser.add_argument('--report_only', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='seed for configurations')
    # parser.add_argument('--distribution', type=json.loads,
    #                     help='Override distribution of a parameter (JSON string)')
    # parser.add_argument('--distribution_file',
    #                     help='Override distribution of a parameter (JSON file)')
    parser.add_argument('--optimize_dataset', default='pool_val-full',
                        help='eval_dataset to use to choose model')
    parser.add_argument('--optimize_metric', default='OPE/iou_seq_mean',
                        help='metric to optimize for')

    parser.add_argument('--no_slurm', dest='use_slurm', action='store_false',
                        help='Submit jobs to slurm or run directly?')
    parser.add_argument('--slurm_max_submit', type=int,
                        help='Maximum number of jobs to put in queue')
    parser.add_argument('--slurm_flags', nargs='+', help='flags for sbatch (without "--")')
    # parser.add_argument('--slurm_group_size', type=int, default=1)

    app.add_setup_data_args(parser)
    app.add_tracker_config_args(parser)
    # TODO: Not all of the `train_args` parameters will be used.
    # Find a cleaner way to keep the defaults for e.g. pool_datasets?
    app.add_train_args(parser)
    app.add_eval_args(parser)

    return parser.parse_args()


def make_input_stream(args, rand):
    '''Yields a stream of (name, kwargs) pairs.'''
    rand = np.random.RandomState(args.seed)
    sample_fn = make_sample_fn(args)
    while True:
        params = sample_fn(rand)
        name = param_search.hash(params)
        yield name, params


def select_best_epoch(args, results):
    # Use optimize_dataset to choose checkpoint of each experiment.
    # If there were multiple trials, take mean over trials.
    return train.summarize_trials([results], val_dataset=args.optimize_dataset,
                                  sort_key=lambda metrics: metrics[args.optimize_metric])


def make_sample_fn(args):

    def sample_train_params(rand):
        return dict(
            model_params=sample_siamfc_params(rand),
            seed=rand.randint(0xffff),
            # Dataset:
            train_dataset=args.train_dataset,
            val_dataset=args.val_dataset,
            eval_datasets=args.eval_datasets,
            pool_datasets=args.pool_datasets,
            pool_split=args.pool_split,
            # Sampling:
            sampler_params=sample_frame_sampler_params(rand),
            augment_motion=False,
            motion_params=None,
            # Training:
            ntimesteps=1,
            num_steps=args.num_steps,
            metrics_resolution=1000,
            period_assess=args.period_assess,
            extra_assess=args.extra_assess,
            period_skip=args.period_skip,
            batchsz=8,
            imwidth=360,
            imheight=360,
            preproc_id='original',
            resize_online=True,
            resize_method='bilinear',
            lr_schedule='remain',
            lr_init=1e-3,
            lr_params=None,
            optimizer='momentum',
            optimizer_params={'momentum': 0.9},
            grad_clip=False,
            grad_clip_params=None,
            siamese_pretrain=None,
            siamese_model_file=None,
            use_gt_train=True,
            gt_decay_rate=1,
            min_gt_ratio=0,
            # Evaluation:
            eval_samplers='full',
            max_eval_videos=args.max_eval_videos,
            eval_tre_num=args.eval_tre_num,
        )

    def sample_frame_sampler_params(rand):
        return dict(
            kind='freq-range-fit',
            min_freq=50,
            max_freq=100,
            use_log=False,
        )

    def sample_siamfc_params(rand):
        return dict(
            target_size=64,
            template_size=127,
            search_size=255,
            aspect_method='perimeter',
            use_gt=True,
            curr_as_prev=True,
            pad_with_mean=True,
            feather=False,
            center_input_range=True,
            keep_uint8_range=False,
            feature_arch='alexnet',
            feature_arch_params=sample_alexnet_params(rand),
            feature_extra_conv_enable=False,
            join_type='single',
            join_arch='xcorr',
            join_params=sample_xcorr_params(rand),
            freeze_siamese=False,
            learnable_prior=False,
            train_multiscale=False,
            # Tracking parameters:
            search_method='local',
            num_scales=rand.choice([3, 5]),
            scale_step=rand.choice([1.01, 1.02, 1.03]),
            scale_update_rate=1,
            report_square=False,
            window_params=sample_window_params(rand),
            window_radius=rand.choice([0.5, 1.0, 2.0]),
            arg_max_eps=rand.choice([0.0, 0.01, 0.05]),
            # Loss parameters:
            wd=0.0,
            loss_params=sample_loss_params(rand),
        )

    def sample_alexnet_params(rand):
        return dict(
            output_layer='conv5',
            output_act='linear',
            freeze_until_layer=None,
            padding='VALID',
            enable_bnorm=True,
        )

    def sample_xcorr_params(rand):
        return dict(
            enable_pre_conv=False,
            pre_conv_params=None,
            learn_spatial_weight=False,
            reduce_channels=True,
            use_mean=False,
            use_batch_norm=True,
            learn_gain=False,
        )

    def sample_window_params(rand):
        return dict(
            normalize_method='mean',
            window_profile=rand.choice(['hann']),
            window_mode=rand.choice(['radial', 'cartesian']),
            combine_method='mul',
        )

    def sample_loss_params(rand):
        return dict(
            method='sigmoid',
            params=sample_sigmoid_params(rand),
        )

    def sample_sigmoid_params(rand):
        return dict(
            balanced=True,
            pos_weight=1,
            label_method='gaussian',
            label_params=sample_gaussian_label_params(rand),
        )

    def sample_gaussian_label_params(rand):
        return dict(sigma=0.2)

    return sample_train_params


if __name__ == '__main__':
    main()
