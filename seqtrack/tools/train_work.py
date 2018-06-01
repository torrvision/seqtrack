import argparse
import functools
import itertools
import json
import numpy as np
import os
import tempfile

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import cnnutil
from seqtrack import helpers
from seqtrack import search
from seqtrack import slurm
from seqtrack import train
from seqtrack.models import util


def _train(args, name, seed):
    tmp_dir = _get_tmp_dir()
    if args.slurm:
        # Python will invoke slurm to run jobs.
        # Use different tmp dir for each job.
        tmp_data_dir = os.path.join(tmp_dir, 'data')
    else:
        # Jobs will be run in for loop.
        # Use specified tmp dir.
        tmp_data_dir = args.tmp_data_dir

    metrics = train.train(
        dir=os.path.join('trials', name),
        model_params=args.model_params,
        seed=seed,
        resume=args.resume,
        summary_dir='summary', summary_name=name,
        verbose_train=args.verbose_train,
        # Args from app.add_tracker_config_args()
        use_queues=args.use_queues,
        nosave=args.nosave,
        period_ckpt=args.period_ckpt,
        period_assess=args.period_assess,
        period_skip=args.period_skip,
        period_summary=args.period_summary,
        period_preview=args.period_preview,
        # save_videos=args.save_videos,
        save_frames=args.save_frames,
        session_config_kwargs=dict(
            gpu_manctrl=args.gpu_manctrl, gpu_frac=args.gpu_frac,
            log_device_placement=args.log_device_placement),
        # Arguments required for setting up data.
        # TODO: How to make this a parameter?
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        eval_datasets=args.eval_datasets,
        pool_datasets=args.pool_datasets,
        pool_split=args.pool_split,
        untar=args.untar,
        data_dir=args.data_dir,
        tar_dir=args.tar_dir,
        tmp_data_dir=tmp_data_dir,
        preproc_id=args.preproc,
        data_cache_dir=args.data_cache_dir,
        # Sampling:
        sampler_params=args.sampler_params,
        augment_motion=False,
        motion_params=None,
        # Args from app.add_eval_args()
        eval_tre_num=args.eval_tre_num,
        eval_samplers=args.eval_samplers,
        max_eval_videos=args.max_eval_videos,
        # Training process:
        ntimesteps=args.ntimesteps,
        batchsz=args.batchsz,
        imwidth=args.imwidth,
        imheight=args.imheight,
        lr_init=args.lr_init,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_rate=args.lr_decay_rate,
        optimizer=args.optimizer,
        # TODO: Take from args.__dict__?
        momentum=args.momentum,
        use_nesterov=args.use_nesterov,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        # weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_grad_norm=args.max_grad_norm,
        # siamese_pretrain=None,
        # siamese_model_file=None,
        num_steps=args.num_steps,
        use_gt_train=args.use_gt_train,
        gt_decay_rate=args.gt_decay_rate,
        min_gt_ratio=args.min_gt_ratio)

    return metrics


def _get_tmp_dir():
    if _is_slurm_job():
        return '/raid/local_scratch/{}-{}'.format(
            os.environ['SLURM_JOB_USER'], os.environ['SLURM_JOB_ID'])
    else:
        return tempfile.mkdtemp()


def _is_slurm_job():
    return 'SLURM_JOB_ID' in os.environ
