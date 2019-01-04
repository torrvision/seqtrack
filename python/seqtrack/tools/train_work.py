from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from seqtrack import train


def work(args, context, seed):
    tmp_dir = context.tmp_dir()
    if tmp_dir is None:
        tmp_data_dir = args.tmp_data_dir
    else:
        tmp_data_dir = os.path.join(tmp_dir, 'data')

    return train.train(
        dir=os.path.join('trials', context.name),
        model_params=args.model_params,
        seed=seed,
        resume=args.resume,
        summary_dir='summary',
        summary_name=context.name,
        verbose_train=args.verbose_train,
        # Args from app.add_tracker_config_args()
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
        # Arguments required for setting up data.
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
        augment_motion=args.augment_motion,
        motion_params=args.motion_params,
        # Args from app.add_eval_args()
        eval_tre_num=args.eval_tre_num,
        # eval_samplers=args.eval_samplers,
        max_eval_videos=args.max_eval_videos,
        # Training process:
        ntimesteps=args.ntimesteps,
        batchsz=args.batchsz,
        imwidth=args.imwidth,
        imheight=args.imheight,
        lr_init=args.lr_init,
        lr_params=args.lr_params,
        optimizer=args.optimizer,
        optimizer_params=args.optimizer_params,
        grad_clip=args.grad_clip,
        grad_clip_params=args.grad_clip_params,
        # siamese_pretrain=None,
        # siamese_model_file=None,
        num_steps=args.num_steps,
        use_gt_train=args.use_gt_train,
        gt_decay_rate=args.gt_decay_rate,
        min_gt_ratio=args.min_gt_ratio)
