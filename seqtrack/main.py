import pdb
import sys
import argparse
import json
import numpy as np
import os
import random
import tensorflow as tf
from functools import partial
from itertools import chain

import logging
logger = logging.getLogger(__name__)

from seqtrack.opts import Opts
from seqtrack import data
# from seqtrack import model
from seqtrack import motion
from seqtrack import track
from seqtrack import train
from seqtrack import sample
from seqtrack.helpers import LazyDict

from seqtrack.models.nornn import Nornn
from seqtrack.models.itermodel import ModelFromIterModel
from seqtrack.models.siamfc import SiamFC


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')
    track.add_tracker_arguments(parser)

    parser.add_argument(
            '-v', '--verbose', help='print arguments',
            action='store_true')
    parser.add_argument(
            '--verbose_train', help='print train losses during train',
            action='store_true')
    parser.add_argument(
            '--report', help='do not train; print report of evaluation results',
            action='store_true')
    parser.add_argument(
            '--evaluate', help='do not train; simply evaluate tracker',
            action='store_true')
    parser.add_argument(
            '--debugmode', help='used for debugging',
            action='store_true')
    parser.add_argument(
            '--histograms', help='generate histograms in summary',
            action='store_true')
    parser.add_argument(
            '--tfdb', help='run tensorflow debugger',
            action='store_true')

    parser.add_argument(
            '--train_dataset',
            help='specify the training dataset; string, list of strings or list of tuples',
            type=json.loads, default='"ilsvrc_train"')
    parser.add_argument(
            '--val_dataset',
            type=json.loads, default='"ilsvrc_val"')
    parser.add_argument(
            '--eval_datasets', nargs='+', help='dataset on which to evaluate tracker (list)',
            type=str, default=['ilsvrc_val', 'otb_50'])
    parser.add_argument(
            '--eval_tre_num', type=int, default=3,
            help='number of points from which to start tracker in evaluation (full sampler only)')
    parser.add_argument(
            '--eval_samplers', nargs='+', help='',
            type=str, default=['full'])
    parser.add_argument(
            '--max_eval_videos', help='max number of videos to evaluate; not applied to OTB',
            type=int, default=100)

    parser.add_argument('--untar', action='store_true',
                        help='Untar dataset? Otherwise data must already exist')
    parser.add_argument('--data_dir', help='Location of datasets')
    parser.add_argument('--tar_dir', help='Location of dataset tarballs')
    parser.add_argument('--tmp_data_dir', default='/tmp/data/',
                        help='Temporary directory in which to untar data')
    parser.add_argument('--data_cache_dir', help='Where to cache the dataset metadata')
    parser.add_argument('--preproc', default='original')

    parser.add_argument(
            '--trainsplit', help='specify the split of train dataset (ILSVRC)',
            type=int, default=9)
    parser.add_argument(
            '--seed_global', help='random seed',
            type=int, default=9)

    # NOTE: (NL) any reason to have two arguments for this option?
    parser.add_argument('--resize-online', dest='useresizedimg', action='store_false')
    parser.set_defaults(useresizedimg=True)
    parser.add_argument(
            '--use_queues', help='enable queues for asynchronous data loading',
            action='store_true')
    # parser.add_argument(
    #         '--heatmap_params', help='JSON string specifying heatmap options',
    #         type=json.loads, default={'Gaussian': True})

    # JV: Move to model.
    # parser.add_argument(
    #         '--losses', nargs='+', help='list of losses to be used',
    #         type=str) # example [l1, iou]
    parser.add_argument(
            '--loss_coeffs', help='list of losses to be used',
            type=json.loads, default='{}')

    parser.add_argument(
            '--cnn_pretrain', help='specify if using pretrained model',
            action='store_true')
    # JV: Move to model.
    # parser.add_argument(
    #         '--cnn_trainable', help='set False to fix pretrained params',
    #         action='store_true')
    parser.add_argument(
            '--siamese_pretrain', help='specify if using pretrained model',
            action='store_true')
    parser.add_argument(
            '--siamese_model_file', help='specify if using pretrained model')

    parser.add_argument(
            '--num_steps', help='number of gradient steps',
            type=int, default=200000)
    parser.add_argument(
            '--batchsz', help='batch size',
            type=int, default=1)
    parser.add_argument(
            '--optimizer', help='optimizer to train the model',
            type=str, default='adam')
    parser.add_argument(
            '--lr_init', help='initial learning rate',
            type=float, default=1e-4)
    parser.add_argument(
            '--lr_decay_rate', help='geometric step for learning rate decay',
            type=float, default=1)
    parser.add_argument(
            '--lr_decay_steps', help='period for decaying learning rate',
            type=int, default=10000)
    # parser.add_argument(
    #         '--wd', help='weight decay', type=float, default=0.0)
    parser.add_argument(
            '--grad_clip', help='gradient clipping flag',
            action='store_true')
    parser.add_argument(
            '--max_grad_norm', help='threshold for gradient clipping',
            type=float, default=5.0)
    parser.add_argument(
            '--gt_decay_rate', help='decay rate for gt_ratio',
            type=float, default=1e-6)
    parser.add_argument(
            '--min_gt_ratio', help='lower bound for gt_ratio',
            type=float, default=0.75)
    parser.add_argument(
            '--curriculum_learning', help='restore variables from a pre-trained model (on short sequences)',
            action='store_true')
    parser.add_argument(
            '--pretrained_cl', help='pretrained model to be used for curriculum_learning',
            type=str, default=None)
    parser.add_argument(
            '--use_gt_train', help='use ground-truth during training',
            action='store_true')
    parser.add_argument(
            '--use_gt_eval', help='use ground-truth during evaluation', # Should be set False in most cases.
            action='store_true')

    parser.add_argument(
            '--color_augmentation', help='JSON string specifying color augmentation',
            type=json.loads, default={'brightness': False,
                                      'contrast': False,
                                      'grayscale': False})

    parser.add_argument(
            '--sampler_params', help='JSON string specifying sampler',
            type=json.loads, default={'kind': 'regular', 'freq': 10})
    parser.add_argument('--augment_motion', help='enable motion augmentation?', action='store_true')
    parser.add_argument(
            '--motion_params', help='JSON string specifying motion augmentation',
            type=json.loads, default={})

    # parser.add_argument(
    #         '--path_data_home', help='location of datasets',
    #         type=str, default='./data')
    parser.add_argument(
            '--nosave', help='no need to save results?',
            action='store_true')
    parser.add_argument(
            '--resume', help='to resume training',
            action='store_true')
    parser.add_argument(
            '--period_ckpt', help='period to save ckpt',
            type=int, default=10000)
    parser.add_argument(
            '--period_assess', help='period to run evaluation',
            type=int, default=10000)
    parser.add_argument(
            '--period_skip', help='until this period skip evaluation',
            type=int, default=10000)
    parser.add_argument(
            '--period_preview', help='period to update summary preview',
            type=int, default=100)
    parser.add_argument(
            '--save_videos', help='create video during evaluation',
            action='store_true')
    parser.add_argument(
            '--save_frames', help='save frames of video during evaluation',
            action='store_true')

    parser.add_argument(
            '--gpu_device', help='set `CUDA_VISIBLE_DEVICES`',
            type=int, default=0)
    parser.add_argument(
            '--no_gpu_manctrl', help='disable manual gpu management',
            dest='gpu_manctrl', action='store_false')
    parser.add_argument(
            '--gpu_frac', help='fraction of gpu memory',
            type=float, default=0.4)

    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return args


def main():
    # sys.settrace(trace)  # Use this to find segfaults.

    args = parse_arguments()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    # TODO: How to get datasets from train_dataset and eval_datasets?
    dataset_names = list(set(chain(
        _datasets_in_sampler(args.train_dataset),
        _datasets_in_sampler(args.val_dataset),
        args.eval_datasets)))
    logger.info('load datasets: %s', _list_str(dataset_names))

    if args.untar:
        datasets = data.untar_and_load_all(
            args.tar_dir, args.tmp_data_dir, args.preproc, dataset_names,
            cache_dir=args.data_cache_dir)
    else:
        datasets = {
            name: data.load(args.data_dir, args.preproc, name,
                            cache=True, cache_dir=args.data_cache_dir) for name in datasets}

    frame_sampler_presets = {
        'full': partial(sample.FrameSampler, kind='full'),
        'train': partial(sample.FrameSampler, ntimesteps=o.ntimesteps, **o.sampler_params)}

    # Create example streams for train and val.
    # Use a separate random number generator for each sampler.
    sampler_specs = {'train': args.train_dataset, 'val': args.val_dataset}
    streams = {}
    for i, mode in enumerate(['train', 'val']):
        seed = o.seed_global + i  # Use a different seed for train and val.
        postproc_fn = (
            None if not o.augment_motion else
            partial(motion.augment, rand=np.random.RandomState(seed), **o.motion_params))
        streams[mode] = sample.sample(
            _make_sampler(sampler_specs[mode], datasets), frame_sampler_presets['train'](),
            postproc_fn=postproc_fn, rand=np.random.RandomState(seed), infinite=True)
    # Create functions to sample finite sets for evaluation.
    eval_sample_fns = {
        # Give each dataset its own random seed.
        (d + '-' + s): partial(
            sample.sample, sample.EpochSampler(datasets[d]), frame_sampler_presets[s](),
            rand=np.random.RandomState(o.seed_global), infinite=False, max_num=100)
        for d in o.eval_datasets for s in o.eval_samplers}

    if o.report:
        train.generate_report(sorted(o.eval_samplers), sorted(o.eval_datasets), o)
        return

    if o.model == 'SiamFC':
        model = ModelFromIterModel(SiamFC(**o.model_params))
    else:
        raise ValueError('unknown model: {}'.format(o.model))

    train.train(model, streams, eval_sample_fns, o, use_queues=o.use_queues)


def _make_sampler(names, datasets):
    '''
    Args:
        names: string (EpochSampler), list of strings (EpochSampler of Concat) or
            list of float-string-pairs (MixtureSampler).
    '''
    if isinstance(names, basestring):
        return sample.EpochSampler(datasets[names])
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, basestring) for name in names):
            concat = data.Concat({name: datasets[name] for name in names})
            return sample.EpochSampler(concat)
        elif all(_is_pair(elem) for elem in names):
            samplers = {name: sample.EpochSampler(datasets[name]) for _, name in names}
            weights = {name: weight for weight, name in names}
            return sample.MixtureSampler(samplers, weights)
    raise ValueError('invalid structure: {}'.format(repr(names)))


def _is_pair(x):
    return (isinstance(x, list) or isinstance(x, tuple)) and len(x) == 2


def _datasets_in_sampler(names):
    if isinstance(names, basestring):
        return [names]
    if isinstance(names, list):
        assert len(names) > 0
        if all(isinstance(name, basestring) for name in names):
            return names
        elif all(_is_pair(elem) for elem in names):
            return [name for _, name in names]
    raise ValueError('invalid structure: {}'.format(repr(names)))


def _quote(x):
    return "'" + x + "'"


def _list_str(x):
    return ', '.join(map(_quote, x))


def trace(frame, event, arg):
    print >> sys.stderr, '%s, %s:%d' % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace


if __name__ == "__main__":
    main()
