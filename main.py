import pdb
import sys
import argparse
import functools
import json
import numpy as np
import random
import tensorflow as tf

from opts           import Opts
import data
import model as model_package
import train
import sample


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')

    parser.add_argument(
            '--verbose', help='print arguments',
            action='store_true')
    parser.add_argument(
            '--verbose_train', help='print train losses during train',
            action='store_true')
    parser.add_argument(
            '--report', help='generate report after training',
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
            '--dataset', help='specify the name of dataset',
            type=str, default='')
    parser.add_argument(
            '--trainsplit', help='specify the split of train dataset (ILSVRC)',
            type=int, default=9)
    parser.add_argument(
            '--seed_global', help='random seed',
            type=int, default=9)

    parser.add_argument(
            '--frmsz', help='size of a square image', type=int, default=241)
    # NOTE: (NL) any reason to have two arguments for this option?
    parser.add_argument('--resize-online', dest='useresizedimg', action='store_false')
    parser.set_defaults(useresizedimg=True)
    parser.add_argument(
            '--use_queues', help='enable queues for asynchronous data loading',
            action='store_true')

    parser.add_argument(
            '--model', help='model!',
            type=str, default='')
    parser.add_argument(
            '--losses', nargs='+', help='list of losses to be used',
            type=str) # example [l1, iou]
    parser.add_argument('--heatmap_stride', type=int, default=1,
            help='stride of heatmap at loss')

    parser.add_argument(
            '--nunits', help='number of hidden units in rnn cell',
            type=int, default=256)
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=20)

    parser.add_argument(
            '--model_params', help='JSON string specifying model',
            type=json.loads, default={})

    parser.add_argument(
            '--num_steps', help='number of epochs',
            type=int, default=100000)
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
    parser.add_argument(
            '--wd', help='weight decay', type=float, default=0.0)
    parser.add_argument(
            '--grad_clip', help='gradient clipping flag',
            action='store_true')
    parser.add_argument(
            '--max_grad_norm', help='threshold for gradient clipping',
            type=float, default=5.0)
    parser.add_argument(
            '--gt_decay_rate', help='decay rate for gt_ratio',
            type=float, default=-1e-2)
    parser.add_argument(
            '--min_gt_ratio', help='lower bound for gt_ratio',
            type=float, default=0.75)
    parser.add_argument(
            '--curriculum_learning', help='restore variables from a pre-trained model (on short sequences)',
            action='store_true')
    parser.add_argument(
            '--model_file', help='pretrained model file to be used for curriculum_learning',
            type=str, default=None)
    parser.add_argument('--object_centric', action='store_true')

    parser.add_argument(
            '--data_augmentation', help='JSON string specifying data augmentation',
            type=json.loads, default={'scale_shift': False,
                                      'flip_up_down': False,
                                      'flip_left_right': False,
                                      'brightness': False,
                                      'contrast': False,
                                      'hue': False,
                                      'saturation': False})

    parser.add_argument(
            '--sampler_params', help='JSON string specifying sampler',
            type=json.loads, default={'kind': 'regular', 'freq': 10})
    parser.add_argument(
            '--eval_datasets', nargs='+', help='dataset on which to evaluate tracker',
            type=str, default=['ILSVRC-train'])
    parser.add_argument(
            '--eval_samplers', nargs='*', help='',
            type=str, default=['train'])

    parser.add_argument(
            '--path_data_home', help='location of datasets',
            type=str, default='./data')
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
            type=int, default=20000)
    parser.add_argument(
            '--visualize_eval', help='create video during evaluation',
            action='store_true')

    parser.add_argument(
            '--gpu_device', help='set `CUDA_VISIBLE_DEVICES`',
            type=int, default=0)
    parser.add_argument(
            '--gpu_frac', help='fraction of gpu memory',
            type=float, default=0.4)

    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return args

def main():
    # sys.settrace(trace) # Use this to find segfaults.

    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    # datasets = data.load_data(o)
    datasets = {
        'ILSVRC-train': data.Data_ILSVRC('train', o),
        'ILSVRC-val':   data.Data_ILSVRC('val', o),
        'OTB-50':       data.Data_OTB('OTB-50', o),
        'OTB-100':      data.Data_OTB('OTB-100', o),
    }

    # Presets are calls to sample.make_frame_sampler with all params given except dataset.
    # Note that these are independent of the arguments to sample.epoch()
    sampler_presets = {
        'full': functools.partial(sample.make_frame_sampler,
            kind='full', ntimesteps=None),
        # The 'train' sampler is the same as used during training.
        # This may be useful for detecting over-fitting.
        'train': functools.partial(sample.make_frame_sampler,
            ntimesteps=o.ntimesteps, **o.sampler_params),
    }
    # Take all dataset-sampler combinations.
    # Different policies are used for choosing trajectories in OTB and ILSVRC:
    # ILSVRC is large and therefore a random subset of videos are used,
    # with 1 object per video.
    # OTB is small and therefore all tracks in all videos are used.
    eval_sets = {}
    for s in o.eval_samplers:
        for d in o.eval_datasets:
            rand = random.Random(o.seed_global)
            eval_sets[d+'-'+s] = functools.partial(sample.epoch,
                dataset=datasets[d],
                rand=rand,
                frame_sampler=functools.partial(
                    sampler_presets[s](dataset=datasets[d]),
                    rand=rand),
                max_videos=None if d.startswith('OTB-') else o.max_eval_videos,
                max_objects=None if d.startswith('OTB-') else 1)

    if o.report:
        train.generate_report(sorted(o.eval_samplers), sorted(o.eval_datasets), o)
        return

    # TODO: Set model_opts from command-line or JSON file?
    model = model_package.load_model(o, model_params=o.model_params)

    train_datasets = {
        'train': datasets['ILSVRC-train'],
        'val': datasets['ILSVRC-val'],
        'OTB-50': datasets['OTB-50']
    }
    train.train(model, train_datasets, eval_sets, o, use_queues=o.use_queues)


def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace

if __name__ == "__main__":
    main()
