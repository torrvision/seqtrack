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
import model
import train
import sample
from helpers import LazyDict


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

    ## parser.add_argument(
    ##         '--dataset', help='specify the name of dataset',
    ##         type=str, default='')
    parser.add_argument(
            '--train_dataset',
            help='specify the training dataset; string, list of strings or list of tuples',
            type=json.loads, default='"ILSVRC-train"')
    parser.add_argument(
            '--eval_datasets', nargs='+', help='dataset on which to evaluate tracker (list)',
            type=str, default=['ILSVRC-val', 'OTB-50'])
    parser.add_argument(
            '--eval_samplers', nargs='+', help='',
            type=str, default=['full'])
    parser.add_argument(
            '--max_eval_videos', help='max number of videos to evaluate; not applied to OTB',
            type=int, default=100)

    parser.add_argument(
            '--trainsplit', help='specify the split of train dataset (ILSVRC)',
            type=int, default=9)
    parser.add_argument(
            '--seed_global', help='random seed',
            type=int, default=9)

    parser.add_argument(
            '--frmsz', help='size of a square image', type=int, default=257)
    # NOTE: (NL) any reason to have two arguments for this option?
    parser.add_argument('--resize-online', dest='useresizedimg', action='store_false')
    parser.set_defaults(useresizedimg=True)
    parser.add_argument(
            '--use_queues', help='enable queues for asynchronous data loading',
            action='store_true')
    parser.add_argument(
            '--heatmap_params', help='JSON string specifying heatmap options',
            type=json.loads, default={'Gaussian': True})

    parser.add_argument(
            '--model', help='model!',
            type=str, default='')
    parser.add_argument(
            '--model_params', help='JSON string specifying model',
            type=json.loads, default={})
    parser.add_argument(
            '--losses', nargs='+', help='list of losses to be used',
            type=str) # example [l1, iou]
    parser.add_argument(
            '--search_scale', help='size of search space relative to target',
            type=int, default=4)
    parser.add_argument(
            '--target_scale', help='size of context relative to target',
            type=int, default=1)
    parser.add_argument(
            '--perspective', help='ic: image-centric, oc: object-centric',
            type=str, default='oc')
    parser.add_argument(
            '--aspect_method', help='method for fixing aspect ratio',
            type=str, default='stretch',
            choices=['stretch', 'area', 'perimeter'])

    parser.add_argument(
            '--cnn_model', help='pretrained CNN model',
            type=str, default='custom')
    parser.add_argument(
            '--cnn_pretrain', help='specify if using pretrained model',
            action='store_true')
    parser.add_argument(
            '--cnn_trainable', help='set False to fix pretrained params',
            action='store_true')

    parser.add_argument(
            '--nunits', help='number of hidden units in rnn cell',
            type=int, default=256)
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=20)

    parser.add_argument(
            '--nepoch', help='number of epochs',
            type=int, default=20)
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
            '--th_prob_stay', help='threshold probability to stay movement',
            type=float, default=0.0)

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
    parser.add_argument('--augment_motion', help='enable motion augmentation?', action='store_true')
    parser.add_argument(
            '--motion_params', help='JSON string specifying motion augmentation',
            type=json.loads, default={})

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
    datasets = LazyDict()
    csv_datasets = [
        'vot2013', 'vot2014', 'vot2016', 'vot2017',
        'otb50', 'otb100', 'otb_diff',
        'tc', 'dtb70', 'nuspro', 'uav123',
        'pool636_train', 'pool636_val',
    ]
    for name in csv_datasets:
        datasets[name] = functools.partial(data.CSV, name, o)
    datasets['ILSVRC-train'] = functools.partial(data.Data_ILSVRC, 'train', o)
    datasets['ILSVRC-val'] = functools.partial(data.Data_ILSVRC, 'val', o)
    datasets['OTB-50'] = functools.partial(data.Data_OTB, 'OTB-50', o)
    datasets['OTB-100'] = functools.partial(data.Data_OTB, 'OTB-100', o)
    # Add some pre-defined unions.
    datasets['vot'] = lambda: data.Concat({name: datasets[name] for name in [
        'vot2013', 'vot2014', 'vot2016', 'vot2017']})
    datasets['pool574'] = lambda: data.Concat({name: datasets[name] for name in [
        'vot2013', 'vot2014', 'vot2016', 'vot2017', 'tc', 'dtb70', 'nuspro']})
    datasets['pool697'] = lambda: data.Concat({name: datasets[name] for name in [
        'vot2013', 'vot2014', 'vot2016', 'vot2017', 'tc', 'dtb70', 'nuspro', 'uav123']})

    # Construct training dataset object from string.
    # If it is a string, use a single dataset.
    # If it is a list of strings, concatenate those datasets.
    # If it is a list of (weight, string) pairs, construct a DatasetMixture.
    if isinstance(o.train_dataset, basestring):
        train_dataset = datasets[o.train_dataset]
    elif isinstance(o.train_dataset, list):
        assert len(o.train_dataset) > 0
        if isinstance(o.train_dataset[0], basestring):
            assert all(isinstance(x, basestring) for x in o.train_dataset)
            train_dataset = data.Concat({name: datasets[name] for name in o.train_dataset})
        else:
            # Must be a list of pairs.
            assert all(len(x) == 2 for x in o.train_dataset)
            train_dataset = sample.DatasetMixture(
                {name: (weight, datasets[name]) for weight, name in o.train_dataset})
    else:
        raise ValueError('train_dataset is not a string or a list: {}'.format(repr(o.train_dataset)))

    # These are the possible choices for evaluation sampler.
    # No need to specify `shuffle`, `max_videos`, `max_objects` here,
    # but `ntimesteps` should be set if applicable.
    sampler_presets = {
        'full':   functools.partial(sample.sample, kind='full'),
        # The 'train' sampler is the same as used during training.
        # This may be useful for detecting over-fitting.
        'train':  functools.partial(sample.sample, ntimesteps=o.ntimesteps, **o.sampler_params),
        # The 'custom' sampler can be modified for a quick and dirty test.
        'custom': functools.partial(sample.sample, kind='regular',
                                    freq=o.sampler_params.get('freq', 10),
                                    ntimesteps=o.ntimesteps),
    }
    # Take all dataset-sampler combinations.
    # Different policies are used for choosing trajectories in OTB and ILSVRC:
    # ILSVRC is large and therefore a random subset of videos are used,
    # with 1 object per video.
    # OTB is small and therefore all tracks in all videos are used.
    eval_sets = {
        # Give each evaluation set its own random seed.
        d+'-'+s: functools.partial(sampler_presets[s], datasets[d],
            rand=np.random.RandomState(o.seed_global),
            max_videos=o.max_eval_videos,
            shuffle=True,
            max_objects=1 if d.startswith('ILSVRC-') else None)
        for d in o.eval_datasets
        for s in o.eval_samplers
    }

    if o.report:
        train.generate_report(sorted(o.eval_samplers), sorted(o.eval_datasets), o)
        return

    # TODO: Set model_opts from command-line or JSON file?
    m = model.load_model(o, model_params=o.model_params)

    train.train(m, {'train': train_dataset, 'val': datasets['ILSVRC-val']},
                eval_sets, o, use_queues=o.use_queues)


def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace

if __name__ == "__main__":
    main()
