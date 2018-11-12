from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


def add_instance_arguments(parser):
    '''Arguments required to instantiate the model.'''
    parser.add_argument('--ntimesteps', type=int, default=1,
                        help='number of time steps for rnn',)
    parser.add_argument('--batchsz', type=int, default=8, help='batch size')
    parser.add_argument('--imwidth', type=int, default=640, help='image resolution')
    parser.add_argument('--imheight', type=int, default=360, help='image resolution')

    # parser.add_argument('--model', default='')
    # parser.add_argument('--model_params', type=json.loads, default={},
    #                     help='JSON string specifying model')


def add_setup_data_args(parser):
    '''Arguments for train.setup_data.'''
    parser.add_argument('--untar', action='store_true',
                        help='Untar dataset? Otherwise data must already exist')
    parser.add_argument('--data_dir', help='Location of datasets (if already installed)')
    parser.add_argument('--tar_dir', help='Location of dataset tarballs')
    parser.add_argument('--tmp_data_dir', default='/tmp/data/',
                        help='Temporary directory in which to untar data (if slurm is disabled)')
    parser.add_argument('--preproc', default='original',
                        help='Name of preprocessing e.g. original, resize_fit_640x640')
    parser.add_argument('--data_cache_dir', help='Where to cache the dataset metadata')

    parser.add_argument('--pool_datasets', nargs='+',
                        default=['tc128_ce', 'dtb70', 'uav123', 'nuspro'],
                        help='datasets to combine for "pool" set')
    parser.add_argument('--pool_split', type=float, default=0.8, help='training fraction')


def add_tracker_config_args(parser):
    '''Tweak training behaviour.'''
    # parser.add_argument('--histograms', action='store_true',
    #                     help='generate histograms in summary (consumes space)')
    parser.add_argument('--use_queues', action='store_true',
                        help='enable queues for asynchronous data loading')

    parser.add_argument('--nosave', action='store_true', help='do not save checkpoints')
    parser.add_argument('--period_ckpt', type=int, default=10000,
                        help='period for saving checkpoints (number of steps)')
    parser.add_argument('--period_assess', type=int, default=10000,
                        help='period for running evaluation (number of steps)')
    parser.add_argument('--period_skip', type=int, default=0,
                        help='until this period skip evaluation (number of steps)')
    parser.add_argument('--period_summary', type=int, default=10,
                        help='period to update summary (number of steps)')
    parser.add_argument('--period_preview', type=int, default=100,
                        help='period to include images in summary (number of steps)')

    parser.add_argument('--visualize', action='store_true',
                        help='create video during evaluation')
    parser.add_argument('--keep_frames', action='store_true',
                        help='keep frames of video during evaluation')

    parser.add_argument('--gpu_manctrl', action='store_true', help='manual gpu management')
    parser.add_argument('--gpu_frac', type=float, default=0.4, help='fraction of gpu memory')
    parser.add_argument('--log_device_placement', action='store_true')


def add_train_args(parser):
    parser.add_argument('--train_dataset', type=json.loads, default='"ilsvrc_train"',
                        help='JSON to specify the training distribution')
    parser.add_argument('--val_dataset', type=json.loads, default='"ilsvrc_val"',
                        help='JSON to specify the validation distribution')

    parser.add_argument('--num_steps', type=int, default=200000,
                        help='number of gradient steps')

    parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--lr_params', type=json.loads, help='kwargs for learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer to train the model')
    parser.add_argument('--optimizer_params', type=json.loads,
                        help='kwargs for the optimizer constructor')
    parser.add_argument('--grad_clip', action='store_true', help='gradient clipping flag')
    parser.add_argument('--grad_clip_params', type=json.loads,
                        default='{"max_grad_norm": 5.0}',
                        help='threshold for gradient clipping')

    parser.add_argument('--use_gt_train', action='store_true',
                        help='use ground-truth during training')
    parser.add_argument('--gt_decay_rate', type=float, default=1e-6,
                        help='decay rate for gt_ratio')
    parser.add_argument('--min_gt_ratio', type=float, default=0.75,
                        help='lower bound for gt_ratio')

    parser.add_argument('--sampler_params', type=json.loads,
                        default={'kind': 'regular', 'freq': 10},
                        help='JSON to specify frame sampler')
    parser.add_argument('--augment_motion', action='store_true',
                        help='enable motion augmentation?')
    parser.add_argument('--motion_params', type=json.loads, default={},
                        help='JSON to specify motion augmentation')

    # parser.add_argument(
    #     '--color_augmentation', help='JSON string specifying color augmentation',
    #     type=json.loads, default={'brightness': False,
    #                               'contrast': False,
    #                               'grayscale': False})


def add_eval_args(parser):
    parser.add_argument('--eval_datasets', nargs='+', default=['ilsvrc_val', 'otb_50'],
                        help='dataset on which to evaluate tracker')
    # TODO: Only use TRE mode with "full" sampler?
    parser.add_argument('--eval_tre_num', type=int, default=3,
                        help='number of starting points for TRE mode')
    # TODO: Maybe remove "train" sampler, and this option?
    # parser.add_argument('--eval_samplers', nargs='+', default=['full'],
    #                     help='frame samplers to use during validation')
    parser.add_argument('--max_eval_videos', type=int, default=100,
                        help='max number of videos to evaluate')

    # parser.add_argument('--seed_global', type=int, default=9, help='random seed')

    # TODO: Should this be used somewhere?
    # parser.add_argument('--resize-online', dest='useresizedimg', action='store_false')


# def add_model_args(parser):
#     '''Args whose value is dependent on model.'''
#     # parser.add_argument('--loss_coeffs', type=json.loads, default='{}',
#     #                     help='list of losses to be used')
#
#     # parser.add_argument('--cnn_pretrain', action='store_true',
#     #                     help='specify if using pretrained model')
#     # parser.add_argument('--siamese_pretrain', action='store_true',
#     #                     help='specify if using pretrained model')
#     # parser.add_argument('--siamese_model_file', help='specify if using pretrained model')
