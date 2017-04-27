import pdb
import sys
import argparse
import functools
import json
import random
import tensorflow as tf

from opts           import Opts
import data
import model
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

    parser.add_argument(
            '--nunits', help='number of hidden units in rnn cell',
            type=int, default=256)
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=20)
    parser.add_argument(
            '--dropout_rnn', help='set dropout for rnn',
            action='store_true')

    parser.add_argument(
            '--dropout_cnn', help='dropout in cnn (only during train)',
            action='store_true')
    parser.add_argument(
            '--model_params', help='JSON string specifying model',
            type=json.loads, default={})

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
            type=float, default=1e-3)
    parser.add_argument(
            '--lr_decay_rate', help='geometric step for learning rate decay',
            type=float, default=1)
    parser.add_argument(
            '--lr_decay_steps', help='period for decaying learning rate',
            type=int, default=10000)
    parser.add_argument(
            '--wd', help='weight decay', type=float, default=0.0)

    parser.add_argument(
            '--sampler_params', help='JSON string specifying sampler',
            type=json.loads, default={'kind': 'regular', 'freq': 10})
    parser.add_argument(
            '--eval_datasets', nargs='+', help='dataset on which to evaluate tracker',
            type=str, default=['ILSVRC-train'])
    parser.add_argument(
            '--eval_samplers', nargs='+', help='',
            type=str, default=['custom'])

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
            '--gpu_device', help='set `CUDA_VISIBLE_DEVICES`',
            type=int, default=0)
    parser.add_argument(
            '--gpu_frac', help='fraction of gpu memory',
            type=float, default=0.4)

    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return args

def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace

if __name__ == "__main__":
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

    sampler_presets = {
        'full':   functools.partial(sample.sample, kind='full'),
        #'custom': functools.partial(sample.sample, kind='regular', freq=10,
        'custom': functools.partial(sample.sample, kind='regular', freq=o.sampler_params['freq'],
            ntimesteps=o.ntimesteps),
    }
    # Take all dataset-sampler combinations.
    eval_sets = {
        # TODO: This will use same set for every evaluation round? Good or bad?
        d+'-'+s: functools.partial(sampler_presets[s], datasets[d], max_sequences=100,
                                   generator=random.Random(o.seed_global))
        for d in o.eval_datasets
        for s in o.eval_samplers
    }

    # TODO: Set model_opts from command-line or JSON file?
    m = model.load_model(o, model_params=o.model_params)

    train.train(m, {'train': datasets['ILSVRC-train'], 'val': datasets['ILSVRC-val']},
                eval_sets, o, use_queues=o.use_queues)
