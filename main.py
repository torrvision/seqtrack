import pdb
import argparse
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
            '--mode', help='choose mode (train, test)',
            type=str, default='')
    parser.add_argument(
            '--debugmode', help='used for debugging',
            action='store_true')

    parser.add_argument(
            '--dataset', help='specify the name of dataset',
            type=str, default='')
    parser.add_argument(
            '--trainsplit', help='specify the split of train dataset (ILSVRC)',
            type=int, default=9)

    parser.add_argument(
            '--frmsz', help='size of a square image', type=int, default=100)
    # NOTE: (NL) any reason to have two arguments for this option?
    parser.add_argument('--resize-online', dest='useresizedimg', action='store_false')
    parser.set_defaults(useresizedimg=True)

    parser.add_argument(
            '--model', help='model!',
            type=str, default='')
    parser.add_argument(
            '--losses', nargs='+', help='list of losses to be used',
            type=str) # example [l1, iou]

    parser.add_argument(
            '--cell_type', help='rnn cell type',
            type=str, default='LSTM')
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
            '--yprev_mode', help='way of using y_prev',
            type=str, default='')
    parser.add_argument(
            '--pass_yinit', help='pass gt y instead pred y during training',
            action='store_true')

    parser.add_argument(
            '--dropout_cnn', help='dropout in cnn (only during train)',
            action='store_true')

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
    # parser.add_argument(
    #         '--lr', help='learning rate',
    #         type=float, default=0.0001)
    # parser.add_argument(
    #         '--lr_update', help='adaptive learning rate',
    #         action='store_true')
    parser.add_argument(
            '--wd', help='weight decay',
            type=float, default=1e-3)

    parser.add_argument(
            '--path_data_home', help='location of datasets',
            type=str, default='./data')
    parser.add_argument(
            '--nosave', help='no need to save results?',
            action='store_true')
    parser.add_argument(
            '--restore', help='to load a pretrained model (for test)',
            action='store_true')
    parser.add_argument(
            '--restore_model', help='model to restore',
            type=str)
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

if __name__ == "__main__":
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    dataset = data.load_data(o)
    ilsvrc_train = data.Data_ILSVRC('train', o)
    ilsvrc_val   = data.Data_ILSVRC('val', o)
    otb50        = data.Data_OTB('OTB-50', o)
    otb100       = data.Data_OTB('OTB-100', o)
    val_sets = {
        'ILSVRC-train-full':
            lambda: random.sample(list(sample.sample(ilsvrc_train, seqtype='full')), 100),
        # 'ILSVRC-val-full':
        #     lambda: random.sample(list(sample.sample(ilsvrc_val, seqtype='full')), 100),
        'ILSVRC-train-sample':
            lambda: random.sample(list(sample.sample(ilsvrc_train, ntimesteps=o.ntimesteps, seqtype='sampling')), 100),
        # 'ILSVRC-val-sample':
        #     lambda: random.sample(list(sample.sample(ilsvrc_val, ntimesteps=o.ntimesteps, seqtype='sampling')), 100),
        #'OTB-50-full':
        #    lambda: sample.sample(otb50, seqtype='full'),
        #'OTB-100-full':
        #    lambda: sample.sample(otb100, seqtype='full'),
        #'OTB-50-sample':
        #    lambda: sample.sample(otb50, ntimesteps=o.ntimesteps, seqtype='sampling'),
        #'OTB-100-sample':
        #    lambda: sample.sample(otb100, ntimesteps=o.ntimesteps, seqtype='sampling'),
    }

    m = lambda inputs: model.load_model(inputs, o)

    assert(o.mode == 'train')
    train.train(m, dataset, val_sets, o)
