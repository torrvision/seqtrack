import pdb
import argparse

from opts           import Opts
import data
import model
from train          import train
from test           import test



def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')

    parser.add_argument(
            '--verbose', help='print arguments', 
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
            '--nosave', help='no need to save results?', 
            action='store_true') 
    parser.add_argument(
            '--restore', help='to restore a pretrained', 
            action='store_true')
    parser.add_argument(
            '--restore_model', help='model to restore', 
            type=str) 
    parser.add_argument(
            '--resume', help='to resume training',
            action='store_true')
    parser.add_argument(
            '--resume_data', help='model to resume',
            type=str)

    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=30)
    parser.add_argument(
            '--dropout_rnn', help='set dropout for rnn', 
            action='store_true')
    parser.add_argument(
            '--nepoch', help='number of epochs', 
            type=int, default=1)
    parser.add_argument(
            '--batchsz', help='batch size', 
            type=int, default=1)
    parser.add_argument(
            '--optimizer', help='optimizer to train the model',
            type=str, default='sgd')
    parser.add_argument(
            '--lr', help='learning rate', 
            type=float, default=0.001)
    parser.add_argument(
            '--lr_update', help='adaptive learning rate', 
            action='store_true')
    parser.add_argument(
            '--wd', help='weight decay', 
            type=float, default=0.0)

    parser.add_argument(
            '--device_number', help='gpu number for manual assignment', 
            type=int, default=0)
    parser.add_argument(
            '--gpu_manctrl', help='control gpu memory manual', 
            action='store_true')
    parser.add_argument(
            '--gpu_frac', help='fraction of gpu memory', 
            type=float, default=0.4)

    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return parser.parse_args()

if __name__ == "__main__":
    # parameter settings
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    loader = data.load_data(o)

    m = model.load_model(o, loader)

    if o.mode == 'train':
        train(m, loader, o)
    elif o.mode == 'test':
        test(m, loader, o)
    else:
        raise ValueError('Currently, only either train or test mode supported')

    print '**Saved in the following directory.'
    print o.path_save if not o.nosave else o.path_save_tmp

