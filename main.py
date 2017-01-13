import pdb
import argparse

from opts           import Opts
#from load_data      import load_data
#from data           import Data_moving_mnist
from data           import load_data
from load_model     import load_model
from train          import train
from test           import test


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')

    parser.add_argument(
            '--verbose', help='print arguments', 
            action='store_true')
    parser.add_argument(
            '--mode', help='mode (train, test)', 
            type=str, default='train')
    parser.add_argument(
            '--debugmode', help='used for debugging', 
            action='store_true')

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
            '--resume_model', help='model to resume',
            type=str)

    parser.add_argument(
            '--dropout_rnn', help='set dropout for rnn', 
            action='store_true')
    parser.add_argument(
            '--nepoch', help='number of epochs', 
            type=int, default=1)

    parser.add_argument(
            '--gpu_manctrl', help='control gpu memory manual', 
            action='store_true')
    parser.add_argument(
            '--gpu_frac', help='fraction of gpu memory', 
            type=float, default=0.5)

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
    
    loader = load_data(o)

    m = load_model(o, loader, is_training=True if o.mode=='train' else False)

    if o.mode == 'train':
        train(m, loader, o)
    elif o.mode == 'test':
        test(m, loader, o)

    pdb.set_trace()


