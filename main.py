import pdb
import argparse

from opts           import Opts
#from load_data      import load_data
from data           import Data_moving_mnist
from load_model     import load_model
from train          import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')
    parser.add_argument('--verbose', help='print arguments', 
            action='store_true')
    parser.add_argument('--mode', help='experiment mode (train, test)', 
            type=str, default='train' )
    parser.add_argument('--dropout_rnn', help='set dropout for rnn', 
            action='store_true')
    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return parser.parse_args()

if __name__ == "__main__":
    # parameter settings
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)

    #TODO: load_data deprecated
    #data_tr, data_va, data_te = load_data(o) # if able to load the whole dataset 

    # Note that depending on the dataset the loading approach can be different
    if o.dataset == 'moving_mnist': 
        loader = Data_moving_mnist(o)
    elif o.dataset =='another_dataset': 
        raise ValueError('dataset not implemented yet')
    else: 
        raise ValueError('wrong dataset')

    model = load_model(o, loader, is_training=True if o.mode=='train' else False)

    train(model, loader, o)

    #TODO: write test model routine; name scope or op scope might be required
    #test(model, data_tr, data_va, o)

    pdb.set_trace()
