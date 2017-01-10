import pdb
import argparse

from opts           import Opts
from load_data      import load_data
from load_model     import load_model
from train          import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')
    parser.add_argument('--verbose', help='print arguments', action='store_true')
    parser.add_argument('--teststr', help='test str argument', type=str, default='')
    parser.add_argument('--testval', help='test int argument', type=int, default=0)
    args = parser.parse_args()

    # print help and args
    if args.verbose: print args
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)

    data_tr, data_va, data_te = load_data(o) # if able to load the whole dataset 

    model = load_model(o)

    '''
    net = train(model, data_tr, data_va, o)

    test(model, data_tr, data_va, o)
    '''

    pdb.set_trace()
