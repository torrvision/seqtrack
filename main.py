import pdb
import argparse

from opts import Opts

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
    o.update_sysarg(args=args)

