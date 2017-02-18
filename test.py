import pdb
import tensorflow as tf
import argparse
import time
import os
import numpy as np

from opts import Opts
from evaluate import evaluate
import data
import model
import draw


def parse_arguments():
    parser = argparse.ArgumentParser(description='rnn tracking - main script')

    parser.add_argument(
            '--verbose', help='print arguments',
            action='store_true')
    parser.add_argument(
            '--debugmode', help='used for debugging',
            action='store_true')

    parser.add_argument(
        '--dataset', help='specify the name of dataset',
        type=str, default='')

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
            '--model', help='model!',
            type=str, default='')

    parser.add_argument(
            '--nunits', help='number of hidden units in rnn cell',
            type=int, default=300)
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=20)
    parser.add_argument(
            '--yprev_mode', help='way of using y_prev',
            type=str, default='')

    parser.add_argument(
            '--batchsz', help='batch size',
            type=int, default=1)
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


def test(m, loader, o, dstype):
    '''
    Note that it is considered that this wrapper serves a test routine with a 
    completely trained model. If you want a on-the-fly evaluations during 
    training, consider carefully which session you will use.
    '''

    saver = tf.train.Saver()
    with tf.Session(config=o.tfconfig) as sess:
        saver.restore(sess, o.restore_model)
        t_start = time.time()
        results = evaluate(sess, m, loader, o, dstype) 

    # save results
    savedir = os.path.join(
            o.path_base, os.path.dirname(o.restore_model)[:-7] + '/evaluations')
    if not os.path.exists(savedir): os.makedirs(savedir)

    # save subset of results (due to memory)
    results_partial = {}
    results_partial['iou_mean'] = results['iou_mean']
    results_partial['success_rates'] = results['success_rates']
    results_partial['auc'] = results['auc']
    results_partial['cle_mean'] = results['cle_mean']
    results_partial['precision_rates'] = results['precision_rates']
    results_partial['cle_representative'] = results['cle_representative']
    np.save(savedir + '/results_partial.npy', results_partial)

    # Plot success and precision plots
    draw.plot_successplot(results['success_rates'], results['auc'], o, savedir)
    draw.plot_precisionplot(
            results['precision_rates'], results['cle_representative'], 
            o, savedir)

    # VISUALIZE TRACKING RESULTS 
    # TODO: not available yet.. can't hold all examples in 'results'. 

    # print
    print '-------------------------------------------------------------------'
    print 'Evaluation finished (time: {0:.3f}).'.format(time.time()-t_start)
    print 'Model: {}'.format(o.model) 
    print 'dataset: {}(''{}'')'.format(o.dataset, dstype)
    print 'iou_mean: {0:.3f}'.format(results['iou_mean'])
    print 'success_rates: [%s]' % ', '.join(
            '%.3f' % val for val in results['success_rates'])
    print 'auc: {0:.3f}'.format(results['auc'])
    print 'cle_mean: {0:.3f}'.format(results['cle_mean'])
    print 'precision_rates: [%s]' % ', '.join(
            '%.3f' % val for val in results['precision_rates'])
    print 'cle_representative: {0:.3f}'.format(results['cle_representative'])
    print 'results and plots are saved at {}'.format(savedir)
    print '-------------------------------------------------------------------'


if __name__ == '__main__':
    '''Test script
    Provide the followings:
        - CUDA_VISIBLE_DEVICES
        - dataset (e.g., bouncing_mnist)
        - model (e.g., rnn_attention_st)
        - restore_model (e.g., ***.ckpt)
        - gpu_manctrl 
        - ntimesteps
        - (optionally) batchsz
    Note that if provided option is inconsitent with the trained model 
    (e.g., ntimesteps), it will fail to restore the model.
    '''
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)
    o._set_gpu_config()
    o._set_dataset_params()

    loader = data.load_data(o)
    m = model.load_model(o)

    # NOTE: depending on dstype, can load a 'longer' or 'full' test sequences
    dstype = 'test' 
    test(m, loader, o, dstype=dstype)

