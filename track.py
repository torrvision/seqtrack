import pdb
import tensorflow as tf
import argparse
import csv
import json
import time
import os
import numpy as np
from PIL import Image

from opts import Opts
import evaluate
import geom_np
import data
import model as model_pkg
import draw
import train


def parse_arguments():
    parser = argparse.ArgumentParser(description='run tracker on one sequence')
    add_tracker_arguments(parser)
    parser.add_argument('model_file',
        help='e.g. iteration-100000; where iteration-100000.index exists')
    parser.add_argument('out_file', help='e.g. track.csv')

    parser.add_argument('--sequence_name', type=str, default='untitled')
    parser.add_argument('--init_rect', type=json.loads, required=True,
        help='e.g. {"xmin": 0.1, "ymin": 0.7, "xmax": 0.4, "ymax": 0.9}')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--image_format', type=str, required=True,
        help='e.g. sequence/%%06d.jpeg')

    parser.add_argument('--gpu_frac', type=float, default='1.0', help='fraction of gpu memory')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_dir', type=str)
    parser.add_argument('--vis_keep_frames', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()
    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    # Load sequence from args.
    frames = range(args.start, args.end+1)
    sequence = {}
    sequence['video_name'] = args.sequence_name
    sequence['image_files'] = [args.image_format % i for i in frames]
    sequence['viewports'] = [geom_np.unit_rect() for _ in frames]
    init_rect = np.asfarray([args.init_rect[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']])
    sequence['labels'] = [init_rect if i == args.start else geom_np.unit_rect() for i in frames]
    sequence['label_is_valid'] = [i == args.start for i in frames]
    width, height = Image.open(args.image_format % args.start).size
    sequence['aspect'] = float(width) / height
    # sequence['original_image_size']

    example = train._make_placeholders(o)
    create_model = model_pkg.load_model(o, model_params=o.model_params)
    model = create_model(train._whiten(train._guard_labels(example), o, stat=None),
                         summaries_collections=['summaries_model'])
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.model_file)
        rect_pred, _ = evaluate.track(sess, example, model, sequence, use_gt=False, verbose=True,
            visualize=args.vis, vis_dir=args.vis_dir, save_frames=args.vis_keep_frames)

    rect_pred = np.asarray(rect_pred).tolist()
    with open(args.out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'xmin', 'ymin', 'xmax', 'ymax'])
        times = range(args.start, args.end+1)
        for t, rect_t in zip(times[1:], rect_pred):
            writer.writerow([t] + map(lambda x: '{:.6g}'.format(x), rect_t))
        # json.dump(rect_pred, sys.stdout)


def add_tracker_arguments(parser):
    parser.add_argument(
            '--frmsz', help='size of a square image', type=int, default=257)
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=1)

    parser.add_argument(
            '--model', help='model!',
            type=str, default='')
    parser.add_argument(
            '--model_params', help='JSON string specifying model',
            type=json.loads, default={})
    parser.add_argument(
            '--cnn_model', help='pretrained CNN model',
            type=str, default='custom')
    parser.add_argument(
            '--nunits', help='number of hidden units in rnn cell',
            type=int, default=256)

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
            '--th_prob_stay', help='threshold probability to stay movement',
            type=float, default=0.0)


# def test(m, loader, o, dstype, fulllen=False, draw_track=False):
#     '''
#     Note that it is considered that this wrapper serves a test routine with a 
#     completely trained model. If you want a on-the-fly evaluations during 
#     training, consider carefully which session you will use.
#     '''
# 
#     saver = tf.train.Saver()
#     with tf.Session(config=o.tfconfig) as sess:
#         saver.restore(sess, o.restore_model)
#         t_start = time.time()
#         results = evaluate(sess, m, loader, o, dstype, fulllen=fulllen)
# 
#     # save results
#     if fulllen:
#         savedir = os.path.join(o.path_base, 
#             os.path.dirname(o.restore_model)[:-7] 
#             + '/evaluations_{}_fulllen'.format(o.dataset))
#     else:
#         savedir = os.path.join(o.path_base, 
#             os.path.dirname(o.restore_model)[:-7] 
#             + '/evaluations_{}_RNNlen'.format(o.dataset))
#     if not os.path.exists(savedir): os.makedirs(savedir)
# 
#     # save subset of results (due to memory)
#     results_partial = {}
#     results_partial['iou_mean'] = results['iou_mean']
#     results_partial['success_rates'] = results['success_rates']
#     results_partial['auc'] = results['auc']
#     results_partial['cle_mean'] = results['cle_mean']
#     results_partial['precision_rates'] = results['precision_rates']
#     results_partial['cle_representative'] = results['cle_representative']
#     np.save(savedir + '/results_partial.npy', results_partial)
# 
#     # print
#     print '-------------------------------------------------------------------'
#     print 'Evaluation finished (time: {0:.3f}).'.format(time.time()-t_start)
#     print 'Model: {}'.format(o.model) 
#     print 'dataset: {}(''{}'')'.format(o.dataset, dstype)
#     print 'iou_mean: {0:.3f}'.format(results['iou_mean'])
#     print 'success_rates: [%s]' % ', '.join(
#             '%.3f' % val for val in results['success_rates'])
#     print 'auc: {0:.3f}'.format(results['auc'])
#     print 'cle_mean: {0:.3f}'.format(results['cle_mean'])
#     print 'precision_rates: [%s]' % ', '.join(
#             '%.3f' % val for val in results['precision_rates'])
#     print 'cle_representative: {0:.3f}'.format(results['cle_representative'])
#     print 'results and plots are saved at {}'.format(savedir)
#     print '-------------------------------------------------------------------'
# 
#     # Plot success and precision plots
#     draw.plot_successplot(results['success_rates'], results['auc'], o, savedir)
#     draw.plot_precisionplot(
#         results['precision_rates'], results['cle_representative'], o, savedir)
# 
#     # VISUALIZE TRACKING RESULTS 
#     if draw_track:
#         draw.show_track_results_fl(results, loader, o, savedir)


# if __name__ == '__main__':
#     '''Test script
#     Provide the followings:
#         - CUDA_VISIBLE_DEVICES
#         - dataset (e.g., bouncing_mnist)
#         - model (e.g., rnn_attention_st)
#         - restore_model (e.g., ***.ckpt)
#         - ntimesteps
#         - yprev_mode
#         - losses
#         - (optionally) batchsz
#     Note that if provided option is inconsitent with the trained model 
#     (e.g., ntimesteps), it will fail to restore the model.
#     '''
#     args = parse_arguments()
#     o = Opts()
#     o.update_by_sysarg(args=args)
#     o._set_gpu_config()
#     o._set_dataset_params()
# 
#     # NOTE: other options can be passed to args or simply put here.
#     if o.dataset == 'ILSVRC':
#         dstype = 'val'
# 
#     loader = data.load_data(o)
# 
#     m = Model(o)
# 
#     test_fl = True
# 
#     # Case: T-length sequences
#     if not test_fl:
#         dstype = 'test' 
#         test(m, loader, o, dstype=dstype)
# 
#     # Case: Full-length sequences
#     else:
#         dstype = 'val' # ILSVRC
#         test(m, loader, o, dstype=dstype, fulllen=True, draw_track=args.draw_track)

if __name__ == '__main__':
    main()
