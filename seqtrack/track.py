import pdb
import tensorflow as tf
import argparse
import csv
import json
import time
import os
import numpy as np
from PIL import Image

from seqtrack.opts import Opts
from seqtrack import evaluate
from seqtrack import geom_np
from seqtrack import data
# from seqtrack import model as model_pkg
from seqtrack import draw
from seqtrack import graph


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

    example = graph.make_placeholders(o.ntimesteps, (o.frmsz, o.frmsz))
    create_model = model_pkg.load_model(o, model_params=o.model_params)
    model = create_model(graph.whiten(graph.guard_labels(example), stat=None),
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
    # parser.add_argument(
    #         '--frmsz', help='size of a square image', type=int, default=257)
    parser.add_argument(
            '--imwidth', type=int, default=640, help='image resolution')
    parser.add_argument(
            '--imheight', type=int, default=360, help='image resolution')
    parser.add_argument(
            '--ntimesteps', help='number of time steps for rnn',
            type=int, default=1)

    parser.add_argument(
            '--model', help='model!',
            type=str, default='')
    parser.add_argument(
            '--model_params', help='JSON string specifying model',
            type=json.loads, default={})
    # parser.add_argument(
    #         '--cnn_model', help='pretrained CNN model',
    #         type=str, default='custom')

    # JV: Move these to the model.
    # parser.add_argument(
    #         '--search_scale', help='size of search space relative to target',
    #         type=int, default=4)
    # parser.add_argument(
    #         '--target_scale', help='size of context relative to target',
    #         type=int, default=1)
    # parser.add_argument(
    #         '--perspective', help='ic: image-centric, oc: object-centric',
    #         type=str, default='oc')
    # parser.add_argument(
    #         '--aspect_method', help='method for fixing aspect ratio',
    #         type=str, default='stretch',
    #         choices=['stretch', 'area', 'perimeter'])

    # parser.add_argument(
    #         '--th_prob_stay', help='threshold probability to stay movement',
    #         type=float, default=0.0)


if __name__ == '__main__':
    main()
