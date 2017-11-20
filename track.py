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

    parser.add_argument('--out_file', help='e.g. track.csv')
    parser.add_argument('--vot', action='store_true')

    parser.add_argument('--sequence_name', type=str, default='untitled')
    parser.add_argument('--init_rect', type=json.loads,
        help='e.g. {"xmin": 0.1, "ymin": 0.7, "xmax": 0.4, "ymax": 0.9}')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--image_format', type=str,
        help='e.g. sequence/%%06d.jpeg')

    parser.add_argument('--gpu_frac', type=float, default='1.0', help='fraction of gpu memory')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_dir', type=str)
    parser.add_argument('--vis_keep_frames', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.vot:
        global vot
        vot = __import__('vot', globals(), locals())

    o = Opts()
    o.update_by_sysarg(args=args)
    o.initialize()

    example = train._make_placeholders(o)
    create_model = model_pkg.load_model(o, model_params=o.model_params)
    model = create_model(train._whiten(train._guard_labels(example), o, stat=None),
                         summaries_collections=['summaries_model'])
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    sess = tf.Session(config=config)
    saver.restore(sess, args.model_file)

    # rect_pred, _ = evaluate.track(sess, example, model, sequence, use_gt=False, verbose=True,
    #     visualize=args.vis, vis_dir=args.vis_dir, save_frames=args.vis_keep_frames)
    tracker = evaluate.SimpleTracker(sess, example, model,
        verbose=True,
        sequence_name=args.sequence_name,
        visualize=args.vis,
        vis_dir=args.vis_dir,
        save_frames=args.vis_keep_frames)

    if args.vot:
        handle = vot.VOT('rectangle')
        vot_init_rect = handle.region()
        init_image = handle.frame()
        if not init_image:
            return
        image_size = Image.open(init_image).size
        init_rect = rect_from_vot(vot_init_rect, image_size)
    else:
        times = range(args.start, args.end+1)
        init_image = args.image_format % times[0]
        init_rect = args.init_rect

    tracker.start(init_image, rect_to_vec(init_rect))
    if args.vot:
        while True:
            image_t = handle.frame()
            if image_t is None:
                break
            pred_t = tracker.next(image_t)
            handle.report(rect_to_vot(rect_from_vec(pred_t), image_size))
    else:
        pred = []
        for t in times[1:]:
            image_t = args.image_format % t
            pred_t = tracker.next(image_t)
            pred.append(pred_t)
    tracker.end()

    if args.out_file is not None:
        # Write to file.
        # pred = np.asarray(pred).tolist()
        with open(args.out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'xmin', 'ymin', 'xmax', 'ymax'])
            for t, rect_t in zip(times[1:], pred):
                writer.writerow([t] + map(lambda x: '{:.6g}'.format(x), rect_t))
            # json.dump(pred, sys.stdout)


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


def rect_from_vot(rect, image_size):
    width, height = image_size
    return {
        'xmin': float(rect.x - 1) / width,
        'ymin': float(rect.y - 1) / height,
        'xmax': float(rect.x + rect.width - 1) / width,
        'ymax': float(rect.y + rect.height - 1) / height,
    }

def rect_to_vot(rect, image_size):
    width, height = image_size
    return vot.Rectangle(**{
        'x': rect['xmin'] * width + 1,
        'y': rect['ymin'] * height + 1,
        'width': (rect['xmax'] - rect['xmin']) * width,
        'height': (rect['ymax'] - rect['ymin']) * height,
    })

def rect_to_vec(rect):
    return np.asfarray([rect[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']])

def rect_from_vec(vec):
    return {k: vec[i] for i, k in enumerate(['xmin', 'ymin', 'xmax', 'ymax'])}


if __name__ == '__main__':
    main()
