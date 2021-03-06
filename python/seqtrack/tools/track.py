from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import csv
import datetime
import json
import time
import os
import numpy as np
from PIL import Image

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import track
from seqtrack import train
from seqtrack import geom_np
from seqtrack import data
from seqtrack import models
from seqtrack.models import itermodel
import trackdat


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()))
    if args.vot:
        global vot
        vot = __import__('vot', globals(), locals())

    if args.model_params and args.model_params_file:
        raise RuntimeError('cannot specify both model_params and model_params_file')
    model_params = {}
    if args.model_params_file:
        with open(args.model_params_file, 'r') as f:
            model_params = json.load(f)
    elif args.model_params:
        model_params = args.model_params

    create_model_fn = models.BY_NAME[args.model_name]
    model_fn = create_model_fn(mode=tf.estimator.ModeKeys.PREDICT, **model_params)
    # TODO: Ensure that the `track` instance does not include ops such as bnorm and summaries.
    # Feed image files here:
    example_with_files = train._make_iter_placeholders(batch_size=1)
    # Evaluate model_fn on these features:
    example = train._load_images_iter(example_with_files)
    with tf.variable_scope('model', reuse=False) as scope:
        tracker = itermodel.TrackerFeed(
            example_with_files,
            itermodel.instantiate_iter_feed(model_fn, example,
                                            run_opts=train._make_run_opts_tracking(),
                                            scope=scope))

    saver = tf.train.Saver()

    sequence_name = (args.sequence_name or
                     'untitled_{}'.format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')))

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.model_file)
        # rect_pred, _ = track.track(sess, example, model, sequence, use_gt=False, verbose=True,
        #                            visualize=args.vis, vis_dir=args.vis_dir, keep_frames=args.vis_keep_frames)

        # tracker = track.SimpleTracker(
        #     sess, model_inst, verbose=True, sequence_name=sequence_name,
        #     visualize=args.vis, vis_dir=args.vis_dir, keep_frames=args.vis_keep_frames)
        # tracker.warmup()

        if args.vot:
            logger.debug('try to obtain VOT handle')
            handle = vot.VOT('rectangle')
            logger.debug('obtained VOT handle')
            init_rect_vot = handle.region()
            logger.debug('init rectangle: %s', init_rect_vot)
            init_image = handle.frame()
            if not init_image:
                return
            imwidth, imheight = Image.open(init_image).size
            logger.debug('imwidth=%d imheight=%s', imwidth, imheight)
            init_rect = from_vot(init_rect_vot, imwidth=imwidth, imheight=imheight)
        else:
            if args.images_file:
                image_files = read_lines(args.images_file)
            else:
                times = range(args.start, args.end + 1)
                image_files = [args.image_format % t for t in times]
            init_rect = args.init_rect
            init_image = image_files[0]
            imwidth, imheight = Image.open(init_image).size

        tracker.start(sess, {
            'image': {'file': [init_image]},
            'aspect': [imwidth / imheight],
            'rect': [rect_to_vec(init_rect)],
        })
        if args.vot:
            while True:
                image_t = handle.frame()
                if image_t is None:
                    break
                # Untested:
                outputs_t = tracker.next(sess, {'image': {'file': [image_t]}})
                rect_vot = to_vot(rect_from_vec(outputs_t['rect'][0]),
                                  imwidth=imwidth, imheight=imheight)
                logger.debug('report rectangle: %s', rect_vot)
                handle.report(rect_vot, outputs_t['score'])
        else:
            pred = []
            for image_t in image_files[1:]:
                pred_t = tracker.next(sess, {'image': {'file': [image_t]}})
                pred.append(pred_t['rect'][0])
        # info = tracker.end()
        # logger.info('tracker speed (eval): %.3g', info['num_frames'] / info['duration_eval'])
        # logger.info('tracker speed (with_load): %.3g',
        #              info['num_frames'] / info['duration_with_load'])
        # logger.info('tracker speed (real): %.3g', info['num_frames'] / info['duration_real'])

    if args.out_file is not None:
        # Write to file.
        # pred = np.asarray(pred).tolist()
        with open(args.out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'xmin', 'ymin', 'xmax', 'ymax'])
            for im_file, rect_t in zip(image_files[1:], pred):
                writer.writerow([im_file] + list(map(lambda x: '{:.6g}'.format(x), rect_t)))


def parse_arguments():
    parser = argparse.ArgumentParser(description='run tracker on one sequence')
    parser.add_argument('--loglevel', default='warning', help='debug, info, warning')
    # add_tracker_arguments(parser)
    app.add_instance_arguments(parser)

    # TODO: Move to app?
    parser.add_argument('--model_name', default='siamfc', help='String to identify model type')
    parser.add_argument('--model_params', type=json.loads, help='JSON string')
    parser.add_argument('--model_params_file', type=str, help='JSON file')

    parser.add_argument('model_file',
                        help='e.g. iteration-100000; where iteration-100000.index exists')

    parser.add_argument('--out_file', help='e.g. track.csv')
    parser.add_argument('--vot', action='store_true')

    parser.add_argument('--sequence_name', type=str)
    parser.add_argument('--init_rect', type=json.loads,
                        help='e.g. {"xmin": 0.1, "ymin": 0.7, "xmax": 0.4, "ymax": 0.9}')
    parser.add_argument('--images_file', type=str,
                        help='Text file containing list of images; overrides image_format')
    parser.add_argument('--image_format', type=str, help='e.g. sequence/%%06d.jpeg')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)

    # parser.add_argument('--gpu_frac', type=float, default='1.0', help='fraction of gpu memory')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_dir', type=str)
    parser.add_argument('--vis_keep_frames', action='store_true')

    return parser.parse_args()


def from_vot(r, imheight, imwidth):
    # Matlab images are rendered in the continuous interval [0.5, size + 0.5].
    # To move them to the range [0, size] we must subtract 0.5.
    return trackdat.dataset.make_rect_pix(
        xmin=r.x - 0.5, xmax=r.x - 0.5 + r.width,
        ymin=r.y - 0.5, ymax=r.y - 0.5 + r.height,
        imheight=imheight,
        imwidth=imwidth)


def to_vot(r, imheight, imwidth):
    return vot.Rectangle(
        x=(r['xmin'] * imwidth + 0.5),
        y=(r['ymin'] * imheight + 0.5),
        width=((r['xmax'] - r['xmin']) * imwidth),
        height=((r['ymax'] - r['ymin']) * imheight))


def rect_to_vec(rect):
    min_pt = (rect['xmin'], rect['ymin'])
    max_pt = (rect['xmax'], rect['ymax'])
    return geom_np.make_rect(min_pt, max_pt)


def rect_from_vec(vec):
    min_pt, max_pt = geom_np.rect_min_max(vec)
    xmin, ymin = min_pt.tolist()
    xmax, ymax = max_pt.tolist()
    return dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def read_lines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return [x.strip() for x in lines]


if __name__ == '__main__':
    main()
