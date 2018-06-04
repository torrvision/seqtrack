from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import csv
import json
import time
import os
import numpy as np
from PIL import Image

import logging
logger = logging.getLogger(__name__)

from seqtrack import app
from seqtrack import evaluate
from seqtrack import geom_np
from seqtrack import data
from seqtrack import draw
from seqtrack import graph
from seqtrack.models.itermodel import ModelFromIterModel
from seqtrack.models.siamfc import SiamFC


def parse_arguments():
    parser = argparse.ArgumentParser(description='run tracker on one sequence')
    parser.add_argument('--loglevel', default='warning', help='debug, info, warning')
    # add_tracker_arguments(parser)
    app.add_instance_arguments(parser)

    # TODO: Move to app?
    parser.add_argument('--model_params', type=json.loads, help='JSON string')
    parser.add_argument('--model_params_file', type=str, help='JSON file')

    parser.add_argument('model_file',
                        help='e.g. iteration-100000; where iteration-100000.index exists')

    parser.add_argument('--out_file', help='e.g. track.csv')
    parser.add_argument('--vot', action='store_true')

    parser.add_argument('--sequence_name', type=str, default='untitled')
    parser.add_argument('--init_rect', type=json.loads,
                        help='e.g. {"xmin": 0.1, "ymin": 0.7, "xmax": 0.4, "ymax": 0.9}')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--image_format', type=str, help='e.g. sequence/%%06d.jpeg')

    # parser.add_argument('--gpu_frac', type=float, default='1.0', help='fraction of gpu memory')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_dir', type=str)
    parser.add_argument('--vis_keep_frames', action='store_true')

    return parser.parse_args()


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

    model_spec = ModelFromIterModel(SiamFC(**model_params))

    example, run_opts = graph.make_placeholders(
        args.ntimesteps, (args.imheight, args.imwidth))
    example_proc = graph.whiten(example)
    with tf.name_scope('model'):
        # TODO: Can ignore image_summaries here?
        outputs, losses, init_state, final_state = model_spec.instantiate(
            example_proc, run_opts, enable_loss=False)
    model_inst = graph.ModelInstance(
        example, run_opts, outputs, init_state, final_state,
        batchsz=outputs['y'].shape.as_list()[0], ntimesteps=args.ntimesteps,
        imheight=args.imheight, imwidth=args.imwidth)

    saver = tf.train.Saver()

    # # Load sequence from args.
    # frames = range(args.start, args.end + 1)
    # sequence = {}
    # sequence['video_name'] = args.sequence_name
    # sequence['image_files'] = [args.image_format % i for i in frames]
    # sequence['viewports'] = [geom_np.unit_rect() for _ in frames]
    # init_rect = np.asfarray([args.init_rect[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']])
    # sequence['labels'] = [init_rect if i == args.start else geom_np.unit_rect() for i in frames]
    # sequence['label_is_valid'] = [i == args.start for i in frames]
    # width, height = Image.open(args.image_format % args.start).size
    # sequence['aspect'] = float(width) / height
    # # sequence['original_image_size']

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.model_file)
        # rect_pred, _ = evaluate.track(sess, example, model, sequence, use_gt=False, verbose=True,
        #                               visualize=args.vis, vis_dir=args.vis_dir, keep_frames=args.vis_keep_frames)

        tracker = evaluate.SimpleTracker(
            sess, model_inst, verbose=True, sequence_name=args.sequence_name,
            visualize=args.vis, vis_dir=args.vis_dir, keep_frames=args.vis_keep_frames)

        if args.vot:
            logger.debug('try to obtain VOT handle')
            handle = vot.VOT('rectangle')
            logger.debug('obtained VOT handle')
            vot_init_rect = handle.region()
            init_image = handle.frame()
            if not init_image:
                return
            image_size = Image.open(init_image).size
            init_rect = rect_from_vot(vot_init_rect, image_size)
        else:
            times = range(args.start, args.end + 1)
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
            # times = range(args.start, args.end + 1)
            for t, rect_t in zip(times[1:], pred):
                writer.writerow([t] + list(map(lambda x: '{:.6g}'.format(x), rect_t)))


if __name__ == '__main__':
    main()
