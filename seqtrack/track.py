import pdb
import tensorflow as tf
import argparse
import csv
import json
import time
import os
import numpy as np
from PIL import Image

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

    parser.add_argument('--out_file', help='e.g. track.csv')
    parser.add_argument('--vot', action='store_true')

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
    if args.vot:
        global vot
        vot = __import__('vot', globals(), locals())

    example = graph.make_placeholders(args.ntimesteps, (args.frmsz, args.frmsz))
    create_model = model_pkg.load_model(args, model_params=args.model_params)
    model = create_model(graph.whiten(graph.guard_labels(example), stat=None),
                         summaries_collections=['summaries_model'])
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
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_frac
    with tf.Session(config=config) as sess:
        saver.restore(sess, args.model_file)
        # rect_pred, _ = evaluate.track(sess, example, model, sequence, use_gt=False, verbose=True,
        #                               visualize=args.vis, vis_dir=args.vis_dir, save_frames=args.vis_keep_frames)

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
                writer.writerow([t] + map(lambda x: '{:.6g}'.format(x), rect_t))


if __name__ == '__main__':
    main()
