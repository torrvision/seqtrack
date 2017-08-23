'''Command-line utility to run tracker.'''

import argparse
import json
import numpy as np
import tensorflow as tf

import data
import evaluate
import sample
import train
import model as model_pkg
import model_graph
import motion


def main():
    parser = argparse.ArgumentParser(description='run tracker on dataset')
    parser.add_argument('--frmsz', type=int, default=241,
        help='size of a square image')
    parser.add_argument('--ntimesteps', type=int, default=20,
        help='number of steps unrolled in RNN')
    parser.add_argument('--output_mode', default='rectangle',
        help='model output type (rectangle, score_map, rect_map)')
    # Model options
    # The options can be different from the training options.
    # However, the model must still be compatible (i.e. possible to load).
    # parser.add_argument('--model', default='')
    parser.add_argument('--model_params', type=json.loads, default={},
        help='JSON string to specify model')
    # Dataset options
    parser.add_argument('--dataset', default='OTB-50',
        help='dataset on which to evaluate tracker')
    parser.add_argument('--path_data_home', default='./data',
        help='location of datasets')
    parser.add_argument('--sample_frames_params', type=json.loads,
        help='JSON string to specify frame sampler',
        # default=json.dumps({'kind': 'regular', 'freq': 10}))
        default=json.dumps({'kind': 'full'}))
    parser.add_argument('--augment_motion_params', type=json.loads,
        help='JSON string to specify motion augmentation',
        default=json.dumps({'kind': 'none'}))
    # Paths.
    parser.add_argument('--path_ckpt', default='./ckpt')
    args = parser.parse_args()

    model = model_pkg.SimpleSearch(
        ntimesteps=args.ntimesteps,
        frmsz=args.frmsz,
        batchsz=None,
        output_mode=args.output_mode,
        stat={'mean': 0.5, 'std': 1.0},
        weight_decay=0.0, # No effect at test time.
        **args.model_params)

    with tf.name_scope('input'):
        example = model_graph.make_example_placeholders(
            batchsz=None,
            ntimesteps=args.ntimesteps,
            frmsz=args.frmsz)
        run_opts = model_graph.make_option_placeholders()

    prediction_crop, window, prediction, init_state, final_state = model_graph.process_sequence(
        model, example, run_opts,
        batchsz=None, ntimesteps=args.ntimesteps, frmsz=args.frmsz)

    eval_model = evaluate.Model(
        batch_size=None,
        image_size=(args.frmsz, args.frmsz),
        sequence_len=args.ntimesteps,
        example=example,
        run_opts=run_opts,
        prediction_crop=prediction_crop,
        window=window,
        prediction=prediction,
        init_state=init_state,
        final_state=final_state,
    )

    dataset = data.load_dataset(args.dataset,
        frmsz=args.frmsz,
        path_data_home=args.path_data_home,
        path_aux='./aux',
        path_stat='./stat')

    sequences = sample.epoch(
        dataset=dataset,
        rand=np.random.RandomState(),
        sample_frames=sample.make_frame_sampler(
            dataset=dataset,
            rand=np.random.RandomState(),
            ntimesteps=args.ntimesteps,
            **args.sample_frames_params),
        augment_motion=motion.make_motion_augmenter(
            rand=np.random.RandomState(),
            **args.augment_motion_params),
        max_videos=None,
        max_objects=None)

    restorer = tf.train.Saver()

    with tf.Session() as sess:
        # Load model from checkpoint.
        # TODO: Need to call init() for anything?
        model_file = tf.train.latest_checkpoint(args.path_ckpt)
        if model_file is None:
            raise ValueError('could not find checkpoint')
        restorer.restore(sess, model_file)

        results = evaluate.evaluate(sess, eval_model, sequences)

    import pprint
    pprint.pprint(results)


if __name__ == '__main__':
    main()
