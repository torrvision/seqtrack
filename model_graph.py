import numpy as np
import tensorflow as tf

import geom
import geom_np

from helpers import load_image_viewport, im_to_arr, most_static_shape

EXAMPLE_KEYS = ['x0', 'y0', 'x', 'y', 'y_is_valid']


# class Dimension:
#     '''Dimension of model instance.'''
#     # Could include dtype here if desired.
#     def __init__(self, batchsz, ntimesteps, frmsz):
#         self.batchsz = batchsz # Can be None.
#         self.ntimesteps = ntimesteps
#         self.frmsz = frmsz


def make_example_placeholders(batchsz, ntimesteps, frmsz, default=None, dtype=tf.float32):
    '''Creates place to feed inputs.

    Images are converted to floating point using tf.image.convert_image_dtype.
    Pixel values are in [0, 1].
    Mean subtraction occurs inside the model.

    Set `batchsz=None` for variable batch size.
    '''
    shapes = {
        'x0':         [batchsz, frmsz, frmsz, 3],
        'y0':         [batchsz, 4],
        'x':          [batchsz, ntimesteps, frmsz, frmsz, 3],
        'y':          [batchsz, ntimesteps, 4],
        'y_is_valid': [batchsz, ntimesteps],
    }
    key_dtype = lambda k: tf.bool if k.endswith('_is_valid') else dtype

    if default is not None:
        assert(set(default.keys()) == set(shapes.keys()))
        example = {
            k: tf.placeholder_with_default(default[k], shapes[k], name='placeholder_'+k)
            for k in shapes.keys()}
    else:
        example = {
            k: tf.placeholder(key_dtype(k), shapes[k], name='placeholder_'+k)
            for k in EXAMPLE_KEYS}
    return example


def make_option_placeholders():
    run_opts = {}
    # Add a placeholder for models that use ground-truth during training.
    run_opts['use_gt'] = tf.placeholder_with_default(False, [], name='use_gt')
    # Add a placeholder that specifies training mode for e.g. batch_norm.
    run_opts['is_training'] = tf.placeholder_with_default(False, [], name='is_training')
    # Add a placeholder for scheduled sampling of y_prev_GT during training
    run_opts['gt_ratio'] = tf.placeholder_with_default(1.0, [], name='gt_ratio')
    return run_opts


def process_sequence(model, example, run_opts, batchsz, ntimesteps, frmsz):
    '''Takes a Model and calls init() and step() on the example.

    Returns prediction in the image reference frame,
    and prediction_crop in the window reference frame.
    The prediction in the window reference frame can be used to apply the loss.

    Example:
        prediction_crop, viewport, prediction, init_state, final_state = model_graph.process_sequence(
            model, example, run_opts, batchsz, ntimesteps, frmsz)

    Args:
        example has fields:
            x0, y0, x, y, y_is_valid
        run_opts has fields:
            use_gt, is_training

    Returns:
        prediction_crop: Model prediction in reference frame of viewport.
        viewport: Search region of model.
        prediction: Model prediction in image reference frame.
        init_state: Initial state.
        final_state: Final state.
            Should be possible to feed final state to initial state to continue
            from end of sequence.
    '''
    example_input = dict(example)
    example = _guard_labels(example, run_opts)
    example_init = {k: example[k] for k in ['x0', 'y0']}
    example_seq = _unstack_dict(example, ['x', 'y', 'y_is_valid'], axis=1)
    # _unstack_example_frames(example)
    state = [None for __ in range(ntimesteps)]
    viewport = [None for __ in range(ntimesteps)]
    prediction_crop = [None for __ in range(ntimesteps)]
    prediction = [None for __ in range(ntimesteps)]

    with tf.variable_scope('model'):
        with tf.name_scope('frame_0'):
            init_state = model.init(example_init, run_opts)
        for t in range(ntimesteps):
            with tf.name_scope('frame_{}'.format(t+1)):
                # TODO: Whiten after crop i.e. in model.
                prediction_crop[t], viewport[t], state[t] = model.step(
                    example_seq[t],
                    state[t-1] if t > 0 else init_state)
                if t == 0:
                    pred_keys = prediction_crop[t].keys()
                # Obtain prediction in image frame.
                inv_viewport_rect = geom.crop_inverse(viewport[t])
                # TODO: Should this be inside this function or outside?
                prediction[t] = crop_prediction_frame(prediction_crop[t], inv_viewport_rect,
                    crop_size=[frmsz, frmsz])

    # TODO: This may include viewport state!
    final_state = state[-1]
    viewport = tf.stack(viewport, axis=1)
    prediction_crop = _stack_dict(prediction_crop, prediction_crop[0].keys(), axis=1)
    prediction = _stack_dict(prediction, prediction[0].keys(), axis=1)

    return prediction_crop, viewport, prediction, init_state, final_state


def _guard_labels(example, run_opts):
    '''Hides the 'y' labels if 'use_gt' is False.

    This prevents the model from accidentally using 'y'.
    '''
    # example['x'] -- [b, t, h, w, 3]
    # example['y']     -- [b, t, 4]
    images = example['x']
    safe = dict(example)
    images_shape = most_static_shape(images)
    safe['y'] = tf.cond(run_opts['use_gt'],
        lambda: example['y'],
        lambda: tf.fill(images_shape[0:2] + [4], float('nan')),
        name='labels_safe')
    return safe


def _stack_dict(frames, keys, axis):
    '''Converts list of dictionaries to dictionary of tensors.'''
    return {
        k: tf.stack([frame[k] for frame in frames], axis=axis)
        for k in keys
    }

def _unstack_dict(d, keys, axis):
    '''Converts dictionary of tensors to list of dictionaries.'''
    # Gather lists of all elements at same index.
    # {'x': [x0, x1], 'y': [y0, y1]} => [[x0, y0], [x1, y1]]
    value_lists = zip(*[tf.unstack(d[k], axis=axis) for k in keys])
    # Create a dictionary from each.
    # [[x0, y0], [x1, y1]] => [{'x': x0, 'y': y0}, {'x': x1, 'y': y1}]
    return [dict(zip(keys, vals)) for vals in value_lists]


def crop_prediction_frame(pred, window_rect, crop_size, name='crop_prediction_frame'):
    '''
    Args:
        pred -- Dictionary.
            pred['y'] -- [n, 4]
            pred['hmap_softmax'] -- [n, h, w, 2]
        window_rect -- [n, 4]
    '''
    with tf.name_scope(name):
        out = {}
        if 'y' in pred:
            out['y'] = geom.crop_rect(pred['y'], window_rect)
        if 'hmap_softmax' in pred:
            out['hmap_softmax'] = geom.crop_image(pred['hmap_softmax'], window_rect,
                crop_size=crop_size)
        if 'score_softmax' in pred:
            out['score_softmax'] = geom.crop_image(pred['score_softmax'], window_rect,
                crop_size=crop_size)
    return out


def load_sequence(sequence, frmsz):
    '''
    Analogous to pipeline.load_images.
    '''
    # Sequence has keys:
    #     'image_files'    # Tensor with shape [n] containing strings.
    #     'viewports'      # Tensor with shape [n] containing strings.
    #     'labels'         # Tensor with shape [n, 4] containing rectangles.
    #     'label_is_valid' # Tensor with shape [n] containing booleans.
    # Example has keys:
    #     'x0'         # First image in sequence, shape [h, w, 3]
    #     'y0'         # Position of target in first image, shape [4]
    #     'x'          # Input images, shape [n-1, h, w, 3]
    #     'y'          # Position of target in following frames, shape [n-1, 4]
    #     'y_is_valid' # Booleans indicating presence of frame, shape [n-1]

    seq_len = len(sequence['image_files'])
    assert(len(sequence['labels']) == seq_len)
    assert(len(sequence['label_is_valid']) == seq_len)
    assert(sequence['label_is_valid'][0] == True)
    images = [
        im_to_arr(load_image_viewport(
            sequence['image_files'][t],
            sequence['viewports'][t],
            size=(frmsz, frmsz)))
        for t in range(seq_len)
    ]
    rects = [
        geom_np.crop_rect(sequence['labels'][t], sequence['viewports'][t])
        for t in range(seq_len)
    ]
    return {
        'x0':         np.array(images[0]),
        'y0':         np.array(rects[0]),
        'x':          np.array(images[1:]),
        'y':          np.array(rects[1:]),
        'y_is_valid': np.array(sequence['label_is_valid'][1:]),
    }
