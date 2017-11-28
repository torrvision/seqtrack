'''Contains functions for constructing the TF graph.'''

import tensorflow as tf


# class ModelInstance:
#     '''ModelInstance describes an instantiated model.
# 
#     model_instance.example['x0']
#     model_instance.example['y0']
#     model_instance.example['x']
#     model_instance.example['y']
#     model_instance.outputs['y']
#     model_instance.state_init: Nested structure of tensors.
#     model_instance.state_final: Nested structure of tensors.
#     model_instance.batchsz
#     model_instance.ntimesteps
#     '''
# 
#     def __init__(self, example, outputs, state_init, state_final, ntimesteps, batchsz):
#         self.outputs = outputs
#         self.state_init = state_init
#         self.state_final = state_final
#         self.ntimesteps = ntimesteps
#         self.batchsz = batchsz


def make_placeholders(ntimesteps, im_size, default=None):
    '''
    Args:
        im_size: (height, width) to construct tensor
    '''
    im_height, im_width = im_size
    shapes = {
        'x0':         [None, im_height, im_width, 3],
        'y0':         [None, 4],
        'x':          [None, ntimesteps, im_height, im_width, 3],
        'y':          [None, ntimesteps, 4],
        'y_is_valid': [None, ntimesteps],
        'aspect':     [None],
    }
    dtype = lambda k: tf.bool if k.endswith('_is_valid') else tf.float32

    if default is not None:
        assert(set(default.keys()) == set(shapes.keys()))
        example = {
            k: tf.placeholder_with_default(default[k], shapes[k], name='placeholder_'+k)
            for k in shapes.keys()}
    else:
        example = {
            k: tf.placeholder(dtype(k), shapes[k], name='placeholder_'+k)
            for k in EXAMPLE_KEYS}
    run_opts = {}
    # Add a placeholder for models that use ground-truth during training.
    run_opts['use_gt'] = tf.placeholder_with_default(False, [], name='use_gt')
    # Add a placeholder that specifies training mode for e.g. batch_norm.
    run_opts['is_training'] = tf.placeholder_with_default(False, [], name='is_training')
    run_opts['is_tracking'] = tf.placeholder_with_default(False, [], name='is_tracking')
    # Add a placeholder for scheduled sampling of y_prev_GT during training
    run_opts['gt_ratio'] = tf.placeholder_with_default(1.0, [], name='gt_ratio')
    return example, run_opts


def whiten(example_raw, stat=None, name='whiten'):
    with tf.name_scope(name) as scope:
        # Normalize mean and variance.
        ## assert(stat is not None)
        # TODO: Check that this does not create two variables:
        mean = tf.constant(stat['mean'] if stat else 0.0, tf.float32, name='mean')
        std = tf.constant(stat['std'] if stat else 1.0,  tf.float32, name='std')
        example = dict(example_raw) # Copy dictionary before modifying.
        # Replace raw x (images) with whitened x (images).
        example['x'] = _whiten_image(example_raw['x'], mean, std, name='x')
        example['x0'] = _whiten_image(example_raw['x0'], mean, std, name='x0')
        return example


def _whiten_image(x, mean, std, name='whiten_image'):
    with tf.name_scope(name) as scope:
        #return tf.divide(x - mean, std, name=scope)
        return tf.divide(x - 0.0, 1.0, name=scope)


# def guard_labels(unsafe):
#     '''Hides the 'y' labels if 'use_gt' is False.
# 
#     This prevents the model from accidentally using 'y'.
#     '''
#     # unsafe['x'] -- [b, t, h, w, 3]
#     # unsafe['y']     -- [b, t, 4]
#     images = unsafe['x']
#     safe = dict(unsafe)
#     safe['y'] = tf.cond(unsafe['use_gt'],
#         lambda: unsafe['y'],
#         lambda: tf.fill(tf.concat([tf.shape(images)[0:2], [4]], axis=0), float('nan')),
#         name='labels_safe')
#     return safe
