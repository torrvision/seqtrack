'''Contains functions for constructing the TF graph.'''

import tensorflow as tf
import numpy as np

from seqtrack import geom
from seqtrack.helpers import load_image_viewport, im_to_arr, pad_to

EXAMPLE_KEYS = ['x0', 'y0', 'x', 'y', 'y_is_valid', 'aspect']


class ModelInstance:
    '''ModelInstance describes an instantiated model.

    self.example['x0']
    self.example['y0']
    self.example['x']
    self.example['y']
    self.outputs['y']
    self.state_init: Nested structure of tensors.
    self.state_final: Nested structure of tensors.
    self.batchsz
    self.ntimesteps
    self.imheight
    self.imwidth

    Note that batchsz may be None to indicate that dynamic batch size is supported.
    '''

    def __init__(self, example, run_opts, outputs, state_init, state_final,
                 batchsz, ntimesteps, imheight, imwidth):
        self.example = example
        self.run_opts = run_opts
        self.outputs = outputs
        self.state_init = state_init
        self.state_final = state_final
        self.batchsz = batchsz
        self.ntimesteps = ntimesteps
        self.imheight = imheight
        self.imwidth = imwidth


def make_placeholders(ntimesteps, im_size, default=None):
    '''
    Args:
        im_size: (height, width) to construct tensor
    '''
    im_height, im_width = im_size
    shapes = {
        'x0': [None, im_height, im_width, 3],
        'y0': [None, 4],
        'x': [None, ntimesteps, im_height, im_width, 3],
        'y': [None, ntimesteps, 4],
        'y_is_valid': [None, ntimesteps],
        'aspect': [None],
    }

    def dtype(k):
        return tf.bool if k.endswith('_is_valid') else tf.float32

    if default is not None:
        assert(set(default.keys()) == set(shapes.keys()))
        example = {
            k: tf.placeholder_with_default(default[k], shapes[k], name='placeholder_' + k)
            for k in shapes.keys()}
    else:
        example = {
            k: tf.placeholder(dtype(k), shapes[k], name='placeholder_' + k)
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
        std = tf.constant(stat['std'] if stat else 1.0, tf.float32, name='std')
        example = dict(example_raw)  # Copy dictionary before modifying.
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


def load_images(image_files, image_size_hw, viewports=None, pad_value=128, name='load_images'):
    '''Loads, crops and resizes images for a sequence.

    Args:
        image_files: Tensor [None]
        image_size_hw: Image dimensions.
        viewports: Tensor [None, 4] or None
    '''
    with tf.name_scope(name) as scope:
        # Read files from disk.
        file_contents = tf.map_fn(tf.read_file, image_files, dtype=tf.string)
        # Decode images.
        images = tf.map_fn(lambda x: tf.image.decode_jpeg(x, channels=3),
                           file_contents, dtype=tf.uint8)
        if viewports is None:
            # Resize entire image.
            images = tf.image.resize_images(images, image_size_hw)
        else:
            # Sample viewport in image.
            # TODO: Avoid casting uint8 -> float32 -> uint8 -> float32.
            images = tf.image.convert_image_dtype(images, tf.float32)
            images = tf.image.crop_and_resize(images, geom.rect_to_tf_box(viewports),
                                              box_ind=tf.range(tf.shape(images)[0]),
                                              crop_size=image_size_hw,
                                              extrapolation_value=pad_value / 255.)
            # tf.image.crop_and_resize converts to float32.
            images = tf.image.convert_image_dtype(images, tf.uint8)
        return images


def load_images_batch(image_files, image_size_hw, viewports=None, name='load_images_batch',
                      **kwargs):
    '''Loads, crops and resizes images for a batch of sequences.

    Args:
        image_files: Tensor [batchsz, None]
        image_size_hw: Image dimensions.
        viewports: Tensor [batchsz, None, 4] or None
    '''
    with tf.name_scope(name) as scope:
        elems = {'image_files': image_files}
        if viewports is not None:
            elems['viewports'] = viewports
        images = tf.map_fn(
            lambda elem: load_images(elem['image_files'], image_size_hw,
                                     elem.get('viewports', None), **kwargs),
            elems, dtype=tf.uint8)
        return images


def py_load_batch(seqs, ntimesteps, im_size):
    '''Loads image data for a batch of sequences and constructs values for feed dict.

    All sequences will be padded to ntimesteps.

    Args:
        im_size: (height, width)

    Example has keys:
        'x0'     # First image in sequence, shape [h, w, 3]
        'y0'         # Position of target in first image, shape [4]
        'x'      # Input images, shape [n-1, h, w, 3]
        'y'          # Position of target in following frames, shape [n-1, 4]
        'y_is_valid' # Booleans indicating presence of frame, shape [n-1]
        'aspect'     # Aspect ratio of original image.
    '''
    sequence_keys = set(['x', 'y', 'y_is_valid'])
    examples = map(lambda x: py_load_batch_elem(x, im_size), seqs)
    # Pad all sequences to o.ntimesteps.
    # NOTE: Assumes that none of the arrays to be padded are empty.
    return {k: np.stack([pad_to(x[k], ntimesteps, axis=0)
                         if k in sequence_keys else x[k]
                         for x in examples])
            for k in EXAMPLE_KEYS}


def py_load_batch_elem(seq, im_size):
    '''Loads image data for one element of a batch.

    Args:
        im_size: (height, width)

    Sequence has keys:
        'image_files'    # Tensor with shape [n] containing strings.
        'labels'         # Tensor with shape [n, 4] containing rectangles.
        'label_is_valid' # Tensor with shape [n] containing booleans.
        'aspect'         # Tensor with shape [] containing aspect ratio.
    '''
    seq_len = len(seq['image_files'])
    assert(len(seq['labels']) == seq_len)
    assert(len(seq['label_is_valid']) == seq_len)
    assert(seq['label_is_valid'][0] is True)
    # f = lambda x: im_to_arr(load_image(x, size=(o.frmsz, o.frmsz), resize=False),
    #                         dtype=np.float32)
    images = [
        im_to_arr(load_image_viewport(seq['image_files'][t], seq['viewports'][t], im_size))
        for t in range(seq_len)
    ]
    return {
        'x0': np.array(images[0]),
        'y0': np.array(seq['labels'][0]),
        'x': np.array(images[1:]),
        'y': np.array(seq['labels'][1:]),
        'y_is_valid': np.array(seq['label_is_valid'][1:]),
        'aspect': seq['aspect'],
    }
