import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from seqtrack import cnnutil
from seqtrack.cnnutil import ReceptiveField, IntRect

from tensorflow.contrib.layers.python.layers.utils import n_positive_integers


def crop():
    pass


def conv2d(inputs, input_rfs, num_outputs, kernel_size, stride=1, padding='SAME', **kwargs):
    assert len(inputs.shape) == 4
    outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride, padding, **kwargs)
    rel_rf = _filter_rf(kernel_size, stride, padding)
    if input_rfs is None:
        output_rfs = None
    else:
        output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def max_pool2d(inputs, input_rfs, kernel_size, stride=2, padding='VALID', **kwargs):
    outputs = slim.max_pool2d(
        inputs,
        kernel_size,
        stride,
        padding,
        **kwargs)
    rel_rf = _filter_rf(kernel_size, stride, padding)
    if input_rfs is None:
        output_rfs = None
    else:
        output_rfs = {k: cnnutil.compose_rf(v, rel_rf) for k, v in input_rfs.items()}
    return outputs, output_rfs


def _filter_rf(kernel_size, stride, padding):
    kernel_size = np.array(n_positive_integers(2, kernel_size))
    stride = np.array(n_positive_integers(2, stride))
    # Get relative receptive field.
    if padding == 'SAME':
        assert np.all(kernel_size % 2 == 1)
        half = (kernel_size - 1) / 2
        rect = IntRect(-half, half+1)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    elif padding == 'VALID':
        rect = IntRect(np.zeros_like(kernel_size), kernel_size)
        rel_rf = ReceptiveField(rect=rect, stride=stride)
    else:
        raise ValueError('invalid padding: {}'.format(padding))
    return rel_rf
