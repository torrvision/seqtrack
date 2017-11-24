import numpy as np

import pprint
import pdb

class IntRect:
    '''Describes a rectangle.

    The elements of the rectangle satisfy min <= u < max.
    '''

    def __init__(self, min=(0, 0), max=(0, 0)):
        # Numbers should be integers, but it is useful to have +inf and -inf.
        self.min = np.array(min)
        self.max = np.array(max)

    def __eq__(a, b):
        return np.array_equal(a.min, b.min) and np.array_equal(a.max, b.max)

    def __str__(self):
        return '{}-{}'.format(tuple(self.min), tuple(self.max))

    def empty(self):
        return not all(self.min < self.max)

    def size(self):
        return self.max - self.min

    def int_center(self):
        s = self.min + self.max - 1
        if not all(s % 2 == 0):
            raise ValueError('rectangle does not have integer center')
        return s / 2

    def intersect(a, b):
        return IntRect(np.maximum(a.min, b.min), np.minimum(a.max, b.max))

class ReceptiveField:
    '''Describes the receptive fields (in an earlier layer) of all pixels in a later layer.

    If y has receptive field rf in x, then pixel y[v] depends on pixels x[u] where
        v*rf.stride + rf.rect.min <= u < v*rf.stride + rf.rect.max
    '''
    def __init__(self, rect=IntRect(), stride=(0, 0)):
        self.rect   = rect
        self.stride = np.array(stride)

    def __eq__(a, b):
        return a.rect == b.rect and a.stride == b.stride

def identity_rf():
    return ReceptiveField(rect=IntRect((0, 0), (1, 1)), stride=(1, 1))

def infinite_rect():
    return IntRect(min=(float('-inf'), float('-inf')), max=(float('+inf'), float('+inf')))

def compose_rf(prev_rf, rel_rf):
    '''Computes the receptive field of the next layer.

    Given relative receptive field in prev of pixels in curr.
    input -> ... -> prev -> curr
    '''
    # curr[v] depends on prev[u] for
    #   v*rel_rf.stride + rel_rf.rect.min <= u <= v*rel_rf.stride + rel_rf.rect.max - 1
    # and prev[u] depends on input[t] for
    #   u*prev_rf.stride + prev_rf.rect.min <= t <= u*prev_rf.stride + prev_rf.rect.max - 1
    #
    # Therefore, curr[v] depends on input[t] for t between
    #   (v*rel_rf.stride + rel_rf.rect.min) * prev_rf.stride + prev_rf.rect.min
    #   (v*rel_rf.stride + rel_rf.rect.max - 1) * prev_rf.stride + prev_rf.rect.max - 1
    # or equivalently
    #   v*(rel_rf.stride*prev_rf.stride) + (rel_rf.rect.min*prev_rf.stride + prev_rf.rect.min)
    #   v*(rel_rf.stride*prev_rf.stride) + ((rel_rf.rect.max-1)*prev_rf.stride + prev_rf.rect.max) - 1
    stride = prev_rf.stride * rel_rf.stride
    min = prev_rf.rect.min + prev_rf.stride * rel_rf.rect.min
    max = prev_rf.rect.max + prev_rf.stride * (rel_rf.rect.max-1)
    return ReceptiveField(IntRect(min, max), stride)

def find_rf(s, t, prev_rf=identity_rf()):
    # Traverse graph from input to output.
    # Return None if no path can be found, or if the receptive field cannot be determined.
    # Careful: Traverses every path (i.e. bad for ResNet).
    if s is t:
        return prev_rf
    if not prev_rf:
        return None
    rfs = []
    for op in s.consumers():
        # Get receptive fields
        rel_rf = _operation_rf(op)
        if not rel_rf:
            # Receptive field cannot be determined.
            continue
        # if len(op.outputs) > 1:
        #     raise ValueError('multiple outputs')
        curr_rf = compose_rf(prev_rf, rel_rf)
        rf = find_rf(op.outputs[0], t, curr_rf)
        rfs.append(rf)
    # Resolve multiple receptive fields.
    return reduce(resolve_rfs, rfs, None)

def _operation_rf(op):
    '''Returns the receptive field of the operation.

    Considers only the first input and the first output.
    Returns None if the receptive field cannot be determined.
    '''
    # Careful: This probably violates encapsulation and might break if TensorFlow changes.

    if op.type in {'Conv2D', 'MaxPool'}:
        data_format = op.get_attr('data_format')
        if data_format != 'NHWC':
            raise ValueError('unexpected format: {}'.format(data_format))
        if op.type == 'Conv2D':
            filter_var = op.inputs[1]
            filter_size = np.array(map(int, filter_var.shape[0:2]))
        else:
            filter_size = np.array(map(int, op.get_attr('ksize')[1:3]))
        rect = _filter_rect_padding(filter_size, op.get_attr('padding'))
        stride = np.array(map(int, op.get_attr('strides')[1:3]))
        return ReceptiveField(rect, stride)
    elif op.type in {'BiasAdd', 'Relu', 'Identity', 'HistogramSummary'}:
        return identity_rf()
    elif op.type in {'Reshape'}:
        return None
    else:
        print str(op)
        raise ValueError('unknown type: {}'.format(op.type))

def _filter_rect_padding(filter_size, padding):
    if padding == 'SAME':
        if all(filter_size % 2 != 0):
            # Both filter dimensions are odd - good.
            # Filter support is [-(size-1)/2, ..., 0, ..., (size-1)/2]
            return IntRect(-(filter_size-1)/2, (filter_size-1)/2+1)
        else:
            # Pad and put the extra row/column at the end.
            # Filter support is [-(size/2-1), ..., 0, ..., size/2-1, size/2]
            return IntRect(-(filter_size/2 - 1), filter_size/2 + 1)
    elif padding == 'VALID':
        return IntRect((0, 0), filter_size)
    else:
        raise ValueError('unknown padding type: {}'.format(padding))

def resolve_rfs(a, b):
    if not a:
        return b
    if not b:
        return a
    if a != b:
        raise ValueError('receptive fields differ: {} vs {}'.format(str(a), str(b)))
    return a
