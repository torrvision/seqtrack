import tensorflow as tf
import numpy as np

# TODO: Make this work for stride other than 2.


def blur_filter(stride):
    assert stride == 2
    return np.array([
        [0.25, 0.5, 0.25],
        [0.5, 1., 0.5],
        [0.25, 0.5, 0.25]], dtype=np.float32)


def upsample_depthwise(x, stride=2, scope=None):
    assert stride == 2
    x_static = x.shape.as_list()
    x_dynamic = tf.shape(x)
    x_shape = [x_static[i] or x_dynamic[i] for i, _ in enumerate(x_static)]

    w = tf.constant(blur_filter(stride))
    # Transform [3, 3] -> [3, 3, 1, 1]
    w = tf.expand_dims(tf.expand_dims(w, -1), -1)
    # Apply same filters to every channel.
    # Transform [3, 3] -> [3, 3, C, 1]
    w = tf.tile(w, [1, 1, x_static[3], 1])

    y_shape = [x_shape[0], 2 * x_shape[1] - 1, 2 * x_shape[2] - 1, x_shape[3]]
    y = tf.nn.depthwise_conv2d_native_backprop_input(
        input_sizes=y_shape,
        filter=w,
        out_backprop=x,
        strides=[1, 2, 2, 1],
        padding='SAME',
    )
    # Restore as much shape information as possible.
    y.set_shape([x_static[0],
                 2 * x_static[1] - 1 if x_static[1] is not None else None,
                 2 * x_static[2] - 1 if x_static[2] is not None else None,
                 x_static[3]])
    return y


def upsample_dense(x, stride=2, scope=None):
    assert stride == 2
    x_static = x.shape.as_list()
    x_dynamic = tf.shape(x)
    x_shape = [x_static[i] or x_dynamic[i] for i, _ in enumerate(x_static)]

    w = tf.constant(blur_filter(stride))
    # Transform [3, 3] -> [3, 3, 1, 1]
    w = tf.expand_dims(tf.expand_dims(w, -1), -1)
    # Apply same filters to every channel.
    # Transform [3, 3] -> [3, 3, C, 1]
    mask = tf.constant(np.identity(x_static[3], dtype=np.float32))
    mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
    w *= mask

    y_shape = [x_shape[0], 2 * x_shape[1] - 1, 2 * x_shape[2] - 1, x_shape[3]]
    y = tf.nn.conv2d_transpose(x, w,
                               output_shape=y_shape,
                               strides=[1, 2, 2, 1],
                               padding='SAME'
                               )
    # Restore as much shape information as possible.
    y.set_shape([x_static[0],
                 2 * x_static[1] - 1 if x_static[1] is not None else None,
                 2 * x_static[2] - 1 if x_static[2] is not None else None,
                 x_static[3]])
    return y

# def upsample_manual(x, stride=2, scope=None):
#     assert stride == 2
#     x_static = x.shape.as_list()
#     x_dynamic = tf.shape(x)
#     x_shape = [x_static[i] or x_dynamic[i] for i, _ in enumerate(x_static)]
#
#     y_shape = [x_shape[0], 2*x_shape[1]-1, 2*x_shape[2]-1, x_shape[3]]
#     y = tf.zeros(y_shape, dtype=tf.float32)
#     y[:,  ::2,  ::2] = x
#     y[:, 1::2,  ::2] = 0.5  * (x[:, :-1,  :  ] + x[:, 1:,    :])
#     y[:,  ::2, 1::2] = 0.5  * (x[:, :,    :-1] + x[:,  :,   1:])
#     y[:, 1::2, 1::2] = 0.25 * (x[:, :-1,  :-1] + x[:,  :-1, 1:] +
#                                x[:, :-1, 1:-1] + x[:, 1:  , 1:])
#     # # Restore as much shape information as possible.
#     # y.set_shape([x_static[0],
#     #              2*x_static[1]-1 if x_static[1] is not None else None,
#     #              2*x_static[2]-1 if x_static[2] is not None else None,
#     #              x_static[3]])
#     print 'input shape:', x.shape
#     print 'output shape:', y.shape
#     return y


upsample = upsample_dense
