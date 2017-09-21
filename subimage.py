import tensorflow as tf

import geom
from helpers import most_specific_shape

#   subimage is a dict with:
#       'image':    shape [..., h, w, c]
#       'viewport': shape [..., 4]

def make(image, viewport=None):
    if viewport is None:
        viewport = geom.unit_rect()
    # Ensure that viewport dimension matches image dimension.
    num_images = most_specific_shape(image)[:-3]
    viewport = tf.ones(num_images + [1]) * viewport
    return {'image': image, 'viewport': viewport}

def from_size(im, size):
    '''Returns top left corner as sub-image.

    Args:
        size: Tensor giving size in pixels.
            Shape is [..., 2].
            Order is (height, width).
    '''
    canvas_height, canvas_width = most_specific_shape(im)[-3:-1]
    canvas_size = tf.stack((canvas_width, canvas_height), axis=-1)
    image_height, image_width = tf.unstack(size, axis=-1)
    size = tf.stack((image_width, image_height), axis=-1)
    rel_size = tf.to_float(size) * (1./tf.to_float(canvas_size))
    im_extent = geom.make_rect(tf.zeros_like(rel_size), rel_size)
    return make(im, im_extent)

# def extract_from_size(im, size, **kwargs):
#     return extract(from_size(im, size), size, **kwargs)

def extract(subimage, crop_size, **kwargs):
    '''Converts a sub-image to an image.'''
    # TODO: Support variable dimension using merge_dims?
    assert len(subimage['image'].shape) == 4
    batch_len = tf.shape(subimage['image'])[0]
    return tf.image.crop_and_resize(
        subimage['image'],
        geom.rect_to_tf_box(subimage['viewport']),
        box_ind=tf.range(batch_len),
        crop_size=crop_size,
        **kwargs)

def crop_and_extract(subimage, rect, crop_size, **kwargs):
    '''Extracts a rectangle from a sub-image.'''
    return extract(crop(subimage, rect), crop_size, **kwargs)

def crop(subimage, rect):
    '''Extract a new sub-image within a sub-image.'''
    # rect/viewport = crop(rect/image, viewport/image)
    # rect/image = crop(rect/viewport, image/viewport)
    #            = crop(rect/viewport, inv(viewport/image))
    return {
        'image':    subimage['image'],
        'viewport': geom.crop_rect(rect, geom.crop_inverse(subimage['viewport'])),
    }
