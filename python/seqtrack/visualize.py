from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.cm as cm

from PIL import Image, ImageDraw, ImageColor

import logging
logger = logging.getLogger(__name__)


COLOR_PRED = ImageColor.getrgb('yellow')
COLOR_GT = ImageColor.getrgb('blue')


def draw_output(im, rect_gt=None, rect_pred=None, hmap_pred=None, color_pred=None, color_gt=None):
    '''Modifies original image and returns an image that may be different.'''
    if color_pred is None:
        color_pred = COLOR_PRED
    if color_gt is None:
        color_gt = COLOR_GT

    draw = ImageDraw.Draw(im)
    if rect_gt is not None:
        rect_gt = _rect_to_int_list(_unnormalize_rect(rect_gt, im.size))
        draw.rectangle(rect_gt, outline=color_gt)
    if rect_pred is not None:
        rect_pred = _rect_to_int_list(_unnormalize_rect(rect_pred, im.size))
        draw.rectangle(rect_pred, outline=color_pred)
    del draw

    if hmap_pred is not None:
        assert len(hmap_pred.shape) == 3
        hmap_pred = Image.fromarray(cm.hot(hmap_pred[:, :, 0], bytes=True)).convert('RGB')
        # Caution: This does not resize with align_corners=True.
        if hmap_pred.size != im.size:  # i.e., OTB
            hmap_pred = hmap_pred.resize(im.size)
        im = Image.blend(im, hmap_pred, 0.5)
    return im


def _unnormalize_rect(r, size):
    # TODO: Avoid duplication. This was copied from track.py.
    width, height = size
    return r * np.array([width, height, width, height])


def _rect_to_int_list(rect):
    return list(map(lambda x: int(round(x)), list(rect)))
