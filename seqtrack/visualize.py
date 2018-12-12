from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from PIL import Image, ImageDraw, ImageColor
import shutil
import subprocess
import tempfile
import matplotlib.cm as cm

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom_np
from seqtrack.helpers import load_image, load_image_viewport, escape_filename


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


# class VideoFileWriter:
#     def __init__(self, root, pattern='%06d.jpeg'):
#         self.root = root
#         self.pattern = pattern
#
#     def visualize(self, sequence_name, sequence, rects_pred, hmaps_pred, keep_frames):
#         if not keep_frames:
#             sequence_dir = tempfile.mkdtemp()
#         else:
#             sequence_dir = os.path.join(self.root, 'frames', escape_filename(sequence_name))
#             os.makedirs(sequence_dir)
#         # if not os.path.isdir(sequence_dir):
#         #     os.makedirs(sequence_dir)
#         sequence_len = len(sequence['image_files'])
#         rects_gt = sequence['labels']
#         is_valid_gt = sequence['label_is_valid']
#         color_pred = ImageColor.getrgb('yellow')
#         color_gt = ImageColor.getrgb('blue')
#         for t in range(sequence_len):
#             # im = load_image_viewport(sequence['image_files'][t],
#             #                          sequence['viewports'][t],
#             #                          image_size)
#             assert np.all(sequence['viewports'][t] == geom_np.unit_rect())
#             im = load_image(sequence['image_files'][t])
#             im = draw_output(im,
#                              rect_gt=(rects_gt[t] if is_valid_gt[t] else None),
#                              rect_pred=(rects_pred[t - 1] if t > 0 else None),
#                              hmap_pred=(hmaps_pred[t - 1] if t > 0 else None))
#             #im.show()
#             im.save(os.path.join(sequence_dir, self.pattern % t))
#         args = ['ffmpeg', '-loglevel', 'error',
#                           # '-r', '1', # fps.
#                           '-y',  # Overwrite without asking.
#                           '-nostdin',  # No interaction with user.
#                           '-i', self.pattern,
#                           '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
#                           os.path.join(os.path.abspath(self.root),
#                                        escape_filename(sequence_name) + '.mp4')]
#         try:
#             subprocess.check_call(args, cwd=sequence_dir)
#         except Exception as ex:
#             logger.warning('error calling ffmpeg: %s', str(ex))
#         finally:
#             if not keep_frames:
#                 shutil.rmtree(sequence_dir)


def _unnormalize_rect(r, size):
    # TODO: Avoid duplication. This was copied from track.py.
    width, height = size
    return r * np.array([width, height, width, height])


def _rect_to_int_list(rect):
    return list(map(lambda x: int(round(x)), list(rect)))
