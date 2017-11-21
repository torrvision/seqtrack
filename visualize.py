import pdb
import numpy as np
import os
from PIL import Image, ImageDraw, ImageColor
import shutil
import subprocess
import tempfile
import matplotlib.cm as cm

import matplotlib
matplotlib.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy

import geom_np
from helpers import load_image, load_image_viewport, escape_filename


COLOR_PRED = ImageColor.getrgb('yellow')
COLOR_GT   = ImageColor.getrgb('blue')


def transparent_cmap(cmap, N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0.3, 0.7, N+4)
    return mycmap


def draw_output_mpl(im, rect_gt=None, rect_pred=None, hmap_pred=None, fname=None, cmap=None):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.size[0] / float(min(im.size)), im.size[1] / float(min(im.size)))

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(im)

    if hmap_pred is not None:
        if hmap_pred.shape[:2] == im.size:
            ax.imshow(np.squeeze(hmap_pred), alpha=1., cmap=cmap)
        else:
            ax.imshow(scipy.misc.imresize(np.squeeze(hmap_pred), (im.size[1], im.size[0])), alpha=1., cmap=cmap)

    ax = plt.gca()
    if rect_gt is not None:
        rect_gt = _rect_to_int_list(_unnormalize_rect(rect_gt, im.size))
        ax.add_patch(Rectangle(
            (rect_gt[0], rect_gt[1]), rect_gt[2]-rect_gt[0], rect_gt[3]-rect_gt[1],
            facecolor='blue', edgecolor='blue', fill=False))
    if rect_pred is not None:
        rect_pred = _rect_to_int_list(_unnormalize_rect(rect_pred, im.size))
        ax.add_patch(Rectangle(
            (rect_pred[0], rect_pred[1]), rect_pred[2]-rect_pred[0], rect_pred[3]-rect_pred[1],
            facecolor='y', edgecolor='y', fill=False))

    #plt.axis('off')
    fig.savefig(fname, dpi=min(im.size))
    plt.close()
    del ax
    del fig


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
        hmap_pred = Image.fromarray(cm.hot(hmap_pred[:,:,0], bytes=True)).convert('RGB')
        # Caution: This does not resize with align_corners=True.
        if hmap_pred.size != im.size: # i.e., OTB
            hmap_pred = hmap_pred.resize(im.size)
        im = Image.blend(im, hmap_pred, 0.5)
    return im


class VideoFileWriter:
    def __init__(self, root, pattern='%06d.jpeg'):
        self.root    = root
        self.pattern = pattern

    def visualize(self, sequence_name, sequence, rects_pred, hmaps_pred, save_frames):
        if not save_frames:
            sequence_dir = tempfile.mkdtemp()
        else:
            sequence_dir = os.path.join(self.root, 'frames', escape_filename(sequence_name))
            os.makedirs(sequence_dir)
        # if not os.path.isdir(sequence_dir):
        #     os.makedirs(sequence_dir)
        sequence_len = len(sequence['image_files'])
        rects_gt = sequence['labels']
        is_valid_gt = sequence['label_is_valid']
        color_pred = ImageColor.getrgb('yellow')
        color_gt = ImageColor.getrgb('blue')
        for t in range(sequence_len):
            # im = load_image_viewport(sequence['image_files'][t],
            #                          sequence['viewports'][t],
            #                          image_size)
            assert np.all(sequence['viewports'][t] == geom_np.unit_rect())
            im = load_image(sequence['image_files'][t])
            im = draw_output(im,
                rect_gt=(rects_gt[t] if is_valid_gt[t] else None),
                rect_pred=(rects_pred[t-1] if t > 0 else None),
                hmap_pred=(hmaps_pred[t-1] if t > 0 else None))
            #im.show()
            im.save(os.path.join(sequence_dir, self.pattern % t))
        args = ['ffmpeg', '-loglevel', 'error',
                          # '-r', '1', # fps.
                          '-y', # Overwrite without asking.
                          '-nostdin', # No interaction with user.
                          '-i', self.pattern,
                          '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                          os.path.join(os.path.abspath(self.root),
                                       escape_filename(sequence_name)+'.mp4')]
        try:
            p = subprocess.Popen(args, cwd=sequence_dir)
            p.wait()
        except Exception as inst:
            print 'error:', inst
        finally:
            if not save_frames:
                shutil.rmtree(sequence_dir)


def _unnormalize_rect(r, size):
    # TODO: Avoid duplication. This was copied from evaluate.py.
    width, height = size
    return r * np.array([width, height, width, height])


def _rect_to_int_list(rect):
    return map(lambda x: int(round(x)), list(rect))
