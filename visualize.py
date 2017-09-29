import pdb
import numpy as np
import os
from PIL import Image, ImageDraw, ImageColor
import shutil
import subprocess
import tempfile
import matplotlib.cm as cm

from helpers import load_image_viewport


class VideoFileWriter:
    def __init__(self, root, pattern='%06d.jpeg'):
        self.root    = root
        self.pattern = pattern

    def visualize(self, sequence_name, sequence, rects_pred, hmaps_pred, image_size, save_frames):
        if not save_frames:
            sequence_dir = tempfile.mkdtemp()
        else:
            sequence_dir = os.path.join(self.root, 'frames/{}'.format(sequence_name))
            os.makedirs(sequence_dir)
        # if not os.path.isdir(sequence_dir):
        #     os.makedirs(sequence_dir)
        sequence_len = len(sequence['image_files'])
        rects_gt = sequence['labels']
        is_valid_gt = sequence['label_is_valid']
        color_pred = ImageColor.getrgb('yellow')
        color_gt = ImageColor.getrgb('blue')
        for t in range(sequence_len):
            # JV: Load image at specified resolution accounting for viewport.
            # im = Image.open(sequence['image_files'][t])
            # if im.mode != 'RGB':
            #     im = im.convert('RGB')
            im = load_image_viewport(sequence['image_files'][t],
                                     sequence['viewports'][t],
                                     image_size)
            draw = ImageDraw.Draw(im)
            if is_valid_gt[t]:
                rect_gt = _rect_to_int_list(_unnormalize_rect(rects_gt[t], im.size))
                draw.rectangle(rect_gt, outline=color_gt)
            if t > 0:
                rect_pred = _rect_to_int_list(_unnormalize_rect(rects_pred[t-1], im.size))
                draw.rectangle(rect_pred, outline=color_pred)
                # draw heatmap
                hmap_pred = Image.fromarray(np.uint8(255*cm.hot(hmaps_pred[t-1,:,:,0])))
                if hmap_pred.size != im.size: # i.e., OTB
                    hmap_pred = hmap_pred.resize(im.size)
                im = Image.blend(im.convert('RGBA'), hmap_pred.convert('RGBA'), 0.5)
                #im.show()
            im.save(os.path.join(sequence_dir, self.pattern % t))
        args = ['ffmpeg', '-loglevel', 'error',
                          # '-r', '1', # fps.
                          '-y', # Overwrite without asking.
                          '-nostdin', # No interaction with user.
                          '-i', self.pattern,
                          '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                          os.path.join(os.path.abspath(self.root), sequence_name+'.mp4')]
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
