import pdb
import numpy as np
import os
from PIL import Image, ImageDraw, ImageColor
import shutil
import subprocess
import tempfile
import matplotlib.cm as cm


class VideoFileWriter:
    def __init__(self, root, pattern='%06d.jpeg'):
        self.root    = root
        self.pattern = pattern

    def visualize(self, sequence_name, sequence, rects_pred, hmaps_pred):
        sequence_dir = tempfile.mkdtemp()
        # if not os.path.isdir(sequence_dir):
        #     os.makedirs(sequence_dir)
        sequence_len = len(sequence['image_files'])
        rects_gt = sequence['labels']
        is_valid_gt = sequence['label_is_valid']
        color_pred = ImageColor.getrgb('yellow')
        color_gt = ImageColor.getrgb('blue')
        for t in range(sequence_len):
            im = Image.open(sequence['image_files'][t])
            if im.mode != 'RGB':
                im = im.convert('RGB')
            draw = ImageDraw.Draw(im)
            if is_valid_gt[t]:
                rect_gt = _rect_to_int_list(_unnormalize_rect(rects_gt[t], im.size))
                draw.rectangle(rect_gt, outline=color_gt)
            if t > 0:
                rect_pred = _rect_to_int_list(_unnormalize_rect(rects_pred[t-1], im.size))
                draw.rectangle(rect_pred, outline=color_pred)
                # draw heatmap
                hmap_pred = Image.fromarray(np.uint8(255*cm.hot(hmaps_pred[t-1,:,:,0])))
                im = Image.blend(im.convert('RGBA'), hmap_pred.convert('RGBA'), 0.5)
                #im.show()
            im.save(os.path.join(sequence_dir, self.pattern % t))
        args = ['ffmpeg', '-loglevel', 'error',
                          '-r', '1', # fps.
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
            shutil.rmtree(sequence_dir)


def _unnormalize_rect(r, size):
    # TODO: Avoid duplication. This was copied from evaluate.py.
    width, height = size
    return r * np.array([width, height, width, height])


def _rect_to_int_list(rect):
    return map(lambda x: int(round(x)), list(rect))
