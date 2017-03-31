import pdb
import datetime
import errno
import numpy as np
import os
from PIL import Image

def get_time():
    dt = datetime.datetime.now()
    time = '{0:04d}{1:02d}{2:02d}_h{3:02d}m{4:02d}s{5:02d}'.format(
            dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second)
    return time

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

def getColormap(N,cmapname):
    '''Returns a function that maps each index in 0, 1, ..., N-1 to a distinct RGB color.'''
    colornorm = colors.Normalize(vmin=0, vmax=N-1)
    scalarmap = cmx.ScalarMappable(norm=colornorm, cmap=cmapname)
    def mapIndexToRgbColor(index):
        return scalarmap.to_rgba(index)
    return mapIndexToRgbColor

def createScalarMap(name='hot', vmin=-10, vmax=10):
    cm = plt.get_cmap(name)
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)

def load_image(fname):
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im

def im_to_arr(x):
    return np.array(x, dtype=np.float32)
