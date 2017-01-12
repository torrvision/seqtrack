import pdb
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def show_moving_mnist(batch):
    vids = batch['inputs']
    if vids.shape[0] > 25: 
        vids = vids[:25] # only draws less than 25 
    pos = batch['labels']
    digits = batch['digits']

    plt.gray()
    plt.show()
    for t in range(vids.shape[1]): # timesteps
        for i in range(vids.shape[0]): # batch
            plt.subplot(5,5,i+1)
            plt.imshow(vids[i,t])
            plt.title(digits[i])
            ax = plt.gca()
            ax.add_patch(
                    Rectangle(
                        pos[i,t][::-1], 28, 28, 
                        facecolor='r', edgecolor='r', fill=False))
            plt.draw()
            plt.axis('off')
        plt.savefig('tmp/frame{0:03d}.png'.format(t))
        plt.close()
    os.system('convert -loop 0 -delay 30 tmp/frame*.png tmp/vid.gif')
