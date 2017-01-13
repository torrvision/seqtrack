import pdb
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import helpers

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

def show_tracking_results_moving_mnist(results, o):
    idx = results['idx']
    inputs = results['inputs']
    labels = results['labels']
    outputs = results['outputs']
    frmsz = o.moving_mnist['frmsz']

    plt.gray()
    plt.show()
    for ib in range(len(idx)):
        for ie in range(o.batchsz):
            helpers.mkdir_p('tmp/results_moving_mnist') # TODO: change properly
            for t in range(o.ntimesteps):
                img_at_t = inputs[ib][ie,t].reshape(frmsz, frmsz)
                pos_gt_at_t = labels[ib][ie,t]
                pos_pred_at_t = outputs[ib][ie,t]

                plt.imshow(img_at_t)
                ax = plt.gca()
                ax.add_patch(
                        Rectangle(
                            pos_gt_at_t[::-1], 28, 28,
                            facecolor='r', edgecolor='r', fill=False))
                ax.add_patch(
                        Rectangle(
                            pos_pred_at_t[::-1], 28, 28,
                            facecolor='b', edgecolor='b', fill=False))
                plt.draw()
                plt.axis('off')
                plt.savefig('tmp/results_moving_mnist/\
                        batch{0:d}_exp{1:d}_frm{2:03d}.png'.format(ib,ie,t))
                plt.close()
            os.system(
                    'convert -loop 0 -delay 30 \
                        tmp/results_moving_mnist/batch{0:d}_exp{1:d}_frm*.png \
                        tmp/results_moving_mnist/batch{2:d}_exp{3:d}_vid.gif'\
                        .format(ib,ie, ib,ie))

                

