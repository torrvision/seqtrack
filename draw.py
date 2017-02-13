import pdb
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import helpers


def show_dataset_batch(batch, dataset, frmsz):
    vids = batch['inputs']
    if vids.shape[0] > 25: 
        vids = vids[:25] # only draws less than 25 
    pos = batch['labels']
    pos = pos * frmsz # TODO: y is relative scale? change for other datasets too
    
    if dataset == 'moving_mnist' or dataset == 'bouncing_mnist':
        digits = batch['digits']
        plt.gray()

    for t in range(vids.shape[1]): # timesteps
        fig = plt.figure(figsize=(12,12))
        for i in range(vids.shape[0]): # batch
            plt.subplot(5,5,i+1)
            if dataset == 'ilsvrc':
                plt.imshow(np.uint8(vids[i,t]))
            elif dataset == 'moving_mnist' or dataset == 'bouncing_mnist':
                plt.imshow(vids[i,t])
            if dataset == 'moving_mnist' or dataset == 'bouncing_mnist':
                plt.title(digits[i])
            ax = plt.gca()
            ax.add_patch(
                Rectangle(
                    (pos[i,t,0], pos[i,t,1]), 
                    pos[i,t,2]-pos[i,t,0], pos[i,t,3]-pos[i,t,1],
                    facecolor='r', edgecolor='r', fill=False))
            plt.draw()
            plt.axis('off')
        plt.savefig('tmp/{}/frame{}.png'.format(dataset, t))
        plt.close()
    os.system('convert -loop 0 -delay 30 tmp/{}/frame*.png tmp/{}/vid.gif'.\
        format(dataset, dataset))

def show_tracking_results_moving_mnist(results, o, save_=False):
    if save_:
        savedir = os.path.join(o.path_eval, 'results_moving_mnist_tmp') 
        helpers.mkdir_p(savedir) 

    idx = results['idx']
    inputs = results['inputs']
    labels = results['labels']
    outputs = results['outputs']
    frmsz = o.moving_mnist['frmsz']

    plt.gray()
    plt.show()
    for ib in range(len(idx)):
        for ie in range(o.batchsz):
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
                plt.savefig(
                        savedir+'/b{0:d}_exp{1:d}_frm{2:03d}.png'.format(ib,ie,t))
                plt.close()
            os.system(
                    'convert -loop 0 -delay 30\
                    {0:s}/b{1:d}_exp{2:d}_frm*.png\
                    {3:s}/b{4:d}_exp{5:d}_vid.gif'\
                            .format(savedir,ib,ie, savedir,ib,ie))

                
def plot_losses(losses, o, intermediate_=False, cnt_=''): # after trainingj
    if not intermediate_:
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(losses['batch'].shape[0])+1, losses['batch'], '-o')
        ax1.set_title('batch losses')
        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(losses['epoch'].shape[0])+1, losses['epoch'], '-o')
        ax2.set_title('epoch losses')
    else:
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.plot(
                np.arange(losses['interm_avg'].shape[0])+1, 
                losses['interm_avg'], '-o')
        ax1.set_title('average intermediate loss')

    if o.nosave:
        outfile = os.path.join(
                o.path_save_tmp, o.exectime+'_losses{}.png'.format(cnt_))   
    else:
        outfile = os.path.join(o.path_loss, 'losses{}.png'.format(cnt_))
    plt.savefig(outfile)
    plt.close()

def plot_losses_train_val(loss_train, loss_val, o, cnt_):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(loss_train.shape[0]), loss_train, 'b-o')
    ax1.plot(np.arange(loss_val.shape[0]), loss_val, 'r-o')
    ax1.set_title('average intermediate loss for evaluation subsets')
    if o.nosave:
        outfile = os.path.join(
            o.path_save_tmp, o.exectime+'_losses_evalsubset{}.png'.format(cnt_))
    else:
        outfile = os.path.join(
                o.path_loss, 'losses_evalsubset{}.png'.format(cnt_))
    plt.savefig(outfile)
    plt.close()



