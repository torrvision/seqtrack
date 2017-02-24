import pdb
import numpy as np
import os

import matplotlib
matplotlib.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import helpers


def show_dataset_batch(batch, dataset, frmsz, stat=None):
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
            if dataset in ['moving_mnist', 'bouncing_mnist']:
                plt.imshow(np.squeeze(vids[i,t], axis=(2,)))
            else:
                # unnormalize using stat
                img = vids[i,t]
                img *= stat['std']
                img += stat['mean']
                plt.imshow(np.uint8(img))

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

def show_track_results(results, loader, dstype, o, iteration=None, nlimit=100):

    nbatches = results['nbatches']
    idx = np.reshape(np.asarray(results['idx']), 
            [nbatches, o.ntimesteps, -1])
    inputs = np.reshape(np.asarray(results['inputs']),
            [nbatches, o.ntimesteps, -1, o.ntimesteps, o.frmsz, o.frmsz, o.ninchannel])
    outputs = np.reshape(np.asarray(results['outputs']), 
            [nbatches, o.ntimesteps, -1, o.ntimesteps, 4])
    labels = np.reshape(np.asarray(results['labels']), 
            [nbatches, o.ntimesteps, -1, o.ntimesteps, 4])
    inputs_length = np.reshape(np.asarray(results['inputs_length']), 
            [nbatches, o.ntimesteps, -1])

    if o.dataset in ['moving_mnist', 'bouncing_mnist']:
        plt.gray()

    cnt = 0
    rand_batch_indices = np.random.permutation(nbatches)
    rand_expinbatch_indices = np.random.permutation(o.batchsz)
    for i in rand_batch_indices: # nbatches
        for b in rand_expinbatch_indices: # number of examples in a batch
            cnt += 1
            if cnt > nlimit:
                return 

            max_inputs_length = np.max(inputs_length[i,:,b])
            fig = plt.figure(figsize=(12,8))
            for t in range(max_inputs_length):
                plt.subplot(5,o.ntimesteps/5,t+1)
                #image
                img = inputs[i,max_inputs_length-1,b,t]
                if o.dataset in ['moving_mnist', 'bouncing_mnist']:
                    plt.imshow(np.squeeze(img, axis=2))
                else: 
                    # unnormalize using stat
                    img *= loader.stat[dstype]['std']
                    img += loader.stat[dstype]['mean']
                    plt.imshow(np.uint8(img))
                #rectangles
                box_gt = labels[i,max_inputs_length-1,b,t] * 100 # 100 scale
                box_pred = outputs[i,max_inputs_length-1,b,t] * 100
                ax = plt.gca()
                ax.add_patch(Rectangle(
                    (box_gt[0], box_gt[1]), 
                    box_gt[2]-box_gt[0], box_gt[3]-box_gt[1], 
                    facecolor='r', edgecolor='r', fill=False))
                ax.add_patch(Rectangle(
                    (box_pred[0], box_pred[1]), 
                    box_pred[2]-box_pred[0], box_pred[3]-box_pred[1], 
                    facecolor='b', edgecolor='b', fill=False))

            '''
            for t in range(o.ntimesteps):
                if t > np.max(inputs_length[i,:,b])-1:
                    break
                plt.subplot(5,o.ntimesteps/5,t+1)
                #image
                img = inputs[i,t,b,inputs_length[i,t,b]-1]
                if o.dataset in ['moving_mnist', 'bouncing_mnist']:
                    plt.imshow(np.squeeze(img, axis=2))
                else: 
                    # unnormalize using stat
                    img *= loader.stat[dstype]['std']
                    img += loader.stat[dstype]['mean']
                    plt.imshow(np.uint8(img))
                #rectangles
                box_gt = labels[i,t,b,inputs_length[i,t,b]-1] * 100 # 100 scale
                box_pred = outputs[i,t,b,inputs_length[i,t,b]-1] * 100
                ax = plt.gca()
                ax.add_patch(Rectangle(
                    (box_gt[0], box_gt[1]), 
                    box_gt[2]-box_gt[0], box_gt[3]-box_gt[1], 
                    facecolor='r', edgecolor='r', fill=False))
                ax.add_patch(Rectangle(
                    (box_pred[0], box_pred[1]), 
                    box_pred[2]-box_pred[0], box_pred[3]-box_pred[1], 
                    facecolor='b', edgecolor='b', fill=False))
            '''

            savedir = os.path.join(o.path_save, 'track_results')
            if not os.path.exists(savedir): helpers.mkdir_p(savedir)
            outfile = os.path.join(
                savedir, 'iteration_{}_{}_idx{}.png'.format(iteration, dstype, idx[i,0,b]))
            plt.savefig(outfile)
            plt.close()

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

    savedir = os.path.join(o.path_save, 'losses')
    if not os.path.exists(savedir): helpers.mkdir_p(savedir)
    outfile = os.path.join(savedir, '{}.png'.format(cnt_))
    plt.savefig(outfile)
    plt.close()

def plot_losses_train_val(loss_train, loss_val, o, cnt_):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(loss_train.shape[0]), loss_train, 'b-o')
    ax1.plot(np.arange(loss_val.shape[0]), loss_val, 'r-o')
    ax1.set_title('average intermediate loss for evaluation subsets')

    savedir = os.path.join(o.path_save, 'losses_evalsubset')
    if not os.path.exists(savedir): helpers.mkdir_p(savedir)
    outfile = os.path.join(savedir, '{}.png'.format(cnt_))
    plt.savefig(outfile)
    plt.close()

def plot_successplot(success_rates, auc, o, savedir):
    '''
    Currently, only one plot is being passed. To compare the performances 
    between other models, you will need to consider drawing all plots at once.
    (easy..)
    Also, the savedir is where the model (that is used to produce results) 
    is located .
    '''
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)
    success_rates_thresholds = np.append(np.arange(0,1,0.1), 1)
    ax1.plot(success_rates_thresholds, success_rates, 
        'b-o', label='auc:{0:.3f}'.format(auc))
    legend = ax1.legend(loc='center left')
    ax1.set_title('success plot')
    ax1.set_xlabel('overlap threshold')
    ax1.set_ylabel('success rate')

    outfile = os.path.join(savedir, 'successplot.png')
    plt.savefig(outfile)
    plt.close()

def plot_precisionplot(precision_rates, cle_representative, o, savedir):
    '''
    Currently, only one plot is being passed. To compare the performances 
    between other models, you will need to consider drawing all plots at once.
    (easy..)
    Also, the savedir is where the model (that is used to produce results) 
    is located .
    '''
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(111)
    precision_rate_thresholds = np.arange(0, 60, 10)
    ax1.plot(precision_rate_thresholds, precision_rates, 
        'b-o', label='cle_representative:{0:.3f}'.format(cle_representative))
    legend = ax1.legend(loc='center right')
    ax1.set_title('precision plot')
    ax1.set_xlabel('location error threshold (pixel)')
    ax1.set_ylabel('precision')
    outfile = os.path.join(savedir, 'precisionplot.png')
    plt.savefig(outfile)
    plt.close()

