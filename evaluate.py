import pdb
import numpy as np
import sys
import os
import time

import draw


def evaluate(sess, m, loader, o, dstype, nbatches_=None, hold_inputs=False, 
        shuffle_local=False):
    '''
    Args: 
        nbatches_: the number of batches to evaluate 
    Returns:
        results: a dictionary that contains evaluation results
    '''
    #--------------------------------------------------------------------------
    # NOTE: 
    # 1. (IMPORTANT) Proper evaluation
    # - Instead of passing all x and receiving y, and evaluate at once, 
    #   the system should yield each y one by one at each time step.
    # - Evaluation should also be made by taking one output at a time.
    #   Otherwise, it's completely offline evaluation, which cannot be the main 
    #   supporting experiment. -> performed by 'split_batch_fortest'
    # - (TODO) If an example in the batch is loger than T (RNN size), 
    #   forward-pass should run multiple times.
    # - Maybe not 'correctly' evaluating during training is okay for now, since 
    #   the network is trained naively (using the whole sequence at once), 
    #   but it must be done correctly during test. -> should be deprecated
    # 
    # 2. Increasing batchsz
    # - You can increase batchsz to speed up, but it didn't reproduce exactly
    #   same results that was obtained with batchsz=1. Didn't look into details.
    # - For official report, use batch size of 1 so that you don't loose 
    #   remainders. This also means that the model should have been trained 
    #   using batchsz of 1 (input tensor size fixed). Otherwise you can't load
    #   the model and run. TODO: need imporevement here..
    #--------------------------------------------------------------------------

    if o.debugmode:
        nbatches = 10
    else:
        if nbatches_ is None: # test all batches. 
            assert(o.batchsz==1)
            nbatches = int(loader.nexps[dstype]/o.batchsz) # NOTE: currently not supporting remainders
        else:
            nbatches = nbatches_

    results = {
            'idx': [], 
            'inputs': [], # TOO large to have in memory
            'inputs_length': [], 
            'inputs_HW': [], 
            'labels': [], 
            'outputs': [],
            'loss': [],
            'nbatches': nbatches}

    #for ib in range(datasz/o.batchsz if not o.debugmode else 100):
    #for ib in range(nbatches if not o.debugmode else 100):
    for ib in range(nbatches):
        t_start = time.time()
        batch = loader.get_batch(ib, o, dstype, shuffle_local=shuffle_local)
        # process each time step one by one
        for t in range(o.ntimesteps):
            batch_split = get_batch_split_fortest(batch, t)

            fdict = {
                m.net['inputs']: batch_split['inputs'],
                m.net['inputs_length']: batch_split['inputs_length'],
                m.net['inputs_HW']: batch_split['inputs_HW'],
                m.net['labels']: batch_split['labels']
                }
            # NOTE: only output up until inputs_length are valid
            outputs, loss = sess.run(
                    [m.net['outputs'], m.net['loss']], feed_dict=fdict)

            results['idx'].append(batch_split['idx'])
            if hold_inputs:
                results['inputs'].append(batch_split['inputs']) # no memory
            results['inputs_length'].append(batch_split['inputs_length'])
            results['inputs_HW'].append(batch_split['inputs_HW'])
            results['labels'].append(batch_split['labels'])
            results['outputs'].append(outputs)
            results['loss'].append(loss)
        sys.stdout.write(
                '\r(during \'{0:s}\') passed {1:d}/{2:d}th batch on '\
                    '[{3:s}] set.. |time: {4:.3f}'.format(
                    o.mode, ib+1, nbatches, dstype, time.time()-t_start))
        sys.stdout.flush()
    print ' '

    results['iou_mean'], results['success_rates'], results['auc'], \
    results['cle_mean'], results['precision_rates'], results['cle_representative'] = \
        evaluate_outputs(
            results['outputs'], results['labels'], 
            results['inputs_length'], results['inputs_HW'], 
            nbatches, o)

    return results

def get_batch_split_fortest(batch, t):
    '''
    Create a batch split by fetching at specific time t and padding with zeros.
    '''
    batchsz = batch['inputs'].shape[0]
    T = batch['inputs'].shape[1]

    inputs = np.zeros_like(batch['inputs'])
    inputs_length = np.zeros_like(batch['inputs_length'])
    labels = np.zeros_like(batch['labels'])

    for b in range(batchsz):
        inputs[b, :t+1] = batch['inputs'][b, :t+1]
        if t < batch['inputs_length'][b]:
            inputs_length[b] = t+1
        else:
            inputs_length[b] = batch['inputs_length'][b]
        labels[b, :t+1] = batch['labels'][b, :t+1]

    splits = {}
    splits['inputs'] = inputs
    splits['inputs_length'] = inputs_length 
    splits['inputs_HW'] = batch['inputs_HW'] # no time dependent
    splits['labels'] = labels
    splits['idx'] = batch['idx'] # no time dependent
    return splits

def evaluate_outputs(outputs, labels, inputs_length, inputs_HW, nbatches, o):
    '''compute the Intersection over Union (IOU)
    Args: 
        outputs: 'outputs' from model; (list)
        labels: ground truth labels corresponding to 'outputs'; (list)
        inputs_length: valid input length 
    Returns:
        evaluation results..
    '''
    # TODO:
    # - test when the segment is not the first segment of full sequence. 
    
    # list to array
    boxA = np.asarray(outputs)
    boxB = np.asarray(labels)
    inputs_length = np.asarray(inputs_length)
    inputs_HW = np.asarray(inputs_HW)

    # reshape the first dimension into nbatches and ntimesteps
    batchsz_est = boxA.shape[1]
    assert(batchsz_est == o.batchsz)
    boxA = np.reshape(boxA, [nbatches, o.ntimesteps, -1, o.ntimesteps, 4]) 
    boxB = np.reshape(boxB, [nbatches, o.ntimesteps, -1, o.ntimesteps, 4]) 
    inputs_length = np.reshape(inputs_length, [nbatches, o.ntimesteps, -1])
    inputs_HW = np.reshape(inputs_HW, [nbatches, o.ntimesteps, -1, 2])

    # back to original scale (pixel)
    scalar = np.concatenate((inputs_HW[:,:,:,[1]], inputs_HW[:,:,:,[0]], 
        inputs_HW[:,:,:,[1]], inputs_HW[:,:,:,[0]]), axis=3)
    boxA *= np.expand_dims(scalar, 3)
    boxB *= np.expand_dims(scalar, 3)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,:,:,:,0], boxB[:,:,:,:,0])
    yA = np.maximum(boxA[:,:,:,:,1], boxB[:,:,:,:,1])
    xB = np.minimum(boxA[:,:,:,:,2], boxB[:,:,:,:,2])
    yB = np.minimum(boxA[:,:,:,:,3], boxB[:,:,:,:,3])

    # compute the area of intersection rectangle
    # NOTE: Wrong. Adding 1 makes huge change
    #interArea = (xB - xA + 1) * (yB - yA + 1) 
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # compute the area of both the prediction and ground-truth rectangles
    # NOTE: Wrong. Adding 1 makes huge change
    '''
    boxAArea = (boxA[:,:,:,:,2] - boxA[:,:,:,:,0] + 1)\
            * (boxA[:,:,:,:,3] - boxA[:,:,:,:,1] + 1)
    boxBArea = (boxB[:,:,:,:,2] - boxB[:,:,:,:,0] + 1)\
            * (boxB[:,:,:,:,3] - boxB[:,:,:,:,1] + 1)
    '''
    boxAArea = (boxA[:,:,:,:,2] - boxA[:,:,:,:,0])\
            * (boxA[:,:,:,:,3] - boxA[:,:,:,:,1])
    boxBArea = (boxB[:,:,:,:,2] - boxB[:,:,:,:,0])\
            * (boxB[:,:,:,:,3] - boxB[:,:,:,:,1])

    # compute the intersection over union by taking the intersection area and 
    # dividing it by the sum of prediction + ground-truth areas - the 
    # interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    iou = iou.astype(np.float32)

    # for precision computation (center location error)
    centerA_x = (boxA[:,:,:,:,0] + boxA[:,:,:,:,2])/2
    centerA_y = (boxA[:,:,:,:,1] + boxA[:,:,:,:,3])/2
    centerB_x = (boxB[:,:,:,:,0] + boxB[:,:,:,:,2])/2
    centerB_y = (boxB[:,:,:,:,1] + boxB[:,:,:,:,3])/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)

    # Success plot 
    # 1. mean iou over only valid length
    # 2. success counter over only valid length (for success plot and auc)
    # 3. area under curve
    iou_valid = []
    success_rate_thresholds = np.append(np.arange(0,1,0.1), 1)
    success_rate_counter = []

    # Precision plot; 
    # 1. center location error 
    # 2. precision plot
    # 3. representative precision error
    cle_valid = []
    precision_rate_thresholds = np.arange(0, 60, 10)
    precision_rate_counter = []
    representative_precision_threshold = 20 # benchmark
    
    # compute success and precision in the same loop
    for i in range(iou.shape[0]): # nbatches
        for t in range(iou.shape[1]): # time splits
            # TODO: besides t==0 condition, need another indicator telling 
            # whether it is first segment or not!
            if t == 0: 
                continue # don't evaluate the output at the first frame
            for b in range(iou.shape[2]): # batchsz 
                # TODO: negative iou score spotted in a few places.
                # might be better to have loss term on iou.
                if iou[i,t,b,inputs_length[i,t,b]-1] < 0:
                    pdb.set_trace()
                iou_valid.append(iou[i,t,b,inputs_length[i,t,b]-1])
                success_rate_counter.append(
                    iou[i,t,b,inputs_length[i,t,b]-1] > success_rate_thresholds)
                cle_valid.append(cle[i,t,b,inputs_length[i,t,b]-1])
                precision_rate_counter.append(
                    cle[i,t,b,inputs_length[i,t,b]-1]<precision_rate_thresholds)

    iou_valid = np.asarray(iou_valid)
    iou_mean = np.mean(iou_valid)
    success_rate_counter = np.asarray(success_rate_counter)
    success_rates = np.mean(success_rate_counter, axis=0)
    auc = np.mean(success_rates)

    precision_rate_counter = np.asarray(precision_rate_counter)
    precision_rates = np.mean(precision_rate_counter, axis=0)
    cle_valid = np.asarray(cle_valid)
    cle_mean = np.mean(cle_valid)
    cle_representative = np.mean(
            cle_valid[cle_valid < representative_precision_threshold])

    return iou_mean, success_rates, auc, \
            cle_mean, precision_rates, cle_representative


