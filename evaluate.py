import pdb
import numpy as np
import sys
import os
import time

import draw
import data


def evaluate(sess, m, loader, o, dstype, nbatches_=None, hold_inputs=False, 
        shuffle_local=False, fulllen=False):
    '''
    Args: 
        nbatches_: the number of batches to evaluate 
    Returns:
        results: a dictionary that contains evaluation results
    '''
    #--------------------------------------------------------------------------
    # NOTE: Proper evaluation
    # - Instead of passing all x and receiving y, and evaluate at once, 
    #   the system should yield each y one by one at each time step.
    # - Evaluation should also be made by taking one output at a time.
    #   Otherwise, it's completely offline evaluation, which cannot be the main 
    #   supporting experiment. -> performed by 'split_batch_fortest'
    # - If an example in the batch is loger than T (RNN size), 
    #   forward-pass should run multiple times.
    #--------------------------------------------------------------------------

    results = {
            'idx': [], 
            'inputs': [], # TOO large to have in memory
            'inputs_valid': [], 
            'inputs_HW': [], 
            'labels': [], 
            'outputs': [],
            'loss': []
            }

    if not fulllen:
        if o.debugmode:
            nbatches = 10
        else:
            if nbatches_ is None: # test all batches. 
                assert(False) # unlikely happen 
                nbatches = int(loader.nexps[dstype]/o.batchsz) # NOTE: currently not supporting remainders
            else:
                nbatches = nbatches_
        results['nbatches'] = nbatches

        for ib in range(nbatches):
            t_start = time.time()
            batch = loader.get_batch(ib, o, dstype, shuffle_local=shuffle_local)
            # process each time step one by one
            for t in range(1, o.ntimesteps+1): # inputs has o.ntimesteps+1 length
                batch_split = get_batch_split_fortest(batch, t) # this is time-wise split

                fdict = {
                    m.net['inputs']: batch_split['inputs'],
                    m.net['inputs_valid']: batch_split['inputs_valid'],
                    m.net['inputs_HW']: batch_split['inputs_HW'],
                    m.net['labels']: batch_split['labels']
                    }

                outputs, loss = sess.run(
                        [m.net['outputs'], m.net['loss']], feed_dict=fdict)

                if 'idx' in batch_split: results['idx'].append(batch_split['idx'])
                if hold_inputs:
                    results['inputs'].append(batch_split['inputs']) # no memory
                results['inputs_valid'].append(batch_split['inputs_valid'])
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

        results = evaluate_outputs(results)
        return results

    
    # Main evaluation routine for full-length sequences.
    else:
        nexps = loader.nexps_fulllen if not o.debugmode else 10
        for i in range(nexps): # one full-length example at a time
            batch_fl = loader.get_batch_fl(i, o)
            batch = data.split_batch_fulllen_seq(batch_fl, o)

            # NOTE: until implementing variable-size batchsz model, split 
            # the batch so that it has o.batchsz shape at 1st dim.
            nsplits = int(np.ceil(batch['nsegments'] / float(o.batchsz)))
            for j in range(nsplits): # to fit 'batchsz' in model definition
                t_start = time.time()
                batch_seg = {}
                for key in ['inputs', 'inputs_valid', 'inputs_HW', 'labels']:
                    batch_seg[key] = batch[key][j*o.batchsz:(j+1)*o.batchsz]

                # if batch_seg is not of length=o.batchsz, zero-padding
                batchsz_curr = batch_seg['inputs'].shape[0]
                if batchsz_curr < o.batchsz:
                    pad = {}
                    pad['inputs'] = np.zeros(
                        [o.batchsz-batchsz_curr, o.ntimesteps+1, 
                            o.frmsz, o.frmsz, o.ninchannel], dtype=np.float32)
                    pad['inputs_valid'] = np.zeros(
                        [o.batchsz-batchsz_curr, o.ntimesteps+1], dtype=np.bool)
                    pad['inputs_HW'] = np.zeros(
                        [o.batchsz-batchsz_curr, 2], dtype=np.bool)
                    pad['labels'] = np.zeros(
                        [o.batchsz-batchsz_curr, o.ntimesteps+1, o.outdim], 
                        dtype=np.bool)
                    batch_seg['inputs'] = np.concatenate(
                        (batch_seg['inputs'], pad['inputs']), axis=0)
                    batch_seg['inputs_valid'] = np.concatenate(
                        (batch_seg['inputs_valid'], pad['inputs_valid']),axis=0)
                    batch_seg['inputs_HW'] = np.concatenate(
                        (batch_seg['inputs_HW'], pad['inputs_HW']), axis=0)
                    batch_seg['labels'] = np.concatenate(
                        (batch_seg['labels'], pad['labels']), axis=0)

                for t in range(1, o.ntimesteps+1):
                    batch_split = get_batch_split_fortest(batch_seg, t) # this is time-wise split
                    if j == 0: # first segment
                        fdict = {
                            m.net['inputs']: batch_split['inputs'],
                            m.net['inputs_valid']: batch_split['inputs_valid'],
                            m.net['inputs_HW']: batch_split['inputs_HW'],
                            m.net['labels']: batch_split['labels']
                            }
                    else: # from second segment need to pass previous outputs
                        fdict = {
                            m.net['inputs']: batch_split['inputs'],
                            m.net['inputs_valid']: batch_split['inputs_valid'],
                            m.net['inputs_HW']: batch_split['inputs_HW'],
                            m.net['labels']: batch_split['labels'],
                            m.net['h_init']: h_last,
                            m.net['C_init']: C_last,
                            m.net['y_init']: y_last
                            }

                    outputs, loss, h_last, C_last, y_last = sess.run(
                        [m.net['outputs'], m.net['loss'], 
                            m.net['h_last'], m.net['C_last'], m.net['y_last']], 
                        feed_dict=fdict)

                    if 'idx' in batch_split: 
                        results['idx'].append(batch_split['idx'])
                    if hold_inputs:
                        results['inputs'].append(batch_split['inputs']) # no memory
                    results['inputs_valid'].append(batch_split['inputs_valid'])
                    results['inputs_HW'].append(batch_split['inputs_HW'])
                    results['labels'].append(batch_split['labels'])
                    results['outputs'].append(outputs)
                    results['loss'].append(loss)
                sys.stdout.write(
                    '\r(during \'{0:s}\') passed {1:d}/{2:d} exp, {3:d}/{4:d} '\
                    'segment on [{5:s}] set.. |time: {6:.3f}'.format(o.mode, 
                        i+1, nexps, j+1, nsplits, dstype, time.time()-t_start))
                sys.stdout.flush()
        print ' '

        results = evaluate_outputs(results)
        return results


def get_batch_split_fortest(batch, t):
    '''
    Create a batch split by fetching at specific time t and padding with zeros.
    '''
    batchsz = batch['inputs'].shape[0]
    T = batch['inputs'].shape[1]

    inputs = np.zeros_like(batch['inputs'])
    inputs_valid = np.zeros_like(batch['inputs_valid'], dtype=np.bool)
    labels = np.zeros_like(batch['labels'])

    #if t == 2: 
        #pdb.set_trace()

    for b in range(batchsz):
        inputs[b, :t+1] = batch['inputs'][b, :t+1]
        # NOTE: only valid at specific time step!
        inputs_valid[b, t] = batch['inputs_valid'][b, t]
        labels[b, :t+1] = batch['labels'][b, :t+1]

    splits = {}
    splits['inputs'] = inputs
    splits['inputs_valid'] = inputs_valid
    splits['inputs_HW'] = batch['inputs_HW'] # no time dependent
    splits['labels'] = labels
    if 'idx' in batch: 
        splits['idx'] = batch['idx'] # no time dependent
    return splits

def evaluate_outputs(results):
    # to numpy array and take only valid to compute (i.e., from frame 1)
    boxA = np.asarray(results['outputs'])
    boxB = np.asarray(results['labels'])
    boxB = boxB[:,:,1:,:]
    inputs_valid = np.asarray(results['inputs_valid'])
    inputs_valid = inputs_valid[:,:,1:]
    inputs_HW = np.asarray(results['inputs_HW'])

    # back to original scale (pixel)
    scalar = np.concatenate((inputs_HW[:,:,[1]], inputs_HW[:,:,[0]], 
        inputs_HW[:,:,[1]], inputs_HW[:,:,[0]]), axis=2)
    boxA *= np.expand_dims(scalar, 2)
    boxB *= np.expand_dims(scalar, 2)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,:,:,0], boxB[:,:,:,0])
    yA = np.maximum(boxA[:,:,:,1], boxB[:,:,:,1])
    xB = np.minimum(boxA[:,:,:,2], boxB[:,:,:,2])
    yB = np.minimum(boxA[:,:,:,3], boxB[:,:,:,3])

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[:,:,:,2] - boxA[:,:,:,0])\
            * (boxA[:,:,:,3] - boxA[:,:,:,1])
    boxBArea = (boxB[:,:,:,2] - boxB[:,:,:,0])\
            * (boxB[:,:,:,3] - boxB[:,:,:,1])

    # compute the intersection over union by taking the intersection area and 
    # dividing it by the sum of prediction + ground-truth areas - the 
    # interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    iou = iou.astype(np.float32)

    # for precision computation (center location error)
    centerA_x = (boxA[:,:,:,0] + boxA[:,:,:,2])/2
    centerA_y = (boxA[:,:,:,1] + boxA[:,:,:,3])/2
    centerB_x = (boxB[:,:,:,0] + boxB[:,:,:,2])/2
    centerB_y = (boxB[:,:,:,1] + boxB[:,:,:,3])/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)

    # Success plot 
    # 1. mean iou over only valid length
    # 2. success counter over only valid length (for success plot and auc)
    # 3. area under curve
    iou_valid = iou[inputs_valid]
    success_rate_thresholds = np.append(np.arange(0,1,0.05), 1)
    success_rate_thresholds = np.tile(
            success_rate_thresholds, (iou_valid.size,1))
    success_rate_table = iou_valid[:,np.newaxis] > success_rate_thresholds

    iou_mean = np.mean(iou_valid)
    success_rates = np.mean(success_rate_table, axis=0)
    auc = np.mean(success_rates)

    # Precision plot; 
    # 1. center location error 
    # 2. precision plot
    # 3. representative precision error
    cle_valid = cle[inputs_valid]
    precision_rate_thresholds = np.arange(0, 60, 5)
    precision_rate_thresholds = np.tile(
            precision_rate_thresholds, (cle_valid.size,1))
    precision_rate_table = cle_valid[:,np.newaxis] < precision_rate_thresholds
    representative_precision_threshold = 20 # benchmark

    cle_mean = np.mean(cle_valid)
    precision_rates = np.mean(precision_rate_table, axis=0)
    cle_representative = np.mean(
            cle_valid[cle_valid < representative_precision_threshold])

    results['iou_mean'] = iou_mean
    results['success_rates'] = success_rates
    results['auc'] = auc
    results['cle_mean'] = cle_mean
    results['precision_rates'] = precision_rates
    results['cle_representative'] = cle_representative

    return results

