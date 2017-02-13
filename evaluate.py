import pdb
import numpy as np
import sys
import os

import draw


def evaluate(sess, m, loader, o, dstype, nbatches_=None, draw_=False):
    '''
    Args: 
        nbatches_: number of batches to evaluate 
    Returns:
    '''
    if o.debugmode:
        nbatches = 100
    else:
        if nbatches_ is None: # test all batches.
            # NOTE: currently not supporting remainder examples
            nbatches = int(loader.nexps[dstype]/o.batchsz) 
        else:
            nbatches = nbatches_

    results = {
            'idx': [], 
            'inputs': [], 
            'inputs_length': [], 
            'labels': [], 
            'outputs': [],
            'loss': []}

    #for ib in range(datasz/o.batchsz if not o.debugmode else 100):
    #for ib in range(nbatches if not o.debugmode else 100):
    for ib in range(nbatches):
        sys.stdout.write(
                '\r(during \'{}\') evaluating {}/{}th batch on [{}] set..'.format(
                    o.mode, ib+1, nbatches, dstype))
        sys.stdout.flush()
        batch = loader.get_batch(ib, o, dstype)

        fdict = {
                m.net['inputs']: batch['inputs'],
                m.net['inputs_length']: batch['inputs_length'],
                m.net['labels']: batch['labels']
                }
        outputs, loss = sess.run(
                [m.net['outputs'], m.net['loss']], feed_dict=fdict)

        results['idx'].append(batch['idx'])
        results['inputs'].append(batch['inputs'])
        results['inputs_length'].append(batch['inputs_length'])
        results['labels'].append(batch['labels'])
        results['outputs'].append(outputs)
        results['loss'].append(loss)
    print ' '

    results['iou'], results['success_rates'], results['auc'], results['cle'] = \
            evaluate_outputs(
                    results['outputs'], 
                    results['labels'], 
                    results['inputs_length'])

    # TODO: maybe separate eval wrapper, considering change of dataset
    if not o.nosave and draw_:
        if o.dataset == 'moving_mnist':
            draw.show_tracking_results_moving_mnist(results, o, save_=True)
        elif o.dataset == 'bouncing_mnist':
            draw.show_tracking_results_bouncing_mnist(results, o, save_=True)
            raise ValueError('not implemented yet')
        else: 
            raise ValueError('wrong dataset!')

    return results

def evaluate_outputs(outputs, labels, inputs_length):
    '''compute the Intersection over Union (IOU)
    Args: 
        param1: 'outputs' from model; (list)
        param2: ground truth labels corresponding to 'outputs'; (list)
    Returns:
        average IOU over all outputs
    '''
    # TODO:
    # 1. should not use the first frame output when evaluating.
    # 2. need to consider actual image size!!!

    # list to array
    boxA = np.asarray(outputs) * 100 # NOTE: ACTUAL IMAGE SIZE SHOULD BE CONSIDERED WHEN COMPUTING IOU.
    boxB = np.asarray(labels) * 100 # NOTE: ACTUAL IMAGE SIZE SHOULD BE CONSIDERED WHEN COMPUTING IOU.
    inputs_length = np.asarray(inputs_length)

    # for batchsz > 1; haven't checked thoroughly yet
    assert(len(boxA.shape)==4 and len(boxB.shape)==4)
    assert(len(inputs_length.shape)==2)
    boxA = np.reshape(boxA, [-1, boxA.shape[2], boxA.shape[3]])
    boxB = np.reshape(boxB, [-1, boxB.shape[2], boxB.shape[3]])
    inputs_length = np.reshape(inputs_length, [-1]) 

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,:,0], boxB[:,:,0])
    yA = np.maximum(boxA[:,:,1], boxB[:,:,1])
    xB = np.minimum(boxA[:,:,2], boxB[:,:,2])
    yB = np.minimum(boxA[:,:,3], boxB[:,:,3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:,:,2] - boxA[:,:,0] + 1)\
            * (boxA[:,:,3] - boxA[:,:,1] + 1)
    boxBArea = (boxB[:,:,2] - boxB[:,:,0] + 1)\
            * (boxB[:,:,3] - boxB[:,:,1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    iou = iou.astype(np.float32)

    # for precision computation (center location error)
    centerA_x = (boxA[:,:,0] + boxA[:,:,2])/2
    centerA_y = (boxA[:,:,1] + boxA[:,:,3])/2
    centerB_x = (boxB[:,:,0] + boxB[:,:,2])/2
    centerB_y = (boxB[:,:,1] + boxB[:,:,3])/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)

    # Success plot 
    # 1. mean iou over only valid length
    # 2. success counter over only valid length (for success plot and auc)
    # NOTE: success plot or auc can mean trash if iou is negative value. 
    # So don't make a judgement during the training, when iou can be negative.
    iou_valid = []
    success_rate_thresholds = np.arange(0,1,0.1) 
    success_rate_counter = []

    # Precision plot; 
    # TODO: need to update this using 'actual' image size (in pixels)
    # 1. center location error 
    # 2. precision plot # TODO: need pixel threshold, not available yet.
    # 3. representative precision error # TODO: need pixel threshold, not available yet.
    cle_valid = []
    
    # compute success and precision in the same loop
    for i in range(iou.shape[0]):
        for j in range(iou.shape[1]):
            if j < inputs_length[i]:
                iou_valid.append(iou[i,j])
                success_rate_counter.append(iou[i,j]>success_rate_thresholds)
                cle_valid.append(cle[i,j])
    iou_mean = np.mean(iou_valid)
    success_rate_counter = np.asarray(success_rate_counter)
    success_rates = np.mean(success_rate_counter, axis=0)
    auc = np.mean(success_rates)
    cle_mean = np.mean(cle_valid)

    return iou_mean, success_rates, auc, cle_mean

def compute_IOU(outputs, labels):
    '''compute the Intersection over Union (IOU)
    Args: 
        param1: 'outputs' from model; (list)
        param2: ground truth labels corresponding to 'outputs'; (list)
    Returns:
        average IOU over all outputs
    '''
    IOU = np.array([], dtype=np.float32)
    for ib in range(len(outputs)):
        for ie in range(outputs[ib].shape[0]):
            for t in range(outputs[ib].shape[1]):
                boxA = outputs[ib][ie,t]
                boxB = labels[ib][ie,t]
                IOU = np.append(IOU, bb_intersection_over_union(boxA, boxB))

    return np.mean(IOU) # return avg IOU

def bb_intersection_over_union(boxA, boxB):
    ''' compute IOU for a single box pair
    This is something from online.
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

