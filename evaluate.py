import pdb
import numpy as np
import sys
import os

import draw


def evaluate(sess, m, loader, o, data_, batch_percent_=1.0, draw_=False):
    if data_ == 'train':
        datasz = loader.ntr
    elif data_ == 'val':
        datasz = loader.nva
    elif data_ == 'test':
        datasz = loader.nte
    else:
        raise ValueError('no valid data kind')

    results = {'idx': [], 'inputs': [], 'labels': [], 'outputs': []}
    #for ib in range(datasz/o.batchsz if not o.debugmode else 100):
    for ib in range(int(datasz/o.batchsz*batch_percent_) 
            if not o.debugmode else 100):
        sys.stdout.write(
                '\r(during \'{}\') evaluating {}th batch on [{}] set..'.format(
                    o.mode, ib+1, data_))
        sys.stdout.flush()
        batch = loader.get_batch(ib, o, data_=data_)

        fdict = {
                m.net['inputs']: batch['inputs'],
                m.net['inputs_length']: batch['inputs_length'],
                m.net['labels']: batch['labels']
                }
        outputs = sess.run(m.net['outputs'], feed_dict=fdict)

        results['idx'].append(batch['idx'])
        results['inputs'].append(batch['inputs'])
        results['labels'].append(batch['labels'])
        results['outputs'].append(outputs)
    print ' '

    # compute IOU
    results['IOU'] = compute_IOU_new(results['outputs'], results['labels'])
    #results['IOU'] = compute_IOU(results['outputs'], results['labels'])

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

def compute_IOU_new(outputs, labels):
    '''compute the Intersection over Union (IOU)
    Args: 
        param1: 'outputs' from model; (list)
        param2: ground truth labels corresponding to 'outputs'; (list)
    Returns:
        average IOU over all outputs
    '''
    # list to array
    boxA = np.asarray(outputs)
    boxB = np.asarray(labels)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,:,:,0], boxB[:,:,:,0])
    yA = np.maximum(boxA[:,:,:,1], boxB[:,:,:,1])
    xB = np.minimum(boxA[:,:,:,2], boxB[:,:,:,2])
    yB = np.minimum(boxA[:,:,:,3], boxB[:,:,:,3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:,:,:,2] - boxA[:,:,:,0] + 1)\
            * (boxA[:,:,:,3] - boxA[:,:,:,1] + 1)
    boxBArea = (boxB[:,:,:,2] - boxB[:,:,:,0] + 1)\
            * (boxB[:,:,:,3] - boxB[:,:,:,1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    iou = iou.astype(np.float32)

    # return the intersection over union value
    return np.mean(iou)

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

