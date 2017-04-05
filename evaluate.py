import pdb
import numpy as np
import sys
import os
import time

import draw
import data
from helpers import load_image, im_to_arr


def track(sess, inputs, model, sequence):
    '''Run an instantiated tracker on a sequence.

    model.outputs      -- Dictionary of tensors
    model.state        -- Dictionary of 2-tuples of tensors
    model.sequence_len -- Integer
    model.batch_size   -- Integer or None

    sequence['image_files'] -- List of strings
    sequence['labels']      -- Numpy array
    '''
    # TODO: Variable batch size.
    # TODO: Run on a batch of sequences for speed.

    first_image = load_image(sequence['image_files'][0], model.image_size)
    first_label = sequence['labels'][0]
    first_image = _single_to_batch(im_to_arr(first_image), model.batch_size)
    first_label = _single_to_batch(first_label, model.batch_size)

    init_state  = {k: v[0] for k, v in model.state.iteritems()}
    final_state = {k: v[1] for k, v in model.state.iteritems()}

    sequence_len = len(sequence['image_files'])

    # If the length of the sequence is greater than the instantiated RNN,
    # it will need to be run in chunks.
    y_chunks = []
    prev_state = {}
    for start in range(1, sequence_len, model.sequence_len):
        rem = sequence_len - start
        # Feed the next `chunk_len` frames into the model.
        chunk_len = min(rem, model.sequence_len)
        images = map(lambda x: load_image(x, model.image_size),
                     sequence['image_files'][start:start+chunk_len])
        # Create single array of all images.
        images = np.array(map(im_to_arr, images))
        images = _single_to_batch(pad_to(images, model.sequence_len), model.batch_size)
        # Create fake y values.
        labels = np.zeros(list(images.shape[:2])+[4])
        feed_dict = {
            inputs['x_raw']:  images,
            inputs['x0_raw']: first_image,
            inputs['y0']:     first_label,
            inputs['y']:      labels,
        }
        if start > 1:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            feed_dict.update({init_state[k]: prev_state[k] for k in init_state})
        # Get output and final state.
        # TODO: Check that this works when `final_state` is an empty dictionary.
        y, prev_state = sess.run([model.outputs['y'], final_state],
                                 feed_dict=feed_dict)
        # Take first element of batch and first `chunk_len` elements of output.
        y = y[0][:chunk_len]
        y_chunks.append(y)

    # Concatenate the results for all chunks.
    y = np.concatenate(y_chunks)
    return y

def pad_to(x, n, mode='constant', axis=0):
    width = [(0, 0) for s in x.shape]
    width[axis] = (0, n - x.shape[axis])
    return np.pad(x, width, mode=mode)

def _single_to_batch(x, batch_size):
    # TODO: Pad to sequence length and batch size?
    x = np.expand_dims(x, 0)
    if batch_size is None:
        return x
    return pad_to(x, batch_size)


def evaluate(sess, inputs, model, sequences):
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

    sequences = list(sequences)
    sequence_results = []
    for i, sequence in enumerate(sequences):
        print 'sequence {} of {}'.format(i+1, len(sequences))
        pred = track(sess, inputs, model, sequence)
        gt = np.array(sequence['labels'])
        is_valid = [True] * len(pred) # sequence['valid']
        # TODO: Convert to original image co-ordinates.
        sequence_results.append(evaluate_track(pred, gt, is_valid))

    results = {}
    for k in sequence_results[0]:
        results[k] = np.mean([r[k] for r in sequence_results], axis=0)
    return results


def evaluate_track(pred, gt, is_valid):
    '''Measures the quality of a track compared to ground-truth.

    Args:
        pred: Numpy array with shape [t, 4]
        gt: Numpy array with shape [t+1, 4]
        is_valid: Iterable of length [t]
    '''
    # take only valid to compute (i.e., from frame 1)
    gt = gt[1:]
    gt = gt[is_valid]
    pred = pred[is_valid]

    # JV: Do this outside this function, and use original aspect ratio?
    # # back to original scale (pixel)
    # scalar = np.concatenate((inputs_HW[:,:,[1]], inputs_HW[:,:,[0]], 
    #     inputs_HW[:,:,[1]], inputs_HW[:,:,[0]]), axis=2)
    # pred *= np.expand_dims(scalar, 2)
    # gt *= np.expand_dims(scalar, 2)

    iou = _compute_iou(pred, gt)
    cle = _compute_precision(pred, gt)

    # Success plot 
    # 1. mean iou over only valid length
    # 2. success counter over only valid length (for success plot and auc)
    # 3. area under curve
    # iou_valid = iou[is_valid]
    iou_valid = iou
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
    # cle_valid = cle[is_valid]
    cle_valid = cle
    precision_rate_thresholds = np.arange(0, 60, 5)
    precision_rate_thresholds = np.tile(
            precision_rate_thresholds, (cle_valid.size,1))
    precision_rate_table = cle_valid[:,np.newaxis] < precision_rate_thresholds
    representative_precision_threshold = 20 # benchmark

    cle_mean = np.mean(cle_valid)
    precision_rates = np.mean(precision_rate_table, axis=0)
    cle_representative = np.mean(
            cle_valid[cle_valid < representative_precision_threshold])

    results = {}
    results['iou_mean'] = iou_mean
    results['success_rates'] = success_rates
    results['auc'] = auc
    results['cle_mean'] = cle_mean
    results['precision_rates'] = precision_rates
    results['cle_representative'] = cle_representative
    return results

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[:,0], boxB[:,0])
    yA = np.maximum(boxA[:,1], boxB[:,1])
    xB = np.minimum(boxA[:,2], boxB[:,2])
    yB = np.minimum(boxA[:,3], boxB[:,3])

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = ((boxA[:,2] - boxA[:,0]) *
                (boxA[:,3] - boxA[:,1]))
    boxBArea = ((boxB[:,2] - boxB[:,0]) *
                (boxB[:,3] - boxB[:,1]))

    # compute the intersection over union by taking the intersection area and 
    # dividing it by the sum of prediction + ground-truth areas - the 
    # interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou.astype(np.float32)

def _compute_precision(boxA, boxB):
    # for precision computation (center location error)
    centerA_x = (np.array(boxA[:,0]) + np.array(boxA[:,2]))/2
    centerA_y = (np.array(boxA[:,1]) + np.array(boxA[:,3]))/2
    centerB_x = (np.array(boxB[:,0]) + np.array(boxB[:,2]))/2
    centerB_y = (np.array(boxB[:,1]) + np.array(boxB[:,3]))/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)
    return cle
