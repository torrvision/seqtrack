import pdb
import numpy as np
import sys
import os
import time
from progressbar import ProgressBar, Bar, Counter, ETA, Percentage # pip install progressbar

import draw
import data
from helpers import load_image, im_to_arr, pad_to


def track(sess, inputs, model, sequence, use_gt):
    '''Run an instantiated tracker on a sequence.

    model.outputs      -- Dictionary of tensors
    model.state        -- Dictionary of 2-tuples of tensors
    model.sequence_len -- Integer
    model.batch_size   -- Integer or None

    sequence['image_files']    -- List of strings of length n.
    sequence['labels']         -- Numpy array of shape [n, 4]
    sequence['label_is_valid'] -- List of booleans of length n.
    sequence['original_image_size'] -- (width, height) tuple.
        Required to compute IOU, etc. with correct aspect ratio.
    '''
    # TODO: Variable batch size.
    # TODO: Run on a batch of sequences for speed.

    first_image = load_image(sequence['image_files'][0], model.image_size, resize=True)
    first_label = sequence['labels'][0]
    first_image = _single_to_batch(im_to_arr(first_image), model.batch_size)
    first_label = _single_to_batch(first_label, model.batch_size)

    init_state  = {k: v[0] for k, v in model.state.iteritems()}
    final_state = {k: v[1] for k, v in model.state.iteritems()}

    sequence_len = len(sequence['image_files'])
    assert(sequence_len >= 2)

    # If the length of the sequence is greater than the instantiated RNN,
    # it will need to be run in chunks.
    y_pred_chunks = []
    hmap_pred_chunks = []
    prev_state = {}
    for start in range(1, sequence_len, model.sequence_len):
        rem = sequence_len - start
        # Feed the next `chunk_len` frames into the model.
        chunk_len = min(rem, model.sequence_len)
        images = map(lambda x: load_image(x, model.image_size, resize=True),
                     sequence['image_files'][start:start+chunk_len]) # Create single array of all images.
        images = np.array(map(im_to_arr, images))
        images = _single_to_batch(pad_to(images, model.sequence_len), model.batch_size)
        y_gt = np.array(sequence['labels'][start:start+chunk_len])
        y_gt = _single_to_batch(pad_to(y_gt, model.sequence_len), model.batch_size)
        feed_dict = {
            inputs['x_raw']:  images,
            inputs['x0_raw']: first_image,
            inputs['y0']:     first_label,
            inputs['y']:      y_gt,
            inputs['use_gt']: use_gt,
        }
        if start > 1:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            feed_dict.update({init_state[k]: prev_state[k] for k in init_state})
        # Get output and final state.
        #y_pred, prev_state = sess.run([model.outputs['y'], final_state],
        #                              feed_dict=feed_dict)
        y_pred, prev_state, hmap_pred = sess.run([model.outputs['y'], final_state,
                                                  model.outputs['hmap']],
                                      feed_dict=feed_dict)
        # Take first element of batch and first `chunk_len` elements of output.
        y_pred = y_pred[0][:chunk_len]
        y_pred_chunks.append(y_pred)
        hmap_pred = hmap_pred[0][:chunk_len]
        hmap_pred_chunks.append(hmap_pred)

    # Concatenate the results for all chunks.
    y_pred = np.concatenate(y_pred_chunks)
    hmap_pred = np.concatenate(hmap_pred_chunks)
    return y_pred, hmap_pred

def _single_to_batch(x, batch_size):
    x = np.expand_dims(x, 0)
    if batch_size is None:
        return x
    return pad_to(x, batch_size)


def evaluate(sess, inputs, model, sequences, visualize=None, use_gt=False):
    '''
    Args:
        nbatches_: the number of batches to evaluate
    Returns:
        results: a dictionary that contains evaluation results
    '''
    sequences = list(sequences)
    sequence_results = []
    pbar = ProgressBar(maxval=len(sequences),
            widgets=['sequence ', Counter(), '/{} ('.format(len(sequences)),
                Percentage(), ') ', Bar(), ' ', ETA()]).start()
    for i, sequence in enumerate(sequences):
        if len(sequence['image_files']) < 2:
            continue
        is_valid = sequence['label_is_valid']
        # Count number of valid labels after first frame (ground truth).
        num_valid = sum([1 for x in is_valid[1:] if x])
        if num_valid < 1:
            continue
        #print 'sequence {} of {}'.format(i+1, len(sequences))
        pbar.update(i+1)
        pred, hmap_pred = track(sess, inputs, model, sequence, use_gt)
        if visualize:
            visualize('sequence_{:06d}'.format(i), sequence, pred, hmap_pred)
        gt = np.array(sequence['labels'])
        # Convert to original image co-ordinates.
        pred = _unnormalize_rect(pred, sequence['original_image_size'])
        gt   = _unnormalize_rect(gt,   sequence['original_image_size'])
        sequence_results.append(evaluate_track(pred, gt, is_valid))
    pbar.finish()

    results = {}
    assert(len(sequence_results) > 0)
    for k in sequence_results[0]:
        # TODO: Store all results and compute this later!
        # (More flexible but breaks backwards compatibility.)
        results_k = np.array([r[k] for r in sequence_results])
        mean = np.mean(results_k, axis=0)
        var = np.var(results_k, axis=0)
        # Compute the standard error of the *bootstrap sample* of the mean.
        # Note that this is different from the standard deviation of a single set.
        # See page 107 of "All of Statistics" (Wasserman).
        std_err = np.sqrt(var / len(results_k))
        results[k]            = mean.tolist() # Convert to list for JSON.
        results[k+'_std_err'] = std_err.tolist()
    return results


def _normalize_rect(r, size):
    width, height = size
    return r / np.array([width, height, width, height])


def _unnormalize_rect(r, size):
    width, height = size
    return r * np.array([width, height, width, height])


def evaluate_track(pred, gt, is_valid):
    '''Measures the quality of a track compared to ground-truth.

    Args:
        pred: Numpy array with shape [t, 4]
        gt: Numpy array with shape [t+1, 4]
        is_valid: Iterable of length [t]
    '''
    # take only valid to compute (i.e., from frame 1)
    gt = gt[1:]
    is_valid = np.where(is_valid[1:])
    gt = gt[is_valid]
    pred = pred[is_valid]

    # iou and cle
    iou = _compute_iou(pred, gt)
    cle = _compute_cle(pred, gt)

    # Success plot: 1. iou, 2. success rates, 3. area under curve
    success_thresholds = np.append(np.arange(0,1,0.05), 1)
    success_thresholds = np.tile(success_thresholds, (iou.size,1))
    success_table = iou[:,np.newaxis] > success_thresholds
    iou_mean        = np.mean(iou)
    success_rates   = np.mean(success_table, axis=0)
    auc             = np.mean(success_rates)

    # Precision plot: 1. cle, 2. precision rates, 3. representative precision error
    precision_thresholds = np.arange(0, 60, 5)
    precision_thresholds_rep = np.tile(precision_thresholds, (cle.size,1))
    precision_table = cle[:,np.newaxis] < precision_thresholds_rep
    representative_precision_threshold = 20 # benchmark
    cle_mean            = np.mean(cle)
    precision_rates     = np.mean(precision_table, axis=0)
    cle_representative  = precision_rates[
        np.where(precision_thresholds == representative_precision_threshold)][0]

    results = {}
    results['iou_mean']             = iou_mean
    results['success_rates']        = success_rates
    results['auc']                  = auc
    results['cle_mean']             = cle_mean
    results['precision_rates']      = precision_rates
    results['cle_representative']   = cle_representative
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
    boxAArea = ((boxA[:,2] - boxA[:,0]) * (boxA[:,3] - boxA[:,1]))
    boxBArea = ((boxB[:,2] - boxB[:,0]) * (boxB[:,3] - boxB[:,1]))

    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the
    # interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou.astype(np.float32)

def _compute_cle(boxA, boxB):
    # compute center location error
    centerA_x = (np.array(boxA[:,0]) + np.array(boxA[:,2]))/2
    centerA_y = (np.array(boxA[:,1]) + np.array(boxA[:,3]))/2
    centerB_x = (np.array(boxB[:,0]) + np.array(boxB[:,2]))/2
    centerB_y = (np.array(boxB[:,1]) + np.array(boxB[:,3]))/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)
    return cle
