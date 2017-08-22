import collections
import numpy as np
import sys
import os
import time
from progressbar import ProgressBar, Bar, Counter, ETA, Percentage # pip install progressbar

import draw
import data
from helpers import load_image_viewport, im_to_arr, pad_to


Model = collections.namedtuple('Model', [
    'batch_size',
    'image_size',
    'sequence_len',
    'example', # Place to feed input.
    'run_opts',
    'window',
    'prediction_crop',
    'prediction', # Place to get output.
    'init_state',
    'final_state',
])


def track(sess, model, sequence, use_gt, prediction_vars=None):
    '''Run an instantiated tracker on a sequence.

    model.sequence_len -- Integer
    model.batch_size   -- Integer or None
    model.example['x0']
    model.example['y0']
    model.example['x']
    model.example['y']
    model.run_opts['use_gt']
    model.prediction[k] for k in prediction_vars

    sequence['image_files']    -- List of strings of length n.
    sequence['labels']         -- Numpy array of shape [n, 4]
    sequence['label_is_valid'] -- List of booleans of length n.
    sequence['viewports']      -- Crop this window from the image.
    sequence['original_image_size'] -- (width, height) tuple.
        Required to compute IOU, etc. with correct aspect ratio.
    '''
    # TODO: Variable batch size.
    # TODO: Run on a batch of sequences for speed.

    if prediction_vars is None:
        prediction_vars = model.prediction.keys()

    first_image = load_image_viewport(
        sequence['image_files'][0],
        sequence['viewports'][0],
        model.image_size)
    first_label = sequence['labels'][0]
    first_image = _single_to_batch(im_to_arr(first_image), model.batch_size)
    first_label = _single_to_batch(first_label, model.batch_size)

    sequence_len = len(sequence['image_files'])
    assert(sequence_len >= 2)

    # If the length of the sequence is greater than the instantiated RNN,
    # it will need to be run in chunks.
    out_pred_chunks = {k: [] for k in prediction_vars}
    # y_pred_chunks = []
    # hmap_pred_chunks = []
    prev_state = {}
    for start in range(1, sequence_len, model.sequence_len):
        rem = sequence_len - start
        # Feed the next `chunk_len` frames into the model.
        chunk_len = min(rem, model.sequence_len)
        images = [
            load_image_viewport(image_file, viewport, model.image_size)
            for image_file, viewport in zip(
                sequence['image_files'][start:start+chunk_len],
                sequence['viewports'][start:start+chunk_len])
        ]
        images = np.array(map(im_to_arr, images))
        images = _single_to_batch(pad_to(images, model.sequence_len), model.batch_size)
        y_gt = np.array(sequence['labels'][start:start+chunk_len])
        y_gt = _single_to_batch(pad_to(y_gt, model.sequence_len), model.batch_size)
        feed_dict = {
            model.example['x0']:      first_image,
            model.example['y0']:      first_label,
            model.example['x']:       images,
            model.example['y']:       y_gt,
            model.run_opts['use_gt']: use_gt,
        }
        if start > 1:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            feed_dict.update({model.init_state[k]: prev_state[k] for k in model.init_state})
        # Get output and final state.
        out_pred, prev_state = sess.run(
            [{k: model.prediction[k] for k in prediction_vars}, model.final_state],
            feed_dict=feed_dict)
        # Take first element of batch and first `chunk_len` elements of output.
        for k in prediction_vars:
            out_pred_chunks[k].append(out_pred[k][0, :chunk_len])

    # Concatenate the results for all chunks.
    for k in prediction_vars:
        out_pred[k] = np.concatenate(out_pred_chunks[k])
    # y_pred = np.concatenate(y_pred_chunks)
    # hmap_pred = np.concatenate(hmap_pred_chunks)
    return out_pred

def _single_to_batch(x, batch_size):
    x = np.expand_dims(x, 0)
    if batch_size is None:
        return x
    return pad_to(x, batch_size)


def evaluate(sess, model, sequences, visualize=None, use_gt=False):
    '''
    Args:
        nbatches_: the number of batches to evaluate
    Returns:
        results: a dictionary that contains evaluation results
    '''
    sequences = list(sequences)
    assert len(sequences) > 0
    sequence_results = []
    pbar = ProgressBar(
        maxval=len(sequences),
        widgets=['sequence ', Counter(), '/{} ('.format(len(sequences)),
                 Percentage(), ') ', Bar(), ' ', ETA()])
    for name, sequence in pbar(sequences):
        if len(sequence['image_files']) < 2:
            continue
        is_valid = sequence['label_is_valid']
        # Count number of valid labels after first frame (ground truth).
        num_valid = sum([1 for x in is_valid[1:] if x])
        if num_valid < 1:
            continue
        #print 'sequence {} of {}'.format(i+1, len(sequences))
        possible_vars = ['y', 'score_softmax', 'hmap_softmax']
        prediction_vars = set(possible_vars).intersection(set(model.prediction.keys()))
        pred = track(sess, model, sequence, use_gt, prediction_vars)
        rect_pred = pred['y']
        hmap_pred = pred.get('score_softmax', pred.get('hmap_softmax', None))
        if visualize:
            visualize(name, sequence, rect_pred, hmap_pred)
        rect_gt = np.array(sequence['labels'])
        # Convert to original image co-ordinates.
        rect_pred = _unnormalize_rect(rect_pred, sequence['original_image_size'])
        rect_gt   = _unnormalize_rect(rect_gt,   sequence['original_image_size'])
        sequence_results.append(evaluate_track(rect_pred, rect_gt, is_valid))

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
