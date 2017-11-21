import pdb
import numpy as np
import sys
import os
import progressbar
import shutil
import subprocess
import tempfile
import time

import draw
import data
from helpers import load_image, load_image_viewport, im_to_arr, pad_to, to_nested_tuple
from helpers import escape_filename
import visualize as visualize_pkg

FRAME_PATTERN = '%06d.jpeg'

import matplotlib.pyplot as plt
mycmap = visualize_pkg.transparent_cmap(plt.cm.hot)


def track(sess, inputs, model, sequence, use_gt,
        # Visualization options:
        visualize=False,
        vis_dir=None,
        save_frames=False,
        save_original=False):
    '''Run an instantiated tracker on a sequence.

    model.outputs      -- Dictionary of tensors
    model.state_init   -- Nested collection of tensors.
    model.state_final  -- Nested collection of tensors.
    model.sequence_len -- Integer
    model.batch_size   -- Integer or None

    sequence['image_files']    -- List of strings of length n.
    sequence['viewports']      -- Numpy array of rectangles [n, 4].
    sequence['labels']         -- Numpy array of shape [n, 4]
    sequence['label_is_valid'] -- List of booleans of length n.
    sequence['aspect']         -- Aspect ratio of original image.
    sequence['original_image_size'] -- (width, height) tuple.
        Required to compute IOU, etc. with correct aspect ratio.
    '''
    # TODO: Variable batch size.
    # TODO: Run on a batch of sequences for speed.

    if visualize:
        assert vis_dir is not None
        if not os.path.exists(vis_dir): os.makedirs(vis_dir, 0755)
        if not save_frames:
            frame_dir = tempfile.mkdtemp()
        else:
            frame_dir = os.path.join(vis_dir, 'frames', escape_filename(sequence['video_name']))
            if not os.path.exists(frame_dir): os.makedirs(frame_dir)

    # JV: Use viewport.
    # first_image = load_image(sequence['image_files'][0], model.image_size, resize=True)
    first_image = load_image_viewport(
        sequence['image_files'][0],
        sequence['viewports'][0],
        model.image_size)
    first_label = sequence['labels'][0]
    # Prepare for input to network.
    batch_first_image = _single_to_batch(im_to_arr(first_image), model.batch_size)
    batch_first_label = _single_to_batch(first_label, model.batch_size)

    if visualize:
        if save_original:
            if 'frmsz257' in sequence['image_files'][0]:
                visualize_pkg.draw_output_mpl(
                        load_image(sequence['image_files'][0].replace('images_frmsz257', 'images')),
                        rect_gt=first_label,
                        fname=os.path.join(frame_dir, FRAME_PATTERN % 0),
                        cmap=mycmap)
            else:
                visualize_pkg.draw_output_mpl(
                        load_image(sequence['image_files'][0]),
                        rect_gt=first_label,
                        fname=os.path.join(frame_dir, FRAME_PATTERN % 0),
                        cmap=mycmap)
        else:
            im_vis = visualize_pkg.draw_output(first_image.copy(), rect_gt=first_label)
            im_vis.save(os.path.join(frame_dir, FRAME_PATTERN % 0))

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
        # JV: Use viewport.
        # images = map(lambda x: load_image(x, model.image_size, resize=True),
        #              sequence['image_files'][start:start+chunk_len]) # Create single array of all images.
        images = [
            load_image_viewport(image_file, viewport, model.image_size)
            for image_file, viewport in zip(
                sequence['image_files'][start:start+chunk_len],
                sequence['viewports'][start:start+chunk_len])
        ]
        labels = sequence['labels'][start:start+chunk_len]
        is_valid = sequence['label_is_valid'][start:start+chunk_len]

        # Prepare data as input to network.
        images_arr = map(im_to_arr, images)
        feed_dict = {
            inputs['x0_raw']:     batch_first_image,
            inputs['y0']:         batch_first_label,
            inputs['x_raw']:      _single_to_batch(pad_to(images_arr, model.sequence_len), model.batch_size),
            inputs['y']:          _single_to_batch(pad_to(labels, model.sequence_len), model.batch_size),
            inputs['y_is_valid']: _single_to_batch(pad_to(is_valid, model.sequence_len), model.batch_size),
            inputs['aspect']:     _single_to_batch(sequence['aspect'], model.batch_size),
            inputs['use_gt']:     use_gt,
        }
        if start > 1:
            # This is not the first chunk.
            # Add the previous state to the feed dictionary.
            tensor, value = to_nested_tuple(model.state_init, prev_state)
            if tensor is not None: # Function returns None if empty.
                feed_dict[tensor] = value
        # Get output and final state.
        y_pred, prev_state, hmap_pred = sess.run(
            [model.outputs['y']['ic'], model.state_final, model.outputs['hmap']['ic']],
            feed_dict=feed_dict)
        # Take first element of batch and first `chunk_len` elements of output.
        y_pred = y_pred[0][:chunk_len]
        hmap_pred = hmap_pred[0][:chunk_len]

        if visualize:
            if save_original:
                if 'frmsz257' in sequence['image_files'][0]:
                    images = map(lambda x: load_image(x.replace('images_frmsz257', 'images')),
                                 sequence['image_files'][start:start+chunk_len]) # Create single array of all images.
                else:
                    images = map(lambda x: load_image(x),
                                 sequence['image_files'][start:start+chunk_len]) # Create single array of all images.
                for i in range(len(images)):
                    t = start + i
                    visualize_pkg.draw_output_mpl(
                            images[i],
                            (labels[i] if is_valid[i] else None),
                            y_pred[i],
                            hmap_pred[i],
                            os.path.join(frame_dir, FRAME_PATTERN % t),
                            mycmap)
            else:
                for i in range(len(images)):
                    t = start + i
                    im_vis = visualize_pkg.draw_output(images[i].copy(),
                        rect_gt=(labels[i] if is_valid[i] else None),
                        rect_pred=y_pred[i],
                        hmap_pred=hmap_pred[i])
                    im_vis.save(os.path.join(frame_dir, FRAME_PATTERN % t))

        y_pred_chunks.append(y_pred)
        hmap_pred_chunks.append(hmap_pred)

    if visualize:
        args = ['ffmpeg', '-loglevel', 'error',
                          # '-r', '1', # fps.
                          '-y', # Overwrite without asking.
                          '-nostdin', # No interaction with user.
                          '-i', FRAME_PATTERN,
                          '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                          os.path.join(os.path.abspath(vis_dir),
                                       escape_filename(sequence['video_name'])+'.mp4')]
        try:
            p = subprocess.Popen(args, cwd=frame_dir)
            p.wait()
        except Exception as inst:
            print 'error:', inst
        finally:
            if not save_frames:
                shutil.rmtree(frame_dir)

    # Concatenate the results for all chunks.
    y_pred = np.concatenate(y_pred_chunks)
    hmap_pred = np.concatenate(hmap_pred_chunks)
    return y_pred, hmap_pred

def _single_to_batch(x, batch_size):
    x = np.expand_dims(x, 0)
    if batch_size is None:
        return x
    return pad_to(x, batch_size)


def _make_progress_bar():
    return progressbar.ProgressBar(widgets=[
        progressbar.SimpleProgress(format='sequence %(value_s)s/%(max_value_s)s'), ' ',
        '(', progressbar.Percentage(), ') ',
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' (', progressbar.ETA(format_finished='ETA: Complete'), ')',
    ])
    # pbar = ProgressBar(maxval=len(sequences),
    #         widgets=['sequence ', Counter(), '/{} ('.format(len(sequences)),
    #             Percentage(), ') ', Bar(), ' ', ETA()]).start()


def evaluate(sess, inputs, model, sequences, use_gt=False, tre_num=1, **kwargs):
    '''
    Returns:
        A dictionary that contains evaluation results.
    '''
    if not tre_num or tre_num < 1:
        tre_num = 1
    sequences = list(sequences)
    sequence_tre_results = []
    bar = _make_progress_bar()
    for i, full_sequence in enumerate(bar(sequences)):
        tre_results = []
        for tre_ind in range(tre_num):
            sequence = extract_tre_sequence(full_sequence, tre_ind, tre_num)
            if sequence is None:
                # raise ValueError('could not extract TRE sequence')
                continue
            if len(sequence['image_files']) < 2:
                #raise ValueError('sequence shorter than 2 frames')
                continue
            is_valid = sequence['label_is_valid']
            # Count number of valid labels after first frame (ground truth).
            num_valid = sum([1 for x in is_valid[1:] if x])
            if num_valid < 1:
                #raise ValueError('no frames with labels after first frame')
                continue
            #print 'sequence {} of {}'.format(i+1, len(sequences))
            pred, hmap_pred = track(sess, inputs, model, sequence, use_gt, **kwargs)
            # if visualize:
            #     visualize(sequence['video_name'], sequence, pred, hmap_pred, save_frames)
            gt = np.array(sequence['labels'])
            # Convert to original image co-ordinates.
            pred = _unnormalize_rect(pred, sequence['original_image_size'])
            gt   = _unnormalize_rect(gt,   sequence['original_image_size'])
            tre_results.append(evaluate_track(pred, gt, is_valid))
        sequence_tre_results.append(tre_results)

    results = {}
    modes = ['OPE'] + (['TRE'] if tre_num > 1 else [])
    for mode in modes:
        results[mode] = {}
        if mode == 'OPE':
            # Take first result (full sequence).
            sequence_results = [tre_results[0] for tre_results in sequence_tre_results]
        elif mode == 'TRE':
            # Concat all results.
            sequence_results = [result for tre_results in sequence_tre_results for result in tre_results]
        else:
            sequence_results = []
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
            results[mode][k]            = mean.tolist() # Convert to list for JSON.
            results[mode][k+'_std_err'] = std_err.tolist()
    return results


def extract_tre_sequence(seq, ind, num):
    is_valid = seq['label_is_valid']
    num_frames = len(is_valid)
    valid_frames = [t for t in range(num_frames) if is_valid[t]]
    min_t, max_t = valid_frames[0], valid_frames[-1]

    start_t = int(round(float(ind)/num*(max_t-min_t))) + min_t
    # Snap to next valid frame (raises StopIteration if none).
    try:
        start_t = next(t for t in valid_frames if t >= start_t)
    except StopIteration:
        return None

    KEYS_SEQ = ['image_files', 'viewports', 'labels', 'label_is_valid']
    KEYS_NON_SEQ = ['aspect', 'original_image_size']
    subseq = {}
    subseq.update({k: seq[k][start_t:] for k in KEYS_SEQ})
    subseq.update({k: seq[k] for k in KEYS_NON_SEQ})
    subseq['video_name'] = seq['video_name'] + ('-seg{}'.format(ind) if ind > 0 else '')
    return subseq


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
