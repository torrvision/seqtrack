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

    first_image = load_image(sequence['image_files'][0])
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
        images = map(load_image, sequence['image_files'][start:start+chunk_len])
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


# def evaluate(sess, inputs, outputs, state, loader, o, dstype, nbatches_=None, hold_inputs=False,
#         shuffle_local=False, fulllen=False):
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

    # if not fulllen: # this is only used during training.
    #     if o.debugmode:
    #         nbatches = 10
    #     else:
    #         if nbatches_ is None: # test all batches. 
    #             assert(False) # unlikely happen 
    #             nbatches = int(loader.nexps[dstype]/o.batchsz) # NOTE: currently not supporting remainders
    #         else:
    #             nbatches = nbatches_
    #     results['nbatches'] = nbatches

    #     for ib in range(nbatches):
    #         t_start = time.time()
    #         batch = loader.get_batch(ib, o, dstype, shuffle_local=shuffle_local)

    #         names = ['target_raw', 'inputs_raw', 'x0_raw', 'y0', 'inputs_valid', 'inputs_HW', 'labels']
    #         fdict = {m.net[name]: batch[name] for name in names}

    #         outputs, loss = sess.run([m.net['outputs'], m.net['loss']], feed_dict=fdict)

    #         keys = ['idx', 'inputs_valid', 'inputs_HW', 'labels']
    #         if hold_inputs:
    #             keys.append('inputs_raw')
    #         for k in keys:
    #             if k in batch:
    #                 results.setdefault(k, []).append(batch[k])
    #         results.setdefault('outputs', []).append(outputs)
    #         results.setdefault('loss', []).append(loss)

    #         sys.stdout.write(
    #                 '\r(during \'{0:s}\') passed {1:d}/{2:d}th batch on '\
    #                     '[{3:s}] set.. |time: {4:.3f}'.format(
    #                     o.mode, ib+1, nbatches, dstype, time.time()-t_start))
    #         sys.stdout.flush()
    #     print ' '

    #     # list -> numpy array
    #     for key in results.keys():
    #         if isinstance(results[key], list):
    #             results[key] = np.asarray(results[key])

    #     # if outtype is heatmap, 
    #     # 1) convert 2-dim heatmaps to 1 dim (softmax); for valid probability
    #     # 2) compute rectangle from heatmap
    #     outtype = 'heatmap'
    #     if outtype == 'heatmap':
    #         results['outputs'] = softmax_v1(results['outputs'])
    #         results['outputs_rec'] = convert_heatmap_to_rec(
    #                 results['outputs'][:,:,:,:,:,0], results['inputs_HW'])

    #     # evaluate
    #     results = evaluate_outputs(results, outtype=outtype)
    #     return results

    # 
    # # Main evaluation routine for full-length sequences.
    # else:
    #     nexps = loader.nexps_fulllen if not o.debugmode else 4
    #     for i in range(nexps): # one full-length example at a time
    #         batch_fl = loader.get_batch_fl(i, o)
    #         batch = data.split_batch_fulllen_seq(batch_fl, o)

    #         outputs_all = []
    #         for j in range(batch['nsegments']): # segments
    #             t_start = time.time()

    #             batch_seg = {}
    #             for key in ['inputs', 'inputs_valid', 'inputs_HW', 'labels']:
    #                 batch_seg[key] = batch[key][j][np.newaxis]

    #             # zero padding to meet batchsz constraints in model; TODO: need to be relaxed.
    #             padlen = o.batchsz - 1
    #             pad = {}
    #             pad['inputs'] = np.zeros(
    #                 [padlen, o.ntimesteps+1, o.frmsz, o.frmsz, o.ninchannel], 
    #                 dtype=np.float32)
    #             pad['inputs_valid'] = np.zeros(
    #                 [padlen, o.ntimesteps+1], dtype=np.bool)
    #             pad['inputs_HW'] = np.zeros(
    #                 [padlen, 2], dtype=np.bool)
    #             pad['labels'] = np.zeros(
    #                 [padlen, o.ntimesteps+1, o.outdim], 
    #                 dtype=np.bool)
    #             batch_seg['inputs'] = np.concatenate(
    #                 (batch_seg['inputs'], pad['inputs']), axis=0)
    #             batch_seg['inputs_valid'] = np.concatenate(
    #                 (batch_seg['inputs_valid'], pad['inputs_valid']),axis=0)
    #             batch_seg['inputs_HW'] = np.concatenate(
    #                 (batch_seg['inputs_HW'], pad['inputs_HW']), axis=0)
    #             batch_seg['labels'] = np.concatenate(
    #                 (batch_seg['labels'], pad['labels']), axis=0)

    #             if j==0:
    #                 x0 = batch_seg['inputs'][:,0,:,:,:]
    #                 y0 = batch_seg['labels'][:,0,:]

    #             if j == 0: # first segment
    #                 fdict = {}
    #                 fdict = {
    #                     m.net['inputs']: batch_seg['inputs'],
    #                     m.net['inputs_valid']: batch_seg['inputs_valid'],
    #                     m.net['inputs_HW']: batch_seg['inputs_HW'],
    #                     m.net['labels']: batch_seg['labels']
    #                     }
    #             else: # from second segment need to pass previous outputs
    #                 fdict = {}
    #                 fdict = {
    #                     m.net['inputs']: batch_seg['inputs'],
    #                     m.net['inputs_valid']: batch_seg['inputs_valid'],
    #                     m.net['inputs_HW']: batch_seg['inputs_HW'],
    #                     m.net['labels']: batch_seg['labels'],
    #                     m.net['h_init']: h_last,
    #                     m.net['C_init']: C_last,
    #                     m.net['y_init']: y_last,
    #                     m.net['x0']: x0,
    #                     m.net['y0']: y0
    #                     }

    #             outputs, h_last, C_last, y_last = sess.run(
    #                 [m.net['outputs'], m.net['h_last'], m.net['C_last'], m.net['y_last']],
    #                 feed_dict=fdict)
    #             outputs_all.append(outputs[0, :, :])

    #             sys.stdout.write(
    #                 '\r(during \'{0:s}\') passed {1:d}/{2:d} exp, {3:d}/{4:d} '\
    #                 'segment on [{5:s}] set.. |time: {6:.3f}'.format(o.mode, 
    #                     i+1, nexps, j+1, batch['nsegments'], dstype, 
    #                     time.time()-t_start))
    #             sys.stdout.flush()

    #         # valid outputs is -1 shorter 
    #         outputs_all = np.reshape(np.asarray(outputs_all), [-1, o.outdim])
    #         outputs_all = outputs_all[:batch_fl['nfrms']-1][np.newaxis]

    #         # TODO: check if it's possible to hold data for ILSVRC as well
    #         if 'idx' in batch_seg: 
    #             results['idx'].append(batch_fl['idx'])
    #         results['inputs'].append(batch_fl['inputs'])
    #         results['inputs_valid'].append(batch_fl['inputs_valid'])
    #         results['inputs_HW'].append(batch_fl['inputs_HW'])
    #         results['labels'].append(batch_fl['labels'])
    #         results['outputs'].append(outputs_all)
    #         results['idx'].append(batch_fl['idx'])
    #     print ' '

    #     results = evaluate_outputs_new(results)

    #     return results


# def get_batch_split_fortest(batch, t):
#     '''
#     Create a batch split by fetching at specific time t and padding with zeros.
#     '''
#     inputs = np.zeros_like(batch['inputs'])
#     inputs_valid = np.zeros_like(batch['inputs_valid'], dtype=np.bool)
#     labels = np.zeros_like(batch['labels'])
# 
#     inputs[:, :t+1]     = batch['inputs'][:, :t+1]
#     inputs_valid[:, t]  = batch['inputs_valid'][:, t] # only valid time step
#     labels[:, :t+1]     = batch['labels'][:, :t+1]
# 
#     splits = {}
#     splits['inputs'] = inputs
#     splits['inputs_valid'] = inputs_valid
#     splits['inputs_HW'] = batch['inputs_HW'] # no time dependent
#     splits['labels'] = labels
#     if 'idx' in batch: 
#         splits['idx'] = batch['idx'] # no time dependent
#     return splits


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
    centerA_x = (boxA[:,0] + boxA[:,2])/2
    centerA_y = (boxA[:,1] + boxA[:,3])/2
    centerB_x = (boxB[:,0] + boxB[:,2])/2
    centerB_y = (boxB[:,1] + boxB[:,3])/2
    cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)
    return cle


# def evaluate_outputs_new(results):
# 
#     ious = []
#     cles = []
#     inputs_valids = []
#     for i in range(len(results['outputs'])):
#         boxA = np.copy(results['outputs'][i])
#         boxB = np.copy(results['labels'][i][:,1:])
#         inputs_valid = np.copy(results['inputs_valid'][i][:,1:])
#         inputs_HW = np.copy(results['inputs_HW'][i])
# 
#         # back to original scale (pixel)
#         scalar = np.concatenate((inputs_HW[:,[1]], inputs_HW[:,[0]], 
#             inputs_HW[:,[1]], inputs_HW[:,[0]]), axis=1)
#         boxA *= np.expand_dims(scalar, 1)
#         boxB *= np.expand_dims(scalar, 1)
# 
#         # determine the (x, y)-coordinates of the intersection rectangle
#         xA = np.maximum(boxA[:,:,0], boxB[:,:,0])
#         yA = np.maximum(boxA[:,:,1], boxB[:,:,1])
#         xB = np.minimum(boxA[:,:,2], boxB[:,:,2])
#         yB = np.minimum(boxA[:,:,3], boxB[:,:,3])
# 
#         # compute the area of intersection rectangle
#         interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
# 
#         # compute the area of both the prediction and ground-truth rectangles
#         boxAArea = (boxA[:,:,2] - boxA[:,:,0])\
#                 * (boxA[:,:,3] - boxA[:,:,1])
#         boxBArea = (boxB[:,:,2] - boxB[:,:,0])\
#                 * (boxB[:,:,3] - boxB[:,:,1])
# 
#         # compute the intersection over union by taking the intersection area and 
#         # dividing it by the sum of prediction + ground-truth areas - the 
#         # interesection area
#         iou = interArea / (boxAArea + boxBArea - interArea)
#         iou = iou.astype(np.float32)
# 
#         # for precision computation (center location error)
#         centerA_x = (boxA[:,:,0] + boxA[:,:,2])/2
#         centerA_y = (boxA[:,:,1] + boxA[:,:,3])/2
#         centerB_x = (boxB[:,:,0] + boxB[:,:,2])/2
#         centerB_y = (boxB[:,:,1] + boxB[:,:,3])/2
#         cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)
# 
#         # to compare with the previous function, let's concatenate all iou and cle
#         ious.extend(iou[0])
#         cles.extend(cle[0])
#         inputs_valids.extend(inputs_valid[0])
# 
#     ious = np.asarray(ious)
#     cles = np.asarray(cles)
#     inputs_valids = np.asarray(inputs_valids)
# 
#     # Success plot 
#     # 1. mean iou over only valid length
#     # 2. success counter over only valid length (for success plot and auc)
#     # 3. area under curve
#     iou_valid = ious[inputs_valids]
#     success_rate_thresholds = np.append(np.arange(0,1,0.05), 1)
#     success_rate_thresholds = np.tile(
#             success_rate_thresholds, (iou_valid.size,1))
#     success_rate_table = iou_valid[:,np.newaxis] > success_rate_thresholds
# 
#     iou_mean = np.mean(iou_valid)
#     success_rates = np.mean(success_rate_table, axis=0)
#     auc = np.mean(success_rates)
# 
#     # Precision plot; 
#     # 1. center location error 
#     # 2. precision plot
#     # 3. representative precision error
#     cle_valid = cles[inputs_valids]
#     precision_rate_thresholds = np.arange(0, 60, 5)
#     precision_rate_thresholds = np.tile(
#             precision_rate_thresholds, (cle_valid.size,1))
#     precision_rate_table = cle_valid[:,np.newaxis] < precision_rate_thresholds
#     representative_precision_threshold = 20 # benchmark
# 
#     cle_mean = np.mean(cle_valid)
#     precision_rates = np.mean(precision_rate_table, axis=0)
#     cle_representative = np.mean(
#             cle_valid[cle_valid < representative_precision_threshold])
# 
#     results['iou_mean'] = iou_mean
#     results['success_rates'] = success_rates
#     results['auc'] = auc
#     results['cle_mean'] = cle_mean
#     results['precision_rates'] = precision_rates
#     results['cle_representative'] = cle_representative
#     return results
# 
# def evaluate_outputs(results, outtype='rectangle'):
#     inputs_valid    = results['inputs_valid']
#     inputs_HW       = results['inputs_HW']
#     boxB            = np.copy(results['labels'])
#     if outtype == 'heatmap':
#         boxA = np.copy(results['outputs_rec'])
#     elif outtype == 'rectangle':
#         boxA = np.copy(results['outputs'])
#     else: 
#         raise ValueError('Not available outtype')
# 
#     # take only valid to compute (i.e., from frame 1)
#     boxB = boxB[:,:,1:,:]
#     inputs_valid = inputs_valid[:,:,1:]
#         
#     # back to original scale (pixel)
#     scalar = np.concatenate((inputs_HW[:,:,[1]], inputs_HW[:,:,[0]], 
#         inputs_HW[:,:,[1]], inputs_HW[:,:,[0]]), axis=2)
#     boxA *= np.expand_dims(scalar, 2)
#     boxB *= np.expand_dims(scalar, 2)
# 
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = np.maximum(boxA[:,:,:,0], boxB[:,:,:,0])
#     yA = np.maximum(boxA[:,:,:,1], boxB[:,:,:,1])
#     xB = np.minimum(boxA[:,:,:,2], boxB[:,:,:,2])
#     yB = np.minimum(boxA[:,:,:,3], boxB[:,:,:,3])
# 
#     # compute the area of intersection rectangle
#     interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
# 
#     # compute the area of both the prediction and ground-truth rectangles
#     boxAArea = (boxA[:,:,:,2] - boxA[:,:,:,0])\
#             * (boxA[:,:,:,3] - boxA[:,:,:,1])
#     boxBArea = (boxB[:,:,:,2] - boxB[:,:,:,0])\
#             * (boxB[:,:,:,3] - boxB[:,:,:,1])
# 
#     # compute the intersection over union by taking the intersection area and 
#     # dividing it by the sum of prediction + ground-truth areas - the 
#     # interesection area
#     iou = interArea / (boxAArea + boxBArea - interArea)
#     iou = iou.astype(np.float32)
# 
#     # for precision computation (center location error)
#     centerA_x = (boxA[:,:,:,0] + boxA[:,:,:,2])/2
#     centerA_y = (boxA[:,:,:,1] + boxA[:,:,:,3])/2
#     centerB_x = (boxB[:,:,:,0] + boxB[:,:,:,2])/2
#     centerB_y = (boxB[:,:,:,1] + boxB[:,:,:,3])/2
#     cle = np.sqrt((centerA_x-centerB_x)**2 + (centerA_y-centerB_y)**2)
# 
#     # Success plot 
#     # 1. mean iou over only valid length
#     # 2. success counter over only valid length (for success plot and auc)
#     # 3. area under curve
#     iou_valid = iou[inputs_valid]
#     success_rate_thresholds = np.append(np.arange(0,1,0.05), 1)
#     success_rate_thresholds = np.tile(
#             success_rate_thresholds, (iou_valid.size,1))
#     success_rate_table = iou_valid[:,np.newaxis] > success_rate_thresholds
# 
#     iou_mean = np.mean(iou_valid)
#     success_rates = np.mean(success_rate_table, axis=0)
#     auc = np.mean(success_rates)
# 
#     # Precision plot; 
#     # 1. center location error 
#     # 2. precision plot
#     # 3. representative precision error
#     cle_valid = cle[inputs_valid]
#     precision_rate_thresholds = np.arange(0, 60, 5)
#     precision_rate_thresholds = np.tile(
#             precision_rate_thresholds, (cle_valid.size,1))
#     precision_rate_table = cle_valid[:,np.newaxis] < precision_rate_thresholds
#     representative_precision_threshold = 20 # benchmark
# 
#     cle_mean = np.mean(cle_valid)
#     precision_rates = np.mean(precision_rate_table, axis=0)
#     cle_representative = np.mean(
#             cle_valid[cle_valid < representative_precision_threshold])
# 
#     results['iou_mean'] = iou_mean
#     results['success_rates'] = success_rates
#     results['auc'] = auc
#     results['cle_mean'] = cle_mean
#     results['precision_rates'] = precision_rates
#     results['cle_representative'] = cle_representative
#     return results
# 
# def softmax_v1(x):
#     e_x = np.exp(x - np.expand_dims(np.max(x,axis=5), axis=-1)) # for numerical stability
#     return e_x / np.expand_dims(np.sum(e_x, axis=5), axis=-1)
# 
# def convert_heatmap_to_rec(heatmap, inputs_HW):
#     nbatches = heatmap.shape[0]
#     batchsz = heatmap.shape[1]
#     ntimesteps = heatmap.shape[2]
# 
#     # pass threshold
#     threshold = 0.7
#     heatmap_binary = heatmap>threshold
# 
#     # heatmap -> rec
#     rec = np.zeros([nbatches, batchsz, ntimesteps, 4], dtype=np.float32)
#     for ib in range(nbatches):
#         for ie in range(batchsz):
#             for t in range(ntimesteps):
#                 indices = np.transpose(np.nonzero(heatmap_binary[ib,ie,t]))
#                 if len(indices) != 0:
#                     rec[ib,ie,t,0] = np.min(indices[:,1])
#                     rec[ib,ie,t,1] = np.min(indices[:,0])
#                     rec[ib,ie,t,2] = np.max(indices[:,1])
#                     rec[ib,ie,t,3] = np.max(indices[:,0])
#                 else: # if there is no foreground fired, output the entire image
#                     rec[ib,ie,t,0] = 0
#                     rec[ib,ie,t,1] = 0
#                     rec[ib,ie,t,2] = inputs_HW[ib, ie, 0]
#                     rec[ib,ie,t,3] = inputs_HW[ib, ie, 1]
# 
#     # rectangle is scaled betwen 0 and 1
#     scalar = np.concatenate((inputs_HW[:,:,[1]], inputs_HW[:,:,[0]], 
#         inputs_HW[:,:,[1]], inputs_HW[:,:,[0]]), axis=2)
#     rec /= np.expand_dims(scalar, 2)
#     
#     return rec
