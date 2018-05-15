import numpy as np
import operator
from itertools import chain

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom_np


def assess_dataset(seqs, predictions, tre_num=1, tre_subseqs=None):
    '''Assesses predictions for an entire dataset.

    Args:
        seqs: Dict that maps subseq name to sequence.
        predictions: Dict that maps subseq name to predictions.
        tre_subseqs: Dict that maps seq name to list of subseq names.

    Returns:
        Dict of scalar metrics.
    '''
    if tre_subseqs is None:
        # Use each sequence by itself.
        tre_subseqs = {name: [name] for name in seqs.keys()}

    # Compute per-frame metrics.
    frame_metrics = {
        name: assess_frames(seqs[name], predictions[name]) for name in seqs.keys()}
    # Compute per-sequence metrics using per-frame metrics.
    sequence_metrics = {
        name: assess_sequence(seqs[name], predictions[name], frame_metrics[name])
        for name in seqs.keys()}

    # TODO: Could compute for all factors of tre_num?
    # That is, if tre_num = 10, report 1, 2, 5, 10.
    modes = ['OPE']
    if tre_num > 1:
        # TODO: Ugly to pass tre_num just for this purpose?
        modes.append('TRE_{}'.format(tre_num))

    metrics = {}
    for mode in modes:
        if mode == 'OPE':
            # Only use original sequence (should have same name).
            mode_subseqs = {seq: [seq] for seq in tre_subseqs.keys()}
        else:
            # Use all TRE splits.
            mode_subseqs = tre_subseqs
        mode_metrics = _summarize(frame_metrics, sequence_metrics, mode_subseqs)
        # Add metrics for OPE and TRE mode.
        for key in mode_metrics:
            metrics[mode + '_' + key] = mode_metrics[key]
    return metrics


FRAME_METRICS = ['iou', 'center_dist', 'oracle_size_iou']


def assess_frames(sequence, pred_rects):
    '''Computes per-frame metrics for entire sequence.

    Args:
        sequence: Dict with keys:
            labels: Array of rects. Shape [num_frames, 4].
            label_is_valid: Array of bools. Shape [num_frames].
        pred_rects: Array of rects. Shape [num_frames - 1, 4].
    '''
    label_present = np.array(sequence['label_is_valid'][1:])
    label_rects = np.array(sequence['labels'][1:])

    iou = geom_np.rect_iou(label_rects, pred_rects)
    # Compute distance between centers of rectangles.
    label_center, label_size = geom_np.rect_center_size(label_rects)
    pred_center, _ = geom_np.rect_center_size(pred_rects)
    center_dist = np.linalg.norm(pred_center - label_center, axis=-1)
    # Compute IOU using ground-truth size.
    oracle_size_rect = geom_np.make_rect_center_size(pred_center, label_size)
    oracle_size_iou = geom_np.rect_iou(label_rects, oracle_size_rect)

    # Set metrics to nan when object is not present.
    return {
        'iou': np.where(label_present, iou, float('nan')),
        'center_dist': np.where(label_present, center_dist, float('nan')),
        'oracle_size_iou': np.where(label_present, oracle_size_iou, float('nan')),
    }


IOU_THRESHOLDS = [0.5]
AUC_NUM_STEPS = 1000
SEQUENCE_METRICS = (
    FRAME_METRICS +
    ['iou_success_{}'.format(thr) for thr in IOU_THRESHOLDS] +
    ['iou_success_auc'])


def assess_sequence(sequence, predictions, frame_metrics):
    '''Computes per-sequence metrics. Uses per-frame metrics.'''
    metrics = {}
    # Take mean of all per-frame metrics.
    for key in FRAME_METRICS:
        metrics[key] = np.nanmean(frame_metrics[key])
    # Add new metrics.
    for thr in IOU_THRESHOLDS:
        metrics['iou_success_{}'.format(thr)] = \
            np.nanmean(_compare_nan(operator.ge, frame_metrics['iou'], thr))

    metrics['iou_success_auc'] = _compute_auc(frame_metrics['iou'], AUC_NUM_STEPS)
    return metrics


def _summarize(frame_metrics, sequence_metrics, tre_subseqs):
    '''
    Args:
        tre_subseqs: Dict that maps seq name to list of subseq names.
            This can be used to switch between OPE and TRE modes.
    '''
    metrics = {}
    all_subseqs = list(chain(*tre_subseqs.values()))
    # Compute the per-frame averages.
    for key in FRAME_METRICS:
        # Concatenate frames from all subsequences and take mean.
        # TODO: Ensure that tracker reporting nan means zero overlap!
        metrics[key + '_frame_mean'] = np.nanmean(np.concatenate(
            [frame_metrics[subseq][key] for subseq in all_subseqs]))
    # Compute the per-sequence averages and bootstrap variances.
    for key in SEQUENCE_METRICS:
        subseq_means = {subseq_name: np.nanmean(sequence_metrics[subseq_name][key])
                        for subseq_name in all_subseqs}
        # Take mean over all subsequences within a sequence (in OPE mode, just one).
        seq_means = [
            np.mean([subseq_means[subseq] for subseq in tre_subseqs[seq]])
            for seq in tre_subseqs.keys()]
        metrics[key + '_seq_mean'] = np.mean(seq_means)
        # Report raw variance for debug.
        metrics[key + '_seq_var'] = np.var(seq_means)
        # Use the bootstrap estimate for the variance of the mean of a set.
        # Note that this is different from the variance of a single set.
        # See page 107 of "All of Statistics" (Wasserman).
        metrics[key + '_seq_mean_var'] = np.var(seq_means) / len(tre_subseqs)
    # Add new metrics.
    all_iou = np.concatenate([frame_metrics[subseq]['iou'] for subseq in all_subseqs])
    for thr in IOU_THRESHOLDS:
        metrics['iou_success_{}'.format(thr)] = np.nanmean(_compare_nan(operator.ge, all_iou, thr))
    metrics['iou_success_auc'] = _compute_auc(all_iou, AUC_NUM_STEPS)
    return metrics


def _compute_auc(iou, num):
    thresholds, step = np.linspace(0, 1, num + 1, retstep=True)
    # success = np.nanmean(iou[:, np.newaxis] >= thresholds, axis=0)
    success = np.nanmean(_compare_nan(operator.ge, iou[:, np.newaxis], thresholds), axis=0)
    return np.trapz(success, dx=step)


def _compare_nan(op, x, y):
    '''Computes op(x, y).astype(float) and restores nan values.

    The resulting array contains {0, 1, nan}.
    '''
    isnan = np.logical_or(np.isnan(x), np.isnan(y))
    with np.errstate(invalid='ignore'):
        val = op(x, y)
    return np.where(isnan, float('nan'), val.astype(float))
