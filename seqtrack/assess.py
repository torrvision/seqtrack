from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import operator

import logging
logger = logging.getLogger(__name__)

from seqtrack import geom_np


def assess_dataset(seqs, predictions, tre_groups=None, timing=None):
    '''Assesses predictions for an entire dataset.

    Args:
        seqs: Dict that maps subseq name to sequence.
        predictions: Dict that maps subseq name to predictions.
        tre_groups: Dict that maps protocol name (e.g. OPE, TRE_3) to:
            dict that maps seq name to list of subseq names.

    Returns:
        Dict of scalar metrics.
    '''
    if tre_groups is None:
        # Use each sequence by itself and call it OPE mode.
        tre_groups = {'OPE': {name: [name] for name in seqs.keys()}}

    # Compute per-frame metrics.
    frame_metrics = {
        name: assess_frames(seqs[name], predictions[name]) for name in seqs.keys()}
    # Compute per-sequence metrics using per-frame metrics.
    sequence_metrics = {
        name: assess_sequence(seqs[name], predictions[name], frame_metrics[name],
                              timing=(None if timing is None else timing[name]))
        for name in seqs.keys()}

    metrics = {}
    for mode in tre_groups:
        mode_metrics = _summarize(frame_metrics, sequence_metrics, tre_groups[mode])
        # Add metrics for all modes.
        for key in mode_metrics:
            metrics[mode + '_' + key] = mode_metrics[key]
    return metrics


FRAME_METRICS = ['iou', 'center_dist', 'oracle_size_iou', 'iou_until_zero']


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
    iou = np.where(label_present, iou, float('nan'))
    center_dist = np.where(label_present, center_dist, float('nan'))
    oracle_size_iou = np.where(label_present, oracle_size_iou, float('nan'))

    # Set IOU to zero after it first reaches zero.
    iou_until_zero = np.where(np.isnan(iou), np.nan,
                              np.where(_is_before_first(iou == 0), iou, 0))

    return {
        'iou': iou,
        'center_dist': center_dist,
        'oracle_size_iou': oracle_size_iou,
        'iou_until_zero': iou_until_zero,
    }


IOU_THRESHOLDS = [0.5, 0.7]
AUC_NUM_STEPS = 1000
SEQUENCE_METRICS = list(itertools.chain(
    FRAME_METRICS,
    ['speed_eval', 'speed_with_load', 'speed_real'],
    ['iou_success_{}'.format(thr) for thr in IOU_THRESHOLDS],
    ['iou_success_auc'],
    ['iou_success_until_failure_{}'.format(thr) for thr in IOU_THRESHOLDS]))


def assess_sequence(sequence, predictions, frame_metrics, timing=None):
    '''Computes per-sequence metrics. Uses per-frame metrics.'''
    metrics = {}
    # Take mean of all per-frame metrics.
    for key in FRAME_METRICS:
        metrics[key] = np.nanmean(frame_metrics[key])
    # metrics['iou_until_zero'] = np.nanmean(
    #     np.where(np.isnan(frame_metrics['iou']), np.nan,
    #              np.where(_is_before_first(frame_metrics['iou'] == 0), frame_metrics['iou'], 0)))
    # Add new metrics.
    for thr in IOU_THRESHOLDS:
        correct = _greater_equal_keep_nan(frame_metrics['iou'], thr)
        metrics['iou_success_{}'.format(thr)] = np.nanmean(correct)
        metrics['iou_success_until_failure_{}'.format(thr)] = np.nanmean(
            np.where(np.isnan(correct), np.nan,
                     np.where(_is_before_first(correct == 0), correct, 0)))

    if timing is not None:
        metrics['speed_eval'] = timing['num_frames'] / timing['duration_eval']
        metrics['speed_with_load'] = timing['num_frames'] / timing['duration_with_load']
        # Maybe this should be num_frames + 1 but at least it is an under-estimate.
        metrics['speed_real'] = timing['num_frames'] / timing['duration_real']

    metrics['iou_success_auc'] = _compute_auc(frame_metrics['iou'], AUC_NUM_STEPS)
    return metrics


def _summarize(frame_metrics, sequence_metrics, tre_groups):
    '''
    Args:
        tre_groups: Dict that maps seq name to list of subseq names.
            This can be used to switch between OPE and TRE modes.
    '''
    metrics = {}
    all_subseqs = list(itertools.chain(*tre_groups.values()))
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
            np.mean([subseq_means[subseq] for subseq in tre_groups[seq]])
            for seq in tre_groups.keys()]
        metrics[key + '_seq_mean'] = np.mean(seq_means)
        # Report raw variance for debug.
        metrics[key + '_seq_var'] = np.var(seq_means)
        # Use the bootstrap estimate for the variance of the mean of a set.
        # Note that this is different from the variance of a single set.
        # See page 107 of "All of Statistics" (Wasserman).
        metrics[key + '_seq_mean_var'] = np.var(seq_means) / len(tre_groups)
    # Add new metrics.
    all_iou = np.concatenate([frame_metrics[subseq]['iou'] for subseq in all_subseqs])
    for thr in IOU_THRESHOLDS:
        metrics['iou_success_{}'.format(thr)] = np.nanmean(_greater_equal_keep_nan(all_iou, thr))
    metrics['iou_success_auc'] = _compute_auc(all_iou, AUC_NUM_STEPS)
    return metrics


def _compute_auc(iou, num):
    thresholds, step = np.linspace(0, 1, num + 1, retstep=True)
    # success = np.nanmean(iou[:, np.newaxis] >= thresholds, axis=0)
    success = np.nanmean(_greater_equal_keep_nan(iou[:, np.newaxis], thresholds), axis=0)
    return np.trapz(success, dx=step)


def _greater_equal_keep_nan(x, y):
    '''
    Args:
        x, y: Float.
    '''
    return _keep_nan(np.greater_equal, x, y)


def _keep_nan(op, x, y):
    '''Computes op(x, y).astype(float) and restores nan values.

    This is for evaluating boolean operations with nans.
    The resulting array contains {0, 1, nan}.
    '''
    isnan = np.logical_or(np.isnan(x), np.isnan(y))
    with np.errstate(invalid='ignore'):
        val = op(x, y)
    return np.where(isnan, float('nan'), val.astype(float))


def _is_before_first(xs):
    '''Returns a list of booleans that indicates whether each element is before the first True.

    Args:
        xs: List of bools.
    '''
    first = next((i for i, x in enumerate(xs) if x), len(xs))
    return [i < first for i in range(len(xs))]
