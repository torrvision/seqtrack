from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import os
import shutil
import subprocess
import tempfile
import time
from fractions import gcd
from PIL import Image

import logging
logger = logging.getLogger(__name__)

from seqtrack import assess
from seqtrack import data
from seqtrack import geom_np
from seqtrack import helpers
# from seqtrack import visualize as visualize_pkg

FRAME_PATTERN = '%06d.jpeg'


# Sequence = collections.namedtuple('Sequence', [
#     'image_files',     # List of strings of length n.
#     'labels',          # Numpy array of shape [n, 4]
#     'label_is_valid',  # List of booleans of length n.
#     'aspect',          # Aspect ratio of original image (width / height).
# ])


def track(sess, tracker, sequence,
          verbose=False,
          # Visualization options:
          visualize=False,
          vis_dir=None,
          keep_frames=False):
    '''Run an instantiated tracker on a sequence.'''

    # assert sequence['label_is_valid'][0]
    tracker.start(sess, {
        'image': {'file': [sequence['image_files'][0]]},
        'rect': [sequence['labels'][0]],
        'aspect': [sequence['aspect']],
    })

    start = time.time()
    # Durations do not include initial frame.
    duration_with_load = 0

    sequence_len = len(sequence['image_files'])
    assert(sequence_len >= 2)
    predictions = []
    for t in range(1, sequence_len):
        start_curr = time.time()
        # TODO: Load image separately.
        curr = tracker.next(sess, {
            'image': {'file': [sequence['image_files'][t]]},
        })
        duration_with_load += time.time() - start_curr
        # Unpack from batch.
        predictions.append(curr['rect'][0])
    duration_real = time.time() - start

    predictions = np.array(predictions)
    timing = {
        'speed_with_load': (sequence_len - 1) / duration_with_load,
        'speed_real': sequence_len / duration_real,
    }

    return predictions, timing


def track_and_assess(sess, model_inst, sequences, tre_num=1, **kwargs):
    '''
    Args:
        kwargs: For track().

    Returns:
        A dictionary that contains evaluation results.
    '''
    subseqs, tre_groups = _split_tre_all(sequences, tre_num)
    predictions = {}
    timing = {}
    bar = helpers.ProgressMeter(interval_time=1)
    for name in bar(subseqs.keys()):
        # TODO: Cache.
        # If we use a subset of sequences, we need to ensure that the subset is the same?
        # Could re-seed video sampler with global step number?
        predictions[name], timing[name] = track(
            sess, model_inst, subseqs[name], **kwargs)
    return assess.assess_dataset(subseqs, predictions, tre_groups, timing=timing)


def _split_tre_all(sequences, tre_num):
    sequences = list(sequences)

    subseqs = {}
    for sequence in sequences:
        for tre_ind in range(tre_num):
            # subseq_name = subseq['video_name']
            subseq_name = _tre_seq_name(sequence['video_name'], tre_ind, tre_num)
            if subseq_name in subseqs:
                raise RuntimeError('name already exists: \'{}\''.format(subseq_name))
            try:
                subseq = _extract_tre_sequence(sequence, tre_ind, tre_num, subseq_name)
            except RuntimeError as ex:
                logger.warning('sequence \'%s\': could not extract TRE sequence %d of %d: %s',
                               sequence['video_name'], tre_ind, tre_num, str(ex))
                # Still insert None if sequence could not be extracted.
                subseq = None
            subseqs[subseq_name] = subseq

    # Build sequence groups for OPE and TRE mode.
    # TODO: Could add all integer factors here?
    tre_group_nums = set([1, tre_num])  # If tre_num is 1, then only OPE.
    tre_groups = {}
    for tre_group_num in tre_group_nums:
        mode = _mode_name(tre_group_num)
        tre_groups[mode] = {}
        for sequence in sequences:
            seq_name = sequence['video_name']
            tre_groups[mode][seq_name] = []
            for tre_ind in range(tre_group_num):
                subseq_name = _tre_seq_name(seq_name, tre_ind, tre_group_num)
                if subseqs[subseq_name] is not None:
                    tre_groups[mode][seq_name].append(subseq_name)
    return subseqs, tre_groups


def _mode_name(tre_num):
    if tre_num == 1:
        return 'OPE'
    elif tre_num > 1:
        return 'TRE_{}'.format(tre_num)
    else:
        raise RuntimeError('tre_num must be at least one: {}'.format(tre_num))


def _tre_seq_name(seq_name, i, n):
    if i == 0:
        return seq_name
    i, n = _simplify_fraction(i, n)
    return '{}_tre_{}_{}'.format(seq_name, i, n)


def _simplify_fraction(i, n):
    if i == 0:
        assert n != 0
        return 0, 1
    g = gcd(i, n)
    return i // g, n // g


def _extract_tre_sequence(seq, ind, num, subseq_name):
    if ind == 0:
        return seq

    is_valid = seq['label_is_valid']
    num_frames = len(is_valid)
    valid_frames = [t for t in range(num_frames) if is_valid[t]]
    min_t, max_t = valid_frames[0], valid_frames[-1]

    start_t = int(round(float(ind) / num * (max_t - min_t))) + min_t
    # Snap to next valid frame (raises StopIteration if none).
    try:
        start_t = next(t for t in valid_frames if t >= start_t)
    except StopIteration:
        raise RuntimeError('no frames with labels')

    subseq = _extract_interval(seq, start_t, None)
    # subseq['video_name'] = seq['video_name'] + ('-seg{}'.format(ind) if ind > 0 else '')
    subseq['video_name'] = subseq_name

    # if len(subseq['image_files']) < 2:
    #     raise RuntimeError('sequence shorter than two frames')
    # Count number of valid labels after first frame (ground truth).
    num_valid = sum(1 for x in subseq['label_is_valid'][1:] if x)
    if num_valid < 1:
        raise RuntimeError('no frames with labels after first frame')
    return subseq


def _extract_interval(seq, start, stop):
    KEYS_SEQ = ['image_files', 'labels', 'label_is_valid']
    KEYS_NON_SEQ = ['aspect']
    subseq = {}
    subseq.update({k: seq[k][start:stop] for k in KEYS_SEQ})
    subseq.update({k: seq[k] for k in KEYS_NON_SEQ})
    return subseq
