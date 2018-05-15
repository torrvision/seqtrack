'''Provides functions for extracting sequences from datasets.

The sequences can be returned as a list or the functions can be generators.

During training, we may use a mixture of multiple datasets.
This makes the concept of an epoch more complex,
since epochs may pass at different rates in different datasets.

For evaluation, we want to run the tracker on all sequences in a dataset,
or possibly a random subset if the dataset is too large.

A sequence sampler has the interface:
    sampler.sample(infinite, rand)
    sampler.dataset
The function sample() returns a track ID that belongs to the dataset.
'''

import pdb
import math
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import geom_np
import trackdat


def sample(sequence_sampler, frame_sampler, postproc_fn=None,
           rand=None, infinite=False, max_num=0):
    '''Produces a finite or infinite stream of sequences from a dataset.

    The sequence sampler describes a distribution of data in a dataset.
    For example, this may be a simple EpochSampler or a MixtureSampler.
    The sequence sampler may support finite and infinite streams.

    The frame sampler chooses a subset of frames from the original sequence.

    Args:
        infinite: If true, then an infinite generator will be returned.
        max_num: If positive, the stream will be limited to this many sequences.

    Caution: If infinite is false, max_num is not positive and the sequence sampler
    does not support finite sets, then the stream may still be infinite.
    '''
    enable_limit = (not infinite and max_num > 0)
    # Note: This will shuffle the sequences even when finite=True and max_num=0.
    # However, this does not seem like a big issue.
    track_ids = sequence_sampler.sample(infinite=infinite, rand=rand)
    frame_sampler.init(sequence_sampler.dataset, rand=rand)
    # Beware: If sequence sampler is a weighted distribution,
    # then it will be an infinite generate even when `infinite` is False.
    num = 0
    for track_id in track_ids:
        if enable_limit and num >= max_num:
            break
        sequence = frame_sampler.sample(track_id)
        # Frame sampler can fail.
        # TODO: Could integrate frame sampler with sequence sampler
        # to e.g. preserve distribution on failure.
        if sequence is None:
            continue
        if postproc_fn is not None:
            sequence = postproc_fn(sequence, rand=rand)
            if sequence is None:
                continue
        yield sequence
        num += 1


class EpochSampler(object):
    '''Standard sampler for a finite dataset.'''

    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self, infinite, rand):
        if infinite:
            while True:
                for track_id in self._sample_epoch(rand=rand):
                    yield track_id
        else:
            for track_id in self._sample_epoch(rand=rand):
                yield track_id

    def _sample_epoch(self, rand):
        track_ids = list(self.dataset.tracks())
        rand.shuffle(track_ids)
        return track_ids


class MixtureSampler(object):
    '''Describes a weighted mixture of datasets.'''

    def __init__(self, samplers, weights):
        '''
        Args:
            samplers: Dictionary that maps dataset name to sampler.
            weights: Dictionary that maps dataset name to weight.
        '''
        self.samplers = samplers
        self.weights = weights
        # Construct concatenation of datasets.
        datasets = {dataset_id: sampler.dataset for dataset_id, sampler in samplers.items()}
        self.dataset = data.Concat(datasets)

    def sample(self, infinite, rand):
        '''
        Args:
            infinite: Ignored. Mixture is always infinite.
        '''
        dataset_ids = sorted(self.samplers.keys())
        weights = np.array([self.weights[dataset_id] for dataset_id in dataset_ids])
        p = weights / np.sum(weights)
        # Start an infinite stream for all constituent samplers.
        streams = {dataset_id: self.samplers[dataset_id].sample(infinite=True, rand=rand)
                   for dataset_id in dataset_ids}
        while True:
            dataset_id = rand.choice(dataset_ids, p=p)
            try:
                track_id = next(streams[dataset_id])
            except StopIteration:
                raise RuntimeError('stream is not infinite: {}'.format(dataset_id))
            # Get track_id in Concat dataset.
            yield self.dataset.join_id(dataset_id, track_id)


class FrameSampler(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def init(self, dataset, rand):
        self.dataset = dataset
        self.rand = rand

    def sample(self, track_id):
        '''
        Args:
            kwargs: For select_frames().
        '''
        return select_frames_dataset(self.dataset, track_id, rand=self.rand, **self.kwargs)


def select_frames_dataset(dataset, track_id, rand, **kwargs):
    '''Selects frames from a full track in a dataset.

    Returns a sequence.
    '''
    labels = dataset.labels(track_id)
    present_frames = sorted(t for t, l in labels.items() if trackdat.dataset.is_present(l))
    if len(present_frames) == 0:
        logger.debug('no present frames: track "%s"', track_id)
        return None
    if len(present_frames) < 2:
        logger.debug('less than two present frames: track "%s"', track_id)
        return None

    t_start, t_stop = present_frames[0], present_frames[-1] + 1
    frames = select_frames_start(t_start, t_stop, present_frames, rand, **kwargs)
    return make_sequence(dataset, track_id, frames)


def select_frames_start(t_start, t_stop, valid_frames, *args, **kwargs):
    t_stop = t_stop - t_start
    valid_frames = [t - t_start for t in valid_frames]
    frames = select_frames(t_stop, valid_frames, *args, **kwargs)
    if frames is None:
        return None
    return [t + t_start for t in frames]


def select_frames(t_stop, valid_frames, rand, kind=None, ntimesteps=None,
                  freq=10, min_freq=10, max_freq=60, use_log=True):
    '''
    Args:
        valid_frames: Sorted list of unique integers in [0, t_stop).

    Returns:
        List of integers in [0, t_stop).
    '''
    if kind == 'full':
        return range(t_stop)
    elif kind == 'sampling':
        # Random subset of present frames.
        # TODO: Change the name when convenient.
        k = min(len(valid_frames), ntimesteps + 1)
        return sorted(rand.choice(valid_frames, k, replace=False))
    elif kind == 'regular':
        # Sample frames with `freq`, regardless of label
        # (only the first frame need to have label).
        # Note also that the returned frames can have length < ntimesteps+1.
        frames = range(rand.choice(valid_frames), t_stop, freq)
        return frames[:ntimesteps + 1]
    elif kind == 'freq-range-fit':
        # Choose frames:
        #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
        # Therefore, for frames [0, ..., ntimesteps], we need:
        #   a + ntimesteps*freq <= t_stop - 1
        # The smallest possible value of a is 0
        #   0 + ntimesteps*freq <= t_stop - 1
        #   ntimesteps*freq <= t_stop - 1
        #   freq <= (t_stop - 1) / ntimesteps
        u = min_freq
        v = min(max_freq, float((t_stop - 1)) / ntimesteps)
        if not u <= v:
            return None
        if use_log:
            f = math.exp(rand.uniform(math.log(u), math.log(v)))
        else:
            f = rand.uniform(u, v)
        # Let n = ntimesteps*f.
        n = int(round(ntimesteps * f))
        # Choose first frame such that all frames are present.
        a = rand.choice([a for a in valid_frames if a + n <= t_stop])
        return [int(round(a + f * t)) for t in range(0, ntimesteps + 1)]
    else:
        raise ValueError('unknown sampler: {}'.format(kind))


def make_sequence(dataset, track_id, frames):
    labels = dataset.labels(track_id)
    label_is_valid = [t in labels and trackdat.dataset.is_present(labels[t]) for t in frames]
    # Skip sequences with no labels (after first label).
    if not any(label_is_valid[1:]):
        logger.debug('sub-sequence contains no labels: track "%s"', track_id)
        return None
    video_id = dataset.video(track_id)
    return {
        'image_files': [dataset.image_file(video_id, t) for t in frames],
        'viewports': [geom_np.unit_rect() for _ in frames],
        'labels': [_rect_from_label(labels.get(t, None)) for t in frames],
        'label_is_valid': label_is_valid,
        'aspect': dataset.aspect(video_id),
        'video_name': track_id,
    }


def _rect_from_label(label):
    if label is None:
        return _invalid_rect()
    if not trackdat.dataset.is_present(label):
        return _invalid_rect()
    if 'rect' not in label:
        raise ValueError('label does not contain rect')
    rect = label['rect']
    min_pt = [rect['xmin'], rect['ymin']]
    max_pt = [rect['xmax'], rect['ymax']]
    return geom_np.make_rect(min_pt, max_pt)


def _invalid_rect():
    return [float('nan')] * 4
