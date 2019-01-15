'''Obtain streams of examples from datasets.

There are two stages:
1. SequenceSampler chooses a track.
2. ExampleSampler extracts an example from the track.

Common examples for SequenceSampler are:
* single (shuffled) epoch
* uniform distribution over tracks
* combine several datasets; choose dataset, then choose track

The ExampleSampler may pre-compute some information for each sequence
to enable efficient sampling.
Therefore, an ExampleSampler constructor takes the `dataset` as a parameter.
The interface is:

SequenceSampler has the interface:
    sequence_sampler.dataset()
    sequence_sampler.sample(infinite, rand)
The function sample() returns a track ID that belongs to the dataset.

Usage:
    sequence_sampler = SequenceSampler(...)
    dataset = sequence_sampler.dataset()
    example_sampler = ExampleSampler(dataset, ...)

    for track_id in sequence_sampler.sample():
        example = example_sampler.sample(track_id)

During training, we may use a mixture of multiple datasets.
This makes the concept of an epoch more complex,
since epochs may pass at different rates in different datasets.

For evaluation, we want to run the tracker on all sequences in a dataset,
or possibly a random subset if the dataset is too large.

ExampleSampler has the interface:
The function sample() returns a dictionary.
    example_sampler.sample(track_id)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import geom_np
from seqtrack.models import itermodel
import trackdat
from trackdat.dataset import is_present as _is_present


class EpochSampler(object):
    '''Standard sampler for a finite dataset.'''

    def __init__(self, rand, dataset, repeat=False):
        self._dataset = dataset
        self._repeat = repeat
        self._rand = rand

    def dataset(self):
        return self._dataset

    def sample(self):
        if self._repeat:
            while True:
                for track_id in self._sample_epoch():
                    yield track_id
        else:
            for track_id in self._sample_epoch():
                yield track_id

    def _sample_epoch(self):
        track_ids = list(self._dataset.tracks())
        self._rand.shuffle(track_ids)
        return track_ids


class MixtureSampler(object):
    '''Weighted mixture of other samplers.'''

    def __init__(self, samplers, p=None):
        '''
        Args:
            samplers: Dictionary that maps dataset name to sampler.
            p: Dictionary that maps dataset name to weight.
        '''
        self._samplers = samplers
        self._p = None if not p else (np.asfarray(p) / np.sum(p))
        # Construct concatenation of datasets.
        datasets = {sampler_id: sampler.dataset() for sampler_id, sampler in samplers.items()}
        self._dataset = data.Concat(datasets)

    def dataset(self):
        return self._dataset

    def sample(self):
        sampler_ids = sorted(self._samplers.keys())
        # Start a stream for all constituent samplers.
        streams = {sampler_id: self._samplers[sampler_id].sample() for sampler_id in sampler_ids}
        while True:
            sampler_id = self._rand.choice(sampler_ids, p=self._p)
            # If the components are not infinite, this may raise StopIteration.
            track_id = next(streams[sampler_id])
            # Get track_id in Concat dataset.
            yield self._dataset.join_id(sampler_id, track_id)


class Sequence(object):

    def __init__(self,
                 image_files,
                 valid_set,
                 rects,
                 aspect,
                 name):
        self.image_files = image_files
        self.valid_set = valid_set
        self.rects = rects
        self.aspect = aspect
        self.name = name

    def __len__(self):
        return len(image_files)


def extract_sequence_from_dataset(dataset, track_id, times=None):
    labels = dataset.labels(track_id)
    if not times:
        # Default: Use all frames between first and last valid frame.
        valid_times = [t for t, label in labels.items() if _is_present(label)]
        if not valid_times:
            raise RuntimeError('sequence contains no labels')
        t_start = min(valid_times)
        t_stop = max(valid_times) + 1
        times = list(range(t_start, t_stop))

    video_id = dataset.video(track_id)
    return Sequence(
        image_files=[dataset.image_file(video_id, t) for t in times],
        valid_set=set(i for i, t in enumerate(times) if t in labels and _is_present(labels[t]))
        rects=np.array([_rect_from_label(labels.get(t, None)) for t in times]),
        aspect=dataset.aspect(video_id),
        name=track_id,
    )


def _rect_from_label(label):
    if label is None:
        return np.full(4, np.nan)
    if not _is_present(label):
        return np.full(4, np.nan)
    if 'rect' not in label:
        raise ValueError('label does not contain rect')
    rect = label['rect']
    min_pt = [rect['xmin'], rect['ymin']]
    max_pt = [rect['xmax'], rect['ymax']]
    return geom_np.make_rect(min_pt, max_pt)


def select_frames(seq, times):
    return Sequence(
        image_files=[seq.image_files[t] for t in times],
        valid_set=set(i for i, t in enumerate(times) if t in seq.valid_set),
        rects=[seq.rects[t] for t in times],
        aspect=seq.aspect,
        name=seq.name,
    )


def sequence_to_example(seq):
    return itermodel.ExampleUnroll(
        features_init={
            'image': {'file': seq.image_files[0]},
            'aspect': seq.aspect,
            'rect': seq.rects[0],
        },
        features={
            'image': {'file': seq.image_files[1:]}
        },
        labels={
            'valid': [(t in seq.valid_set) for t in range(1, len(seq))],
            'rect': seq.rects[1:],
        },
    )


def example_fn_from_times_fn(times_fn):
    def f(rand, seq, **kwargs):
        times = times_fn(rand, len(seq), seq.valid_set, **kwargs)
        return sequence_to_example(select_frames(seq, times))
    return f


def times_uniform(rand, seq_len, valid_set, ntimesteps):
    if len(valid_set) < 2:
        raise RuntimeError('not enough labels: {}'.format(len(valid_set)))
    valid_frames = sorted(valid_set)
    times = rand.choice(valid_frames, size=ntimesteps + 1, replace=True)
    return times


def times_regular(rand, seq_len, valid_set, freq):
    # Sample frames with `freq`, regardless of label
    # (only the first frame need to have label).
    # Note also that the returned frames can have length < ntimesteps+1.
    a = rand.choice(sorted(seq.valid_set))
    times = [int(round(a + freq * i)) for i in range(ntimesteps + 1)]
    times = [t for t in times if t < len(seq)]
    return sequence_to_example(select_frames(seq, times))


def times_freq_range(rand, seq_len, valid_set, ntimesteps, min_freq, max_freq, use_log):
    # Choose frames:
    #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
    # Therefore, for frames [0, ..., ntimesteps], we need:
    #   a + ntimesteps*freq <= seq_len - 1
    # The smallest possible value of a is 0
    #   0 + ntimesteps*freq <= seq_len - 1
    #   ntimesteps*freq <= seq_len - 1
    #   freq <= (seq_len - 1) / ntimesteps
    u = min_freq
    v = min(max_freq, (seq_len - 1) / ntimesteps)
    if not u <= v:
        raise RuntimeError('cannot satisfy freqency range')
    if use_log:
        freq = math.exp(rand.uniform(math.log(u), math.log(v)))
    else:
        freq = rand.uniform(u, v)
    # Let n = ntimesteps*freq.
    n = int(round(ntimesteps * freq))
    # Choose first frame such that all frames are present.
    a = rand.choice(sorted([a for a in valid_set if a + n <= seq_len]))
    times = [int(round(a + freq * t)) for t in range(ntimesteps + 1)]
    return times


uniform = example_fn_from_times_fn(times_uniform)
regular = example_fn_from_times_fn(times_regular)
freq_range = example_fn_from_times_fn(times_freq_range)

EXAMPLE_FNS = {name: globals()[name] for name in  [
    'uniform',
    'regular',
    'freq_range',
]}


def choose_pair_uniform_range(rand, seq, is_valid, low, high):
    '''Samples two frames whose distance apart is in [low, high).'''
    n = len(seq)
    subset = sorted(seq.valid_set)
    if len(subset) < 2:
        # Need at least two labels.
        return None
    # |x - y| is in [gap_min, gap_max].
    max_dist = subset[-1] - subset[0]
    min_dist = min(b - a for a, b in zip(subset, subset[1:]))

    if high is None or high > n:
        high = n  # Max distance is n - 1.
    if low is None or low < 1:
        low = 1
    assert low < high

    # Does [low, high) = [low, high - 1] intersect [min_dist, max_dist]?
    # If (low <=) high - 1 < min_dist (<= max_dist), there are no pairs.
    # If (min_dist <=) max_dist < low (<= high - 1), there are no pairs.
    if high - 1 < min_dist or max_dist < low:
        return None
    # # To avoid these situations, clip the interval to [min_dist, max_dist].
    # assert min_dist <= max_dist
    # low = max(min_dist, min(max_dist, low))
    # high = max(min_dist, min(max_dist, high - 1)) + 1

    # Obtain a list of all frames that have a partner in [low, high + 1).
    candidates_a = find_with_neighbor(is_valid, low, high)
    # Uniformly choose frame from candidates.
    a = rand.choice(candidates_a)
    candidates_b = [t for t in range(a + low, a + high) if is_valid[t]]
    assert len(candidates_b) > 0
    # Uniformly choose partner from candidates.
    b = rand.choice(candidates_b)

    # Flip order.
    # a, b = rand.choice([(a, b), (b, a)])
    return (a, b)


def find_with_neighbor(is_valid, low, high):
    '''Finds the elements that have a subsequent neighbour with distance in [low, high).

    Finds the indices i such that is_valid[i] is true and
    there exists j such that j - i in [low, high) and is_valid[j] is true.

    Args:
        is_valid: List of bools with length n.

    >>> find_with_neighbor([0] * 4, 1, 3)
    []
    >>> find_with_neighbor([1] * 4, 1, 4)
    [0, 1, 2]
    >>> find_with_neighbor([1] * 4, 2, 4)
    [0, 1]
    >>> find_with_neighbor([1, 0, 1, 0, 1], 1, 2)
    []
    >>> find_with_neighbor([1, 0, 1, 0, 1], 1, 3)
    [0, 2]
    >>> find_with_neighbor([1, 1, 0, 0, 1, 1], 2, 3)
    []
    >>> find_with_neighbor([1, 1, 0, 0, 1, 1], 1, 3)
    [0, 4]
    >>> find_with_neighbor([1, 1, 0, 0, 1, 1], 1, 4)
    [0, 1, 4]
    >>> find_with_neighbor([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], 1, 4)
    [1]
    >>> find_with_neighbor([0, 1, 0, 0, 0, 1, 0, 0, 1, 0], 1, 4)
    [5]
    >>> find_with_neighbor([1, 1, 0, 1], 2, 3)
    [1]
    '''

    n = len(is_valid)
    if low is None:
        low = 1
    assert low >= 1
    if high is None:
        high = n
    # If |x - y| in [low, high) and y > x
    # then y - x in [low, high).
    # This is equivalent to y in [x + low, x + high).
    result = []
    x = 0
    a = low
    b = high
    # Initialize count.
    count = sum(1 for y in range(a, b) if is_valid[y])
    # Continue until [a, b) is outside [0, n)
    while a < n:
        if is_valid[x] and count > 0:
            result.append(x)
        # Advance one position.
        x += 1
        # Lose element a.
        if is_valid[a]:
            count -= 1
        # Gain element b.
        if b < n and is_valid[b]:
            count += 1
        a += 1
        b += 1
    return result


def choose_disjoint_uniform(rand, is_valid, k):
    '''Chooses an initial frame and a subsequent consecutive sequence.

    At least one of the consecutive frames must have a valid label.

    Note that some algorithms might require a rectangle even if the label is not valid.
    This can be achieved by interpolating between valid frames or filling with the previous value.

    >>> choose_disjoint_uniform(np.random, [1, 0, 0, 1], 2)
    [0, 2, 3]
    >>> choose_disjoint_uniform(np.random, [1, 1], 1)
    [0, 1]
    >>> choose_disjoint_uniform(np.random, [1, 1, 1, 1], 3)
    [0, 1, 2, 3]
    >>> choose_disjoint_uniform(np.random, [1, 0, 0, 1], 3)
    [0, 1, 2, 3]
    >>> choose_disjoint_uniform(np.random, [1, 1, 0, 0], 3)
    [0, 1, 2, 3]
    >>> choose_disjoint_uniform(np.random, [0, 0, 1, 0, 1], 2)
    [2, 3, 4]
    '''
    # TODO: Could use mocking to better test edge cases above?
    # (have `rand` return first or last)

    n = len(is_valid)
    subset = [t for t in range(n) if is_valid[t]]
    if len(subset) < 2:
        return None  # Need at least 2 frames with labels.

    # We need to find frames t0, t1, t1 + 1, ..., t1 + k - 1.
    # The first frame must come before the last k frames and the last valid label.
    t0_high = min(n - k, subset[-1])
    if not subset[0] < t0_high:
        return None  # No frames with labels that satisfy the constraint.
    candidates_t0 = [t for t in subset if t < t0_high]
    t0 = rand.choice(candidates_t0)

    # Find another frame with a label after t0.
    # This could become any of t1, ..., t1 + k - 1.
    candidates_u = [t for t in subset if t > t0]
    u = rand.choice(candidates_u)
    # Now we choose a range of length n that contains u.
    # t1 in [u - k + 1, u] and t1 in [t0 + 1, n - k]
    t1_low = max(u - k + 1, t0 + 1)
    t1_high = min(u, n - k) + 1
    assert t1_low < t1_high
    t1 = rand.randint(t1_low, t1_high)

    return [t0] + [t1 + i for i in range(k)]
