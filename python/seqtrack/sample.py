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

import collections
import math
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import geom_np
import trackdat
from trackdat.dataset import is_present as _is_present


class EpochSampler(object):
    '''Standard sampler for a finite dataset.'''

    def __init__(self, rand, dataset, repeat=False):
        self._rand = rand
        self._dataset = dataset
        self._repeat = repeat

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

    def __init__(self, rand, samplers, p=None):
        '''
        Args:
            samplers: Dictionary that maps dataset name to sampler.
            p: Dictionary that maps dataset name to weight.
        '''
        self._rand = rand
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
    '''Describes a sequence with optional labels.

    The sequence spans frames `0 <= t < len(sequence)`.
    '''

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
        return len(self.image_files)


def select_frames(seq, times):
    '''Extracts a subset of times from a sequence.'''
    try:
        image_files = [seq.image_files[t] for t in times]
    except IndexError as ex:
        raise RuntimeError('invalid times (sequence length {}): {}'.format(len(seq), str(times)))
    return Sequence(
        image_files=image_files,
        valid_set=set(i for i, t in enumerate(times) if t in seq.valid_set),
        rects=[seq.rects[t] for t in times],
        aspect=seq.aspect,
        name=seq.name,
    )


def extract_sequence_from_dataset(dataset, track_id, times=None):
    '''Constructs a Sequence object for a track in a dataset.

    If `times` is not specified, the sequence extends from the first to the last valid frame.
    '''
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
        valid_set=set(i for i, t in enumerate(times) if t in labels and _is_present(labels[t])),
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


class ExampleTypeKeys(object):
    '''Types of example for training.

    Different examples may have different `features` and `labels`.

    CONSECUTIVE:
    The frames represent consecutive frames of video.
    features:
        image_init: [b, h, w, c]
        rect_init: [b, 4]
        images:
            data: [b, t, h, w, c]
    labels:
        valid: [b, t]
        rects: [b, t, 4]

    UNORDERED:
    The images have no temporal relation.
    The `features` and `labels` are the same as CONSECUTIVE.

    SEPARATE_INIT:
    The initial frame defines the appearance model.
    But then this model should be used from the position in the next frame.
    The `features` and `labels` are the same as CONSECUTIVE.
    The first label should be valid.
    '''

    CONSECUTIVE = 'consecutive'
    UNORDERED = 'unordered'
    SEPARATE_INIT = 'separate_init'


ExampleSequence = collections.namedtuple('ExampleSequence', [
    'features_init',
    'features',
    'labels',
])


ExampleStep = collections.namedtuple('ExampleStep', [
    'features_init',
    'features_curr',
    'labels_curr',
])


class InvalidExampleException(Exception):

    def __init__(self, message):
        self._message = message

    def __str__(self):
        return self._message


def sequence_to_example(seq):
    '''Turns a sequence into an example.

    The first label must be valid.
    There must be at least one valid label in the rest of the sequence.
    Otherwise an `InvalidExampleException` will be raised.
    '''
    is_valid = [(t in seq.valid_set) for t in range(len(seq))]
    if not is_valid[0]:
        raise InvalidExampleException('first label is not valid')
    if not any(is_valid[1:]):
        raise InvalidExampleException('no valid labels after first label')

    return ExampleSequence(
        features_init={
            'image': {'file': seq.image_files[0]},
            'aspect': seq.aspect,
            'rect': seq.rects[0],
        },
        features={
            'image': {'file': seq.image_files[1:]}
        },
        labels={
            'valid': is_valid[1:],
            'rect': seq.rects[1:],
        },
    )


def _example_fn_from_times_fn(times_fn):
    '''Transforms a function that returns times into one that returns an example.'''
    def f(rand, seq, **kwargs):
        times = times_fn(rand, len(seq), seq.valid_set, **kwargs)
        return sequence_to_example(select_frames(seq, times))
    return f


def times_uniform(rand, seq_len, valid_set, ntimesteps):
    '''Chooses a random set of valid frames.

    We use the data structure of a sequence, but the frames have no order.
    '''
    if len(valid_set) < 2:
        raise InvalidExampleException('not enough labels: {}'.format(len(valid_set)))
    valid_frames = sorted(valid_set)
    times = rand.choice(valid_frames, size=ntimesteps + 1, replace=True)
    return times


def times_regular(rand, seq_len, valid_set, ntimesteps, freq):
    '''Samples a sub-sequence with a fixed frequency.

    May choose a sub-sequence with no valid frames except the first.
    '''
    # Sample frames with `freq`, regardless of label
    # (only the first frame need to have label).
    # Note also that the returned frames can have length < ntimesteps+1.
    a = rand.choice(sorted(valid_set))
    times = [int(round(a + freq * i)) for i in range(ntimesteps + 1)]
    times = [t for t in times if t < seq_len]
    return times


def times_freq_range(rand, seq_len, valid_set, ntimesteps, min_freq, max_freq, use_log):
    '''Samples a sub-sequence with a frequency in [min_freq, max_freq].

    Chooses frames `a + round(freq * i)` for `i` in [0, 1, ..., ntimesteps].
    The first frame `a` must be valid.
    Raises an exception if there are no such sub-sequences with `freq >= min_freq`.
    May choose a sub-sequence where all subsequent frames are not valid.
    '''
    # Choose frames:
    #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
    # Therefore, for frames [0, ..., ntimesteps], we need:
    #   a + ntimesteps*freq <= seq_len - 1
    # The smallest possible value of a is t_first = min(valid_set).
    #   t_first + ntimesteps*freq <= seq_len - 1
    #   ntimesteps*freq <= seq_len - 1 - t_first
    #   freq <= (seq_len - 1 - t_first) / ntimesteps
    t_first = min(valid_set)
    u = min_freq
    v = min(max_freq, (seq_len - 1 - t_first) / ntimesteps)
    if not u <= v:
        raise InvalidExampleException('cannot satisfy freqency range')
    if use_log:
        freq = math.exp(rand.uniform(math.log(u), math.log(v)))
    else:
        freq = rand.uniform(u, v)
    # Let n = ntimesteps*freq.
    n = int(round(ntimesteps * freq))
    # Choose first frame such that all frames are present.
    a = rand.choice(sorted([a for a in valid_set if a + n <= seq_len - 1]))
    times = [int(round(a + freq * t)) for t in range(ntimesteps + 1)]
    return times


def times_pair_range(rand, seq_len, valid_set, low, high):
    '''Samples two valid frames whose distance apart is in [low, high).

    Raises an exception if no such pairs exist.

    The first frame is chosen uniformly from the feasible set (of frames with a partner).
    The second frame is chosen uniformly from the feasible set (of partners).
    '''
    # Need at least two labels.
    if len(valid_set) < 2:
        raise InvalidExampleException('not enough labels: {}'.format(len(valid_set)))
    subset = sorted(valid_set)
    # |x - y| is in [gap_min, gap_max].
    max_dist = subset[-1] - subset[0]
    min_dist = min(b - a for a, b in zip(subset, subset[1:]))

    if low is not None and high is not None:
        assert low < high
    if high is None:
        high = seq_len  # Max distance is seq_len - 1.
    if low is None:
        low = 1
    high = min(seq_len, high)
    low = max(1, low)

    # Does [low, high) = [low, high - 1] intersect [min_dist, max_dist]?
    # If (low <=) high - 1 < min_dist (<= max_dist), there are no pairs.
    # If (min_dist <=) max_dist < low (<= high - 1), there are no pairs.
    if high - 1 < min_dist or max_dist < low:
        raise InvalidExampleException('cannot satisfy range')
    # # To avoid these situations, clip the interval to [min_dist, max_dist].
    # assert min_dist <= max_dist
    # low = max(min_dist, min(max_dist, low))
    # high = max(min_dist, min(max_dist, high - 1)) + 1

    # Obtain a list of all frames that have a partner in [low, high + 1).
    candidates_a = find_with_subsequent_within(seq_len, valid_set, low, high)
    if not candidates_a:
        raise InvalidExampleException('no pairs in desired range: [{}, {})'.format(low, high))
    # Uniformly choose frame from candidates.
    a = rand.choice(candidates_a)
    candidates_b = [t for t in range(a + low, a + high) if (t in valid_set)]
    assert len(candidates_b) > 0
    # Uniformly choose partner from candidates.
    b = rand.choice(candidates_b)

    # Flip order.
    # a, b = rand.choice([(a, b), (b, a)])
    return [a, b]


def find_with_subsequent_within(seq_len, valid_set, low, high):
    '''Finds the elements that have a subsequent neighbour with distance in [low, high).

    Finds the indices i such that (i in valid_set) and
    there exists j such that j - i in [low, high) and (j in valid_set) is true.

    >>> find_with_subsequent_within(4, set(), 1, 3)
    []
    >>> find_with_subsequent_within(4, set(range(4)), 1, 4)
    [0, 1, 2]
    >>> find_with_subsequent_within(4, set(range(4)), 2, 4)
    [0, 1]
    >>> find_with_subsequent_within(5, set([0, 2, 4]), 1, 2)
    []
    >>> find_with_subsequent_within(5, set([0, 2, 4]), 1, 3)
    [0, 2]
    >>> find_with_subsequent_within(6, set([0, 1, 4, 5]), 2, 3)
    []
    >>> find_with_subsequent_within(6, set([0, 1, 4, 5]), 1, 3)
    [0, 4]
    >>> find_with_subsequent_within(6, set([0, 1, 4, 5]), 1, 4)
    [0, 1, 4]
    >>> find_with_subsequent_within(10, set([1, 4, 8]), 1, 4)
    [1]
    >>> find_with_subsequent_within(10, set([1, 5, 8]), 1, 4)
    [5]
    >>> find_with_subsequent_within(4, set([0, 1, 3]), 2, 3)
    [1]
    '''

    if low is None:
        low = 1
    assert low >= 1
    if high is None:
        high = seq_len
    # If |x - y| in [low, high) and y > x
    # then y - x in [low, high).
    # This is equivalent to y in [x + low, x + high).
    result = []
    x = 0
    a = low
    b = high
    # Initialize count.
    count = sum(1 for y in range(a, b) if (y in valid_set))
    # Continue until [a, b) is outside [0, seq_len)
    while a < seq_len:
        if (x in valid_set) and count > 0:
            result.append(x)
        # Advance one position.
        x += 1
        # Lose element a.
        if (a in valid_set):
            count -= 1
        # Gain element b.
        if b < seq_len and (b in valid_set):
            count += 1
        a += 1
        b += 1
    return result


def times_disjoint(rand, seq_len, valid_set, ntimesteps):
    '''Samples an initial frame and a consecutive sequence.
    The initial frame will not occur within the consecutive sequence.
    The first frame of the sequence must be valid, and it should contain another valid frame.

    Examples of (deterministic) edge cases:
    >>> sorted(times_disjoint(np.random, 3, set(range(3)), 2))
    [0, 1, 2]
    >>> times_disjoint(np.random, 4, set([0, 2, 3]), 2)
    [0, 2, 3]
    >>> times_disjoint(np.random, 4, set([0, 1, 3]), 2)
    [3, 0, 1]
    >>> times_disjoint(np.random, 8, set([2, 5, 7]), 3)
    [2, 5, 6, 7]
    >>> times_disjoint(np.random, 8, set([0, 4, 5]), 4)
    [0, 4, 5, 6, 7]
    >>> times_disjoint(np.random, 8, set([0, 4, 5, 6]), 4)
    [0, 4, 5, 6, 7]
    '''
    assert ntimesteps >= 2
    # Find all candidates for the consecutive part.
    starts = sorted(valid_set)
    # Require that the sub-sequence does not extend beyond the length of the sequence.
    starts = [s for s in starts if s + ntimesteps - 1 < seq_len]
    # Require that there is at least one other valid frame within the sub-sequence.
    is_valid = [int(t in valid_set) for t in range(seq_len)]
    # num_valid_until[i] = sum(is_valid[:i])
    # sum(is_valid[i:j]) = sum(is_valid[:j]) - sum(is_valid[:i])
    #                    = num_valid_until[j] - num_valid_until[i]
    num_valid_until = np.cumsum([0] + is_valid)
    num_in_subseq_at = {s: num_valid_until[s + ntimesteps] - num_valid_until[s] for s in starts}
    starts = [s for s in starts if num_in_subseq_at[s] >= 2]
    # Require that there is at least one valid frame outside the sub-sequence.
    num_valid = len(is_valid)
    starts = [s for s in starts if num_valid - num_in_subseq_at[s] >= 1]
    if not starts:
        raise InvalidExampleException('no examples exist')
    s = rand.choice(starts)
    subseq = list(range(s, s + ntimesteps))

    others = valid_set.difference(subseq)
    t = rand.choice(sorted(others))
    return [t] + subseq


def times_disjoint_freq_range(rand, seq_len, valid_set, ntimesteps, min_freq, max_freq):
    '''Samples an initial frame and a consecutive sequence with regular frequency.
    The initial frame will not occur within the consecutive sequence.
    The first frame of the sequence must be valid, and it should contain another valid frame.

    >>> sorted(times_disjoint_freq_range(np.random, 10, set([0, 3, 6, 9]), 3, 3, 3))
    [0, 3, 6, 9]
    '''
    assert ntimesteps >= 2
    subseq = times_freq_range(rand, seq_len, valid_set,
                              ntimesteps=ntimesteps - 1,
                              min_freq=min_freq,
                              max_freq=max_freq,
                              use_log=False)
    min_time = min(subseq)
    max_time = max(subseq)
    others = [t for t in sorted(valid_set) if not (min_time <= t <= max_time)]
    if len(others) == 0:
        raise RuntimeError('no remaining frames for init')
    t = rand.choice(others)
    return [t] + subseq


uniform = _example_fn_from_times_fn(times_uniform)
regular = _example_fn_from_times_fn(times_regular)
freq_range = _example_fn_from_times_fn(times_freq_range)
pair_range = _example_fn_from_times_fn(times_pair_range)
disjoint = _example_fn_from_times_fn(times_disjoint)
disjoint_freq_range = _example_fn_from_times_fn(times_disjoint_freq_range)

EXAMPLE_FNS = {name: globals()[name] for name in  [
    'uniform',
    'regular',
    'freq_range',
    'pair_range',
    'disjoint',
    'disjoint_freq_range',
]}
