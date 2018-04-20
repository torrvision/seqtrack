'''A sampler is a function that maps a dataset to an ordered collection of sequences.

The sequences can be returned as a list or the functions can be generators.
'''

import pdb
import math
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)

from seqtrack import data
from seqtrack import geom_np


# Thoughts:
#
# During training, we just need a never-ending stream of examples.
# However, sometimes during training, we want to obtain an epoch.
# During evaluation, we either want to
# - take n sequences from a larger (possibly infinite) stream or
# - run the tracker on all videos in a dataset
#
# The use of asynchronous data loading, balanced mixtures of datasets, etc
# makes it difficult to have a perfect resume
# (which might otherwise be done at the epoch).
#
# We could have a DataSource class.
# This can have an epoch or not.


def sample(dataset, rand=None, shuffle=False, max_videos=None, max_objects=None,
           kind=None, ntimesteps=None, freq=10, min_freq=10, max_freq=60, use_log=True):
    '''
    For training, set `shuffle=True`, `max_objects=1`, `ntimesteps` as required.

    Note that all samplers for training use `ntimesteps` to limit the sequence length.
    The `full` sampler does not use `ntimesteps`.

    This sampler comprises options that are used to choose a list of trajectories
    (`shuffle`, `max_videos`, `max_objects`)
    and options that are used to choose a list of frames
    (`kind`, `ntimesteps`, `freq`, ...).

    Args:
        dataset: Dataset object such as ILSVRC or OTB.
        rand: Random number rand (can be module `random`).
        shuffle: Whether to shuffle the videos.
            Note that if shuffled sequences are desired,
            then max_objects should be 1,
            otherwise all trajectories from the same video
            will be returned together.
        max_videos: Maximum number of videos to use, or None.
            There may still be multiple tracks per video.
        max_objects: Maximum number of objects per video, or None.
        kind: Type of sampler to use.
            {'full', 'sampling', 'regular', 'freq-range-fit'}
        ntimesteps: Maximum number of frames after first frame, or None.
    '''

    tracks_by_video = data.get_tracks_by_video(dataset)

    def _select_frames(valid_frames):
        valid_frames = sorted(valid_frames)
        valid_frames_set = set(valid_frames)
        t_min, t_max = valid_frames[0], valid_frames[-1]
        video_len = t_max - t_min + 1
        is_valid = {t: t in valid_frames_set for t in range(t_min, t_max + 1)}

        if kind == 'sampling':
            k = min(len(valid_frames), ntimesteps+1)
            return sorted(rand.choice(valid_frames, k, replace=False))
        elif kind == 'freq-range-fit':
            # TODO: The scope of this sampler should include
            # choosing objects within videos.
            # Choose frames:
            #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
            # Therefore, for frames [0, ..., ntimesteps], we need:
            #   a + ntimesteps*freq <= video_len - 1
            # The smallest possible value of a is valid_frames[0]
            #   valid_frames[0] + ntimesteps*freq <= video_len - 1
            #   ntimesteps*freq <= video_len - 1 - valid_frames[0]
            #   freq <= (video_len - 1 - valid_frames[0]) / ntimesteps
            u = min_freq
            v = min(max_freq, float((video_len - 1) - valid_frames[0]) / ntimesteps)
            if not u <= v:
                return None
            if use_log:
                f = math.exp(rand.uniform(math.log(u), math.log(v)))
            else:
                f = rand.uniform(u, v)
            # Let n = ntimesteps*f.
            n = int(round(ntimesteps * f))
            # Choose first frame such that all frames are present.
            a = rand.choice([a for a in valid_frames if a + n <= t_max])
            return [int(round(a + f*t)) for t in range(0, ntimesteps+1)]
        elif kind == 'regular':
            ''' Sample frames with `freq`, regardless of label
            (only the first frame need to have label).
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,0,1,0,0].
            Note also that the returned frames can have length < ntimesteps+1.
            Adaptive frequency or gradually increasing frequency as a
            Curriculum Learning might be tried.
            '''
            frames = range(rand.choice(valid_frames), video_len, freq)
            return frames[:ntimesteps+1]
        elif kind == 'full':
            ''' The full sequence from first 1 to last 1, regardless of label.
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,1,0,0,1,1].
            This option is used to evaluate full-length sequences.
            '''
            return range(t_min, t_max+1)
        else:
            raise ValueError('unknown sampler: {}'.format(kind))

    assert((ntimesteps is None) == (kind == 'full'))
    # videos = list(dataset.videos) # copy

    videos = list(sample_videos(dataset, shuffle, rand))

    ## JV: Shuffle all videos even if max_videos specified.
    ## if max_videos is not None and len(videos) > max_videos:
    ##     videos = rand.choice(videos, max_videos, replace=False)
    ## else:
    ##     if shuffle:
    ##         rand.shuffle(videos)
    if not shuffle and (max_videos is not None and max_videos < len(videos)):
        raise ValueError('enable shuffle or remove limit on number of videos')

    num_videos = 0
    for video_id in videos:
        if max_videos is not None and not num_videos < max_videos:
            break

        ## JV: Shuffle all trajectories even if max_objects specified.
        ## trajectories = dataset.tracks[video_id]
        ## if max_objects is not None and len(trajectories) > max_objects:
        ##     trajectories = rand.choice(trajectories, max_objects, replace=False)
        # Construct (index, trajectory) pairs.
        video_tracks = list(enumerate(tracks_by_video[video_id]))
        if max_objects is not None and len(video_tracks) > max_objects:
            rand.shuffle(video_tracks)

        ## for cnt, trajectory in enumerate(video_tracks):
        num_objects = 0
        for index, track_id in video_tracks:
            if max_objects is not None and not num_objects < max_objects:
                break

            labels = dataset.labels(track_id)
            present_frames = [t for t, l in labels.items() if not l.get('absent', False)]
            if len(present_frames) == 0:
                logger.warning('no present frames: track "%s" in video "%s"', track_id, video_id)
                continue
            frames = _select_frames(present_frames)
            if not frames:
                logger.warning('failed to select frames: track "%s" in video "%s"',
                               track_id, video_id)
                continue
            label_is_valid = [t in labels and not labels[t].get('absent', False) for t in frames]
            # Skip sequences with no labels (after first label).
            num_labels = sum(1 for x in label_is_valid if x)
            if num_labels < 2:
                logger.warning('less than two labels: track "%s" in video "%s"', track_id, video_id)
                continue
            # width, height = dataset.original_image_size[video_id]
            yield {
                'image_files':         [dataset.image_file(video_id, t) for t in frames],
                'viewports':           [geom_np.unit_rect() for _ in frames],
                'labels':              [_rect_from_label(labels.get(t, None)) for t in frames],
                'label_is_valid':      label_is_valid,
                'aspect':              dataset.aspect(video_id),
                # 'original_image_size': dataset.original_image_size[video_id],
                'video_name':          (video_id + '-{}'.format(index)) if len(video_tracks) > 1
                                       else video_id,
            }
            num_objects += 1

        if num_objects > 0:
            num_videos += 1


def _rect_from_label(label):
    if label is None:
        return _invalid_rect()
    if label.get('absent', False):
        return _invalid_rect()
    if 'rect' not in label:
        raise ValueError('label does not contain rect')
    rect = label['rect']
    return [rect['xmin'], rect['ymin'], rect['xmax'], rect['ymax']]


def _invalid_rect():
    return [float('nan')] * 4


def sample_videos(dataset, shuffle, rand):
    '''
    Args:
        dataset:
            Must either have dataset.sample_videos(shuffle, rand) that returns a list or
            dataset.videos that is a list.
            Typically either DatasetMixture or data.Data_ILSVRC, data.CSV, etc.

    Returns:
        Finite generator of length len(dataset.videos).
    '''
    try:
        videos = dataset.sample_videos(shuffle, rand)
    except AttributeError:
        # videos = list(dataset.videos)
        videos = data.get_videos(dataset)
        if shuffle:
            rand.shuffle(videos)
    for video in videos:
        yield video


class DatasetMixture(data.Concat):

    '''
    If shuffle is False, the component will still be chosen stochastically
    but the videos within the component will not be shuffled.
    '''

    def __init__(self, components):
        datasets = {k: dataset for k, (_, dataset) in components.items()}
        super(DatasetMixture, self).__init__(datasets)
        # data.Concat.__init__(self, datasets)
        self.components = components

    def sample_videos(self, shuffle, rand):
        names = self.components.keys()
        weights = np.array([weight for weight, _ in (self.components[k] for k in names)])
        p = weights / np.sum(weights)

        samplers = {name: (video for video in []) for name in names}
        for i in range(len(data.get_videos(self))):
            name = rand.choice(names, p=p)
            try:
                video = next(samplers[name])
            except StopIteration:
                samplers[name] = sample_videos(self.datasets[name], shuffle, rand)
                video = next(samplers[name])
            yield name + '/' + video
