'''A sampler is a function that maps a dataset to an ordered collection of sequences.

The sequences can be returned as a list or the functions can be generators.
'''

import pdb
import math
import numpy as np
import os

import data
import geom_np


def sample(dataset, rand=None, shuffle=False, max_videos=None, max_objects=None,
           kind=None, ntimesteps=None, freq=10, min_freq=10, max_freq=60):
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
    def _select_frames(is_valid, valid_frames):
        if kind == 'sampling':
            k = min(len(valid_frames), ntimesteps+1)
            return sorted(rand.choice(valid_frames, k, replace=False))
        elif kind == 'freq-range-fit':
            # TODO: The scope of this sampler should include
            # choosing objects within videos.
            video_len = len(is_valid)
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
            f = math.exp(rand.uniform(math.log(u), math.log(v)))
            # Let n = ntimesteps*f.
            n = int(round(ntimesteps * f))
            # Choose first frame such that all frames are present.
            a = rand.choice([a for a in valid_frames if a + n <= video_len - 1])
            return [int(round(a + f*t)) for t in range(0, ntimesteps+1)]
        elif kind == 'regular':
            ''' Sample frames with `freq`, regardless of label
            (only the first frame need to have label).
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,0,1,0,0].
            Note also that the returned frames can have length < ntimesteps+1.
            Adaptive frequency or gradually increasing frequency as a
            Curriculum Learning might be tried.
            '''
            num_frames = len(is_valid)
            frames = range(rand.choice(valid_frames), num_frames, freq)
            return frames[:ntimesteps+1]
        elif kind == 'full':
            ''' The full sequence from first 1 to last 1, regardless of label.
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,1,0,0,1,1].
            This option is used to evaluate full-length sequences.
            '''
            return range(valid_frames[0], valid_frames[-1]+1)
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
    for video in videos:
        if max_videos is not None and not num_videos < max_videos:
            break

        ## JV: Shuffle all trajectories even if max_objects specified.
        ## trajectories = dataset.tracks[video]
        ## if max_objects is not None and len(trajectories) > max_objects:
        ##     trajectories = rand.choice(trajectories, max_objects, replace=False)
        # Construct (index, trajectory) pairs.
        trajectories = list(enumerate(dataset.tracks[video]))
        if max_objects is not None and len(trajectories) > max_objects:
            rand.shuffle(trajectories)

        ## for cnt, trajectory in enumerate(trajectories):
        num_objects = 0
        for cnt, trajectory in trajectories:
            if max_objects is not None and not num_objects < max_objects:
                break

            frame_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
            frames = _select_frames(frame_is_valid, trajectory.keys())
            if not frames:
                continue
            label_is_valid = [(t in trajectory) for t in frames]
            # Skip sequences with no labels (after first label).
            num_labels = sum(1 for x in label_is_valid if x)
            if num_labels < 2:
                continue
            width, height = dataset.original_image_size[video]
            yield {
                'image_files':         [dataset.image_file(video, t) for t in frames],
                'viewports':           [geom_np.unit_rect() for _ in frames],
                'labels':              [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid':      label_is_valid,
                'aspect':              float(width) / float(height),
                'original_image_size': dataset.original_image_size[video],
                'video_name':          video + '-{}'.format(cnt) if len(trajectories) > 1 else video,
            }
            num_objects += 1

        if num_objects > 0:
            num_videos += 1


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
        videos = list(dataset.videos)
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
        for i in range(len(self.videos)):
            name = rand.choice(names, p=p)
            try:
                video = next(samplers[name])
            except StopIteration:
                samplers[name] = sample_videos(self.datasets[name], shuffle, rand)
                video = next(samplers[name])
            yield name + '/' + video
