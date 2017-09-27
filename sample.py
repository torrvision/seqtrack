'''A sampler is a function that maps a dataset to an ordered collection of sequences.

The sequences can be returned as a list or the functions can be generators.
'''

import pdb
import math
import os

def sample(dataset, generator=None, shuffle=False, max_videos=None, max_objects=None,
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
        generator: Random number generator (can be module `random`).
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
            return sorted(generator.sample(valid_frames, k))
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
            f = math.exp(generator.uniform(math.log(u), math.log(v)))
            # Let n = ntimesteps*f.
            n = int(round(ntimesteps * f))
            # Choose first frame such that all frames are present.
            a = generator.choice([a for a in valid_frames if a + n <= video_len - 1])
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
            frames = range(generator.choice(valid_frames), num_frames, freq)
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
    videos = list(dataset.videos) # copy
    if max_videos is not None and len(videos) > max_videos:
        videos = generator.sample(videos, max_videos)
    else:
        if shuffle:
            generator.shuffle(videos)

    for video in videos:
        trajectories = dataset.tracks[video]
        if max_objects is not None and len(trajectories) > max_objects:
            trajectories = generator.sample(trajectories, max_objects)
        # Do not shuffle objects within a sequence.
        # Assume that if the sampler is used for SGD, then max_objects = 1.

        for cnt, trajectory in enumerate(trajectories):
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
                'labels':              [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid':      label_is_valid,
                'aspect':              float(width) / float(height),
                'original_image_size': dataset.original_image_size[video],
                'video_name':          video + '-{}'.format(cnt) if len(trajectories) > 1 else video,
            }

def _invalid_rect():
    return [float('nan')] * 4
