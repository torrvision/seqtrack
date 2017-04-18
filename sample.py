'''The functions in this file create collections of sequences from datasets.

The sequences can be returned as a list or the functions can act as generators.
'''

import pdb
import math
import os
import random

def sample(dataset, generator=None, shuffle=False, kind=None, ntimesteps=None,
           freq=10, min_freq=10, max_freq=60, max_sequences=None):
    '''
    Args:
        dataset: Dataset object such as ILSVRC or OTB.
        ntimesteps: Maximum number of frames after first frame.
            If None, then there is no limit.
    '''
    def _select_frames(is_valid, valid):
        if kind == 'sampling':
            k = min(len(valid), ntimesteps+1)
            return sorted(generator.sample(valid, k))
        elif kind == 'freq-range-fit':
            # TODO: The scope of this sampler should include
            # choosing objects within videos.
            video_len = len(is_valid)
            # Choose frames:
            #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
            # Therefore, for frames [0, ..., ntimesteps], we need:
            #   a + ntimesteps*freq <= video_len - 1
            # The smallest possible value of a is valid[0]
            #   valid[0] + ntimesteps*freq <= video_len - 1
            #   ntimesteps*freq <= video_len - 1 - valid[0]
            #   freq <= (video_len - 1 - valid[0]) / ntimesteps
            u = min_freq
            v = min(max_freq, float((video_len - 1) - valid[0]) / ntimesteps)
            if not u <= v:
                return None
            f = math.exp(generator.uniform(math.log(u), math.log(v)))
            # Let n = ntimesteps*f.
            n = int(round(ntimesteps * f))
            # Choose first frame such that all frames are present.
            a = generator.choice([a for a in valid if a + n <= video_len - 1])
            return [int(round(a + f*t)) for t in range(0, ntimesteps+1)]
        elif kind == 'regular':
            ''' Sample frames with `freq`, regardless of label
            (only the first frame need to have label).
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,0,1,0,0].
            Note also that the returned frames can have length < ntimesteps+1.
            Adaptive frequency or gradually increasing frequency as a
            Curriculum Learning might be tried.
            '''
            frames = range(generator.choice(valid), len(is_valid), freq)
            return frames[:ntimesteps+1]
        elif kind == 'full':
            ''' The full sequence from first 1 to last 1, regardless of label.
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,1,0,0,1,1].
            This option is used to evaluate full-length sequences.
            '''
            return range(valid[0], valid[-1]+1)

    # Use global generator if none supplied.
    generator = generator or random
    assert((ntimesteps is None) == (kind == 'full'))
    num_videos = len(dataset.videos)
    # indices = np.generator.permutation(num_videos) if shuffle else range(num_videos)
    indices = range(num_videos)
    if shuffle:
        random.shuffle(indices)
    videos = list(dataset.videos[i] for i in indices)
    num_sequences = 0
    for video in videos:
        if kind is not 'full':
            trajectories = [generator.choice(dataset.tracks[video])]
        else:
            trajectories = dataset.tracks[video]
        for trajectory in trajectories:
            frame_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
            frames = _select_frames(frame_is_valid, trajectory.keys())
            if not frames:
                continue
            label_is_valid = [(t in trajectory) for t in frames]
            # Skip sequences with no labels (after first label).
            num_labels = sum(1 for x in label_is_valid if x)
            if num_labels < 2:
                continue
            num_sequences += 1
            yield {
                'image_files':    [dataset.image_file(video, t) for t in frames],
                'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid': label_is_valid,
                'original_image_size': dataset.original_image_size[video],
            }
            if max_sequences is not None and num_sequences >= max_sequences:
                return

def _invalid_rect():
    return [float('nan')] * 4
