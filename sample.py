'''The functions in this file create collections of sequences from datasets.

The sequences can be returned as a list or the functions can act as generators.
'''

import pdb
import numpy as np
import math
import os
import random

def sample(dataset, ntimesteps=None, seqtype=None, shuffle=False,
           freq=10, min_freq=10, max_freq=60):
    '''
    Args:
        dataset: Dataset object such as ILSVRC or OTB.
        ntimesteps: Maximum number of frames after first frame.
            If None, then there is no limit.
    '''
    def _select_frames(is_valid, valid):
        if seqtype == 'sampling':
            k = min(len(valid), ntimesteps+1)
            return sorted(random.sample(valid, k))
        elif seqtype == 'freq-range-fit':
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
            freq = math.exp(random.uniform(math.log(u), math.log(v)))
            # Let n = ntimesteps*freq.
            n = int(round(ntimesteps * freq))
            # Choose first frame such that all frames are present.
            a = random.choice([a for a in valid if a + n <= video_len - 1])
            return [int(round(a + freq*t)) for t in range(0, ntimesteps+1)]
        elif seqtype == 'regular':
            ''' Sample frames with `freq`, regardless of label
            (only the first frame need to have label).
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,0,1,0,0].
            Note also that the returned frames can have length < ntimesteps+1.
            Adaptive frequency or gradually increasing frequency as a
            Curriculum Learning might be tried.
            '''
            frames = range(random.choice(valid), len(is_valid), freq)
            return frames[:ntimesteps+1]
        elif seqtype == 'full':
            ''' The full sequence from first 1 to last 1, regardless of label.
            Thus, the returned frames can be `SPARSE`, e.g., [1,1,1,1,0,0,1,1].
            This option is used to evaluate full-length sequences.
            '''
            return range(valid[0], valid[-1]+1)

    num_videos = len(dataset.videos)
    indices = np.random.permutation(num_videos) if shuffle else range(num_videos)
    videos = list(dataset.videos[i] for i in indices)
    for video in videos:
        if ntimesteps and seqtype is not 'full':
            trajectories = [random.choice(dataset.tracks[video])]
        else:
            trajectories = dataset.tracks[video]
        for trajectory in trajectories:
            frame_is_valid = [(t in trajectory) for t in range(dataset.video_length[video])]
            frames = _select_frames(frame_is_valid, trajectory.keys())
            if not frames:
                continue
            yield {
                'image_files':    [dataset.image_file(video, t) for t in frames],
                'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid': [(t in trajectory) for t in frames],
                'original_image_size': dataset.original_image_size[video],
            }

def _invalid_rect():
    return [float('nan')] * 4
