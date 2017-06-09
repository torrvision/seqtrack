'''A sampler is a function that maps a dataset to an ordered collection of sequences.

The sequences can be returned as a list or the functions can be generators.
'''

import argparse
import functools
import itertools
import math
import numpy as np
import os
import progressbar

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
            yield {
                'image_files':    [dataset.image_file(video, t) for t in frames],
                'labels':         [trajectory.get(t, _invalid_rect()) for t in frames],
                'label_is_valid': label_is_valid,
                'original_image_size': dataset.original_image_size[video],
            }


def _invalid_rect():
    return [float('nan')] * 4


class Motion:

    def __init__(self,
                 dataset,
                 ntimesteps,
                 min_sample_path_length,
                 max_sample_path_length,
                 min_original_path_length=0.0):
        self.dataset = dataset
        self.ntimesteps = ntimesteps
        self.min_original_path_length = min_original_path_length
        self.min_sample_path_length = min_sample_path_length
        self.max_sample_path_length = max_sample_path_length

    def init(self):
        self.cdfs = {}
        for video in self.dataset.videos:
            self.cdfs[video] = [
                motion_cdf(track, relative=True)
                for track in self.dataset.tracks[video]
            ]
        # Get pairs (video, obj_ind)
        objects = [(video, obj_ind)
            for video in self.dataset.videos
            for obj_ind in range(len(self.dataset.tracks[video]))
        ]
        # Filter pairs for minimum length.
        path_length = lambda video, obj_ind: self.cdfs[video][obj_ind][-1][1]
        objects = [(video, obj_ind) for video, obj_ind in objects
            if path_length(video, obj_ind) >= self.min_original_path_length
        ]
        # Index objects by video.
        self.object_subset = {}
        for video, obj_ind in objects:
            self.object_subset.setdefault(video, []).append(obj_ind)
        self.video_subset = [v for v in self.dataset.videos if v in self.object_subset]
        print 'subset of videos:', len(self.video_subset), 'of', len(self.dataset.videos)

    def sample(self, generator):
        video = generator.choice(self.video_subset)
        obj_ind = generator.choice(self.object_subset[video])
        cdf = self.cdfs[video][obj_ind]
        _, full_path_length = cdf[-1]
        assert full_path_length >= self.min_original_path_length
        # Limit example path length to maximum possible.
        # (Do this before choosing length to avoid always using same path.)
        lower_limit = self.min_sample_path_length
        upper_limit = min(self.max_sample_path_length, full_path_length)
        sample_path_length = generator.uniform(lower_limit, upper_limit)
        d0 = generator.uniform(0, full_path_length - sample_path_length)
        d1 = d0 + sample_path_length
        sample_dists = np.linspace(d0, d1, self.ntimesteps+1)
        cdf_dists = [dist for t, dist in cdf]
        inds = np.round(np.interp(sample_dists, cdf_dists, range(len(cdf_dists)))).astype(int)
        times = [t for t, dist in [cdf[ind] for ind in inds]]
        rects = [self.dataset.tracks[video][obj_ind][t] for t in times]
        print 'desired path length:', sample_path_length
        sample_track = dict(zip(times, rects))
        sample_cdf = motion_cdf(sample_track)
        # print 'sample cdf:', sample_cdf
        print 'sample path length:', sample_cdf[-1][1]
        return {
            'image_files':    [self.dataset.image_file(video, t) for t in times],
            'labels':         [self.dataset.tracks[video][obj_ind][t] for t in times],
            'label_is_valid': [True] * (self.ntimesteps+1),
            'original_image_size': self.dataset.original_image_size[video],
        }


def motion_cdf(track, relative=True, epsilon=1e-2):
    # track is a dictionary that maps frame index to rectangle
    track = sorted(track.items())
    times = [t for t, rect in track]
    rects = np.array([rect for t, rect in track])
    abs_dist = np.sum(np.abs(np.diff(rects, axis=0)), axis=1)
    if relative:
        rect_min = rects[:, 0:2]
        rect_max = rects[:, 2:4]
        size = np.sum(np.abs(rect_max - rect_min) + epsilon, axis=1)
        mean_size = 0.5 * (size[:-1] + size[1:])
        dist = abs_dist / mean_size
    else:
        dist = abs_dist
    total_dist = [0.0] + np.cumsum(dist).tolist()
    return zip(times, total_dist)


def main():
    import data
    import random
    from PIL import Image, ImageDraw, ImageColor

    def render_image(dst, src, rect, frmsz):
        im = Image.open(src)
        draw = ImageDraw.Draw(im)
        rect = (np.array(rect)*frmsz).tolist()
        draw.rectangle(rect, outline=ImageColor.getrgb('yellow'))
        del draw
        im.save(dst)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ilsvrc_dir', default='')
    parser.add_argument('--frmsz', type=int, default=241)
    parser.add_argument('--path_aux', default='aux')
    parser.add_argument('--path_stat', default='stat')
    parser.add_argument('--ntimesteps', type=int, default=5)
    args = parser.parse_args()

    print 'load dataset...'
    dataset = data.ILSVRC('train', frmsz=args.frmsz, path_data=args.ilsvrc_dir,
        path_aux=args.path_aux, path_stat=args.path_stat)
    sampler = Motion(dataset, ntimesteps=args.ntimesteps,
        min_sample_path_length=1.0*args.ntimesteps,
        max_sample_path_length=1.0*args.ntimesteps,
        min_original_path_length=1.0*args.ntimesteps)
    print 'initialize sampler...'
    sampler.init()

    sequence = sampler.sample(random)
    for t in range(args.ntimesteps+1):
        render_image(dst='{}.jpeg'.format(t),
                     src=sequence['image_files'][t],
                     rect=sequence['labels'][t],
                     frmsz=args.frmsz)


if __name__ == '__main__':
    main()
