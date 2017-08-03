'''A sampler is a function that maps a dataset to an ordered collection of sequences.

The sequences can be returned as a list or the functions can be generators.
'''

import argparse
import collections
import functools
import itertools
import math
import numpy as np
import os
import progressbar
import re

import motion


class DatasetSampler:
    '''DatasetSampler is a dataset with a method of sampling sequences.
    '''

    def __init__(self, dataset, sample_frames, augment_motion):
        '''
        dataset:
            Must have
                dataset.videos
                dataset.image_file(video, frame)
                dataset.video_tracks(video)
        sample_frames:
            Function that maps (video, obj) pair to list of frames.
        augment_motion:
            Function that maps motion.Sequence to motion.Sequence.
        '''
        self.dataset         = dataset
        self._sample_frames  = sample_frames
        self._augment_motion = augment_motion

    def sample_sequence(self, video, obj):
        full_trajectory = self.dataset.video_tracks(video)[obj]
        frames = self._sample_frames(video, obj) # May contain duplicates.
        if not frames:
            return None
        # Make sequence using chosen frames.
        sequence = motion.Sequence(
            image_files=[dataset.image_file(video, t) for t in frames],
            viewports=[geom_np.crop_identity() for __ in frames],
            # Re-index from 0 to len(frames)-1.
            trajectory={
                i: full_trajectory[t] for i, t in enumerate(frames)
                if t in full_trajectory
            },
        )
        # Add augmentation.
        return self._augment_motion(sequence)


def epoch(dataset_sampler, rand, max_objects=None, max_videos=None):
    '''
    Args:
        dataset_sampler:
            Must have:
                dataset_sampler.sample_sequence(video, obj)

        dataset: Dataset object such as ILSVRC or OTB.
        rand: Random number generator (can be module `random`).
        frame_sampler: Function that samples frames for (video, object) pair.
            This function is given the video and object (instead of just the trajectory)
            in case it uses some internal cache for speed purposes.
            Note that the frame sampler must match the dataset.
        max_videos: Maximum number of videos to use, or None.
            There may still be multiple tracks per video.
        max_objects: Maximum number of objects per video, or None.

    Returns:
        Iterable collection of motion.Sequence objects.
    '''
    videos = list(dataset_sampler.dataset.videos)
    rand.shuffle(videos)
    num_videos = 0
    for video in videos:
        if max_videos is not None and num_videos >= max_videos:
            break
        video_tracks = dataset_sampler.dataset.video_tracks(video)
        objs = range(len(video_tracks))
        rand.shuffle(objs)
        num_objs = 0
        for obj in objs:
            if max_objects is not None and num_objs >= max_objects:
                break
            sequence = dataset_sampler.sample_sequence(video, obj)
            if sequence is None:
                continue
            name = '{}-{}'.format(_escape(video), obj)
            yield (name, sequence)
            num_objs += 1
        if num_objs > 0:
            num_videos += 1

def _escape(s):
    s = re.sub('/', '-', s)
    return s


def make_frame_sampler(dataset, kind, params=None):
    '''A frame sampler chooses frames within a trajectory.

    A sampler maps (video, object, rand, ntimesteps) to a list of frame indices.

    It can return an empty list or None to reject a trajectory.
    It may return less than ntimesteps+1 frames.

    A frame sampler may store information locally to make sampling more efficient.
    This is the reason that dataset and ntimesteps are provided to this function.
    '''
    params = params or {}
    if kind == 'motion':
        return make_motion_sampler(dataset, **params)
    else:
        if kind == 'full':
            f = full
        if kind == 'repeat_one':
            f = repeat_one
        elif kind == 'all_with_label':
            f = all_with_label
        elif kind == 'random_with_label':
            f = random_with_label
        elif kind == 'regular':
            f = regular
        elif kind == 'freq_range_fit':
            f = freq_range_fit
        else:
            raise ValueError('unknown kind of frame sampler: {}'.format(kind))
        return functools.partial(f, dataset=dataset, **params)


def full(video, obj, rand, dataset, ntimesteps):
    '''ntimesteps is ignored'''
    times = dataset.video_tracks(video)[obj].keys()
    t_first = min(times)
    t_last = max(times)
    return range(t_first, t_last+1)

def repeat_one(video, obj, rand, dataset, ntimesteps):
    '''Repeat a single frame. Useful for single-frame videos (images).'''
    times = dataset.video_tracks(video)[obj].keys()
    t = rand.choice(times)
    return [t for __ in range(ntimesteps+1)]

def all_with_label(video, obj, rand, dataset, ntimesteps):
    '''ntimesteps is ignored'''
    times = sorted(dataset.video_tracks(video)[obj].keys())
    return times

def random_with_label(video, obj, rand, dataset, ntimesteps):
    times = dataset.video_tracks(video)[obj].keys()
    return rand.sample(times, min(ntimesteps+1, len(times)))

def regular(video, obj, rand, dataset, ntimesteps, freq):
    times = dataset.video_tracks(video)[obj].keys()
    t0 = rand.choice(times)
    subset = range(t0, dataset.video_length[video], freq)
    return subset[:ntimesteps+1]

def freq_range_fit(video, obj, rand, dataset, ntimesteps,
        min_freq, max_freq):
    times = sorted(dataset.video_tracks(video)[obj].keys())
    video_len = dataset.video_length[video]
    # Choose frames:
    #   a, round(a+freq), round(a+2*freq), round(a+3*freq), ...
    # Therefore, for frames [0, ..., ntimesteps], we need:
    #   a + ntimesteps*freq <= video_len - 1
    # The smallest possible value of a is times[0]
    #   times[0] + ntimesteps*freq <= video_len - 1
    #   ntimesteps*freq <= video_len - 1 - times[0]
    #   freq <= (video_len - 1 - times[0]) / ntimesteps
    u = min_freq
    v = min(max_freq, float((video_len - 1) - times[0]) / ntimesteps)
    if not u <= v:
        return None
    f = math.exp(rand.uniform(math.log(u), math.log(v)))
    # Let n = ntimesteps*f.
    n = int(round(ntimesteps * f))
    # Choose first frame such that all frames are present.
    a = rand.choice([a for a in times if a + n <= video_len - 1])
    ideal_times = a + f * np.arange(0, ntimesteps+1)
    # Snap to nearest time with a label.
    return _snap(ideal_times, times)


def _snap(ideal, possible):
    inds = np.round(np.interp(ideal, possible, range(len(possible)))).astype(int)
    return (np.array(possible)[inds]).tolist()


def make_motion_sampler(dataset, **kwargs):
    sampler = Motion(dataset, **kwargs)
    return sampler.sample

class Motion:
    # Could be implemented as a function with yield.

    def __init__(self,
                 dataset,
                 min_speed,
                 max_speed):
                 # min_original_speed=0.0):
        self.dataset = dataset
        self.min_speed = min_speed
        self.max_speed = max_speed
        # self.min_original_speed = min_original_speed

        self.cdfs = {}
        for video in self.dataset.videos:
            self.cdfs[video] = [
                motion_cdf(track, relative=True)
                for track in self.dataset.video_tracks(video)
            ]

    def sample(self, video, obj_ind, rand, ntimesteps):
        # video = rand.choice(self.video_subset)
        # obj_ind = rand.choice(self.object_subset[video])
        cdf = self.cdfs[video][obj_ind]
        _, full_path_length = cdf[-1]
        # min_original_path_length = self.min_original_speed * ntimesteps
        # assert full_path_length >= min_original_path_length
        # Limit example path length to maximum possible.
        # (Do this before choosing length to avoid always using same path.)
        min_sample_path_length = float(self.min_speed) * ntimesteps
        if self.max_speed is None:
            max_sample_path_length = float('inf')
        else:
            max_sample_path_length = float(self.max_speed) * ntimesteps
        if full_path_length < min_sample_path_length:
            return None
        sample_path_length = rand.uniform(
            min_sample_path_length,
            min(max_sample_path_length, full_path_length),
        )
        d0 = rand.uniform(0, full_path_length - sample_path_length)
        d1 = d0 + sample_path_length
        sample_dists = np.linspace(d0, d1, ntimesteps+1)
        cdf_dists = [dist for t, dist in cdf]
        inds = np.round(np.interp(sample_dists, cdf_dists, range(len(cdf_dists)))).astype(int)
        times = [t for t, dist in [cdf[ind] for ind in inds]]
        return times


def motion_cdf(track, relative=True, epsilon=1e-2):
    # track is a dictionary that maps frame index to rectangle
    track = sorted(track.items())
    times = [t for t, rect in track]
    rects = np.array([rect for t, rect in track])
    abs_dist = 0.5 * np.sum(np.abs(np.diff(rects, axis=0)), axis=1)
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
