import functools
import math
import numpy as np
import random

import geom_np


class Sequence:

    def __init__(self, image_files, viewports, trajectory):
        '''
        image_files -- List of image files.
        viewports -- List of rectangles.
            The actual image under consideration is the sub-image of this rectangle.
        trajectory -- Dictionary that maps a subset of frames to a rectangle.
            These rectangles are relative to the viewport, not the whole image.
        '''
        self.image_files = image_files
        self.viewports   = viewports
        self.trajectory  = trajectory

    def crop(self, windows):
        '''Crops the sequence according to a sequence of windows.

        The windows are relative to the viewport.
        '''
        return Sequence(
            image_files=self.image_files,
            # The new viewport (relative to the whole image) is such that
            #   window = crop(new_viewport, viewport)
            # and therefore
            #   crop(window, inv_viewport) = new_viewport
            viewports=[
                geom_np.crop_rect(window, geom_np.crop_inverse(viewport))
                for viewport, window in zip(self.viewports, windows)
            ],
            # Within the frame of the viewport, crop the rectangle.
            trajectory={
                t: geom_np.crop_rect(rect, windows[t])
                for t, rect in self.trajectory.items()
            },
        )

    # TODO: Maybe this should be external to this package?
    def to_feed_dict(self):
        n = len(self.image_files)
        rects = [self.trajectory.get(t, _invalid_rect()) for t in range(n)]
        rect_is_present = [t in self.trajectory for t in range(n)]
        return {
            'image_files':    self.image_files,
            'viewports':      self.viewports,
            'labels':         rects,
            'label_is_valid': rect_is_present,
        }


# def make_augment_func(kind, params):
#     '''
#     Returns:
#         Function that maps Sequence to Sequence.
#     '''
#     return functools.partial(get_augment_func(kind), **params)

def augment(sequence, rand, kind, params):
    '''
    Returns:
        Function that maps Sequence to Sequence.
    '''
    return get_augment_func(kind)(sequence, rand, **params)


def get_augment_func(kind):
    '''
    Returns:
        Function that maps Sequence to Sequence.
    '''
    if kind == 'none':
        return lambda sequence: sequence
    elif kind == 'static':
        return augment_static
    elif kind == 'add_random_walk':
        return add_random_walk
    else:
        raise ValueError('unknown kind of motion augmentation: {}'.format(kind))


def augment_static(sequence,
                   rand,
                   min_diameter=0.1,
                   max_diameter=0.5):
                   # max_attempts=20):
    eps = 0.01

    times = sequence.trajectory.keys()
    rects = np.array(sequence.trajectory.values())
    rects_min, rects_max = geom_np.rect_min_max(rects)
    rects_size = np.maximum(0.0, rects_max - rects_min)
    orig_diameter = np.exp(np.mean(np.log(rects_size + eps), axis=-1))
    orig_mean_diameter = np.exp(np.mean(np.log(orig_diameter)))
    # Treat the bounding box of the object's path as a static object in an image.
    orig_region_min = np.amin(rects_min, axis=0)
    orig_region_max = np.amax(rects_max, axis=0)
    orig_region_size = orig_region_max - orig_region_min
    orig_region = geom_np.make_rect(orig_region_min, orig_region_max)

    # Require:
    #   min_diameter < scale * orig_mean_diameter < max_diameter
    min_scale = np.amax(min_diameter / orig_mean_diameter)
    max_scale = np.amin(max_diameter / orig_mean_diameter)
    # Find scale such that region does not exceed image size.
    #   scale * orig_region < 1.0
    max_scale = min(max_scale, np.amin(1.0 / orig_region))
    if not min_scale <= max_scale:
        return None
    # Sample a new scale.
    scale = math.exp(rand.uniform(math.log(min_scale), math.log(max_scale)))
    # Size of original extent after scaling.
    region_size = orig_region_size * scale

    # Now that size of region is known, choose its position.
    gap = 1.0 - region_size
    assert np.all(gap >= 0.0)
    # Choose global translation of track.
    offset = rand.uniform(size=(2,)) * gap
    region_min = offset
    region_max = offset + region_size
    region = geom_np.make_rect(region_min, region_max)

    # Find window that gives this result.
    orig_region = np.expand_dims(orig_region, 0)
    window = geom_np.crop_solve_window(orig_region, region)
    # geom_np.crop_rect(orig_region, window) == region
    # Use same window in every frame.
    # window = np.array([[window]] * len(sequence.image_files))
    import pdb ; pdb.set_trace()
    return sequence.crop(window)


def add_random_walk(sequence,
                    rand,
                    translate_path_args=None,
                    exp_sigma_scale=1.0,
                    min_diameter=0.1,
                    max_diameter=0.5,
                    max_attempts=20):
    '''
    Args:
        translate_path_args:
            Keyword arguments to `sample_random_path`.
            For example, {'kind': 'laplace', 'params': {'scale': 0.1}}
    '''

    eps = 0.01

    times = sequence.trajectory.keys()
    rects = np.array(sequence.trajectory.values())
    rects_min, rects_max = geom_np.rect_min_max(rects)
    rects_size = np.maximum(0.0, rects_max - rects_min)
    orig_diameter = np.exp(np.mean(np.log(rects_size + eps), axis=-1))
    orig_mean_diameter = np.exp(np.mean(np.log(orig_diameter)))
    # Treat the bounding box of the object's path as a static object in an image.
    orig_region_min = np.amin(rects_min, axis=0)
    orig_region_max = np.amax(rects_max, axis=0)
    orig_region_size = orig_region_max - orig_region_min
    orig_region = geom_np.make_rect(orig_region_min, orig_region_max)

    # Now we need to find a random walk
    # such that the extent does not go out of the image frame.
    ok = False
    num_attempts = 0
    while not ok:
        if num_attempts >= max_attempts:
            # Fail.
            print 'random walk: exceeded maximum number of attempts'
            return None
        # Sample size of object.
        # TODO: Maximum scale.
        base_diameter = math.exp(random.uniform(math.log(min_diameter), math.log(max_diameter)))
        rel_center = sample_random_path(sequence.num_frames, dim=2, rand=rand, **(translate_path_args or {}))
        scale = np.exp(random_walk_normal(sequence.num_frames, dim=1, rand=rand,
                                          scale=abs(math.log(exp_sigma_scale))))
        # Check whether this object can fit in the frame with this path.
        center = base_diameter * rel_center
        # Size of original extent after scaling in each frame.
        # (Object diameter in each frame is simply base_diameter * scale.)
        region_size = orig_region_size / orig_mean_diameter * base_diameter * scale
        region_min = center - 0.5 * region_size
        region_max = center + 0.5 * region_size
        # Ensure that region will fit inside image frame.
        region_extent_min = np.amin(region_min, axis=0)
        region_extent_max = np.amax(region_max, axis=0)
        region_extent_size = region_extent_max - region_extent_min
        gap = 1.0 - region_extent_size
        ok = np.all(gap >= 0.0)
        num_attempts += 1

    # Choose global translation of track.
    offset = -region_extent_min + rand.uniform(size=(2,)) * gap
    region_min += offset
    region_max += offset
    region = geom_np.make_rect(region_min, region_max)

    # We have generated a trajectory for the extent of the original motion.
    # The problem is now to find the sequence of windows which gives that trajectory,
    # treating the original extent as a static object.

    # Turn sampled trajectory into the window that gives the trajectory.
    # That is, find the window such that:
    #   crop(input, window) = output

    # If c = crop(a, b), then a = crop(c, inv(b)).
    #   c = a * b
    #   c * inv(b) = (a * b) * inv(b)
    #   c * inv(b) = a * (b * inv(b)) # Is there associativity?
    #   c * inv(b) = a * e
    #   a = c * inv(b)
    # What about if we know a and c, and want to find b?

    orig_region = np.expand_dims(orig_region, 0)
    window = geom_np.crop_solve_window(orig_region, region)
    # geom_np.crop_rect(orig_region, window) == region
    return sequence.crop(window)


def random_walk_normal(n, dim, rand, scale=1.0):
    # TODO: Use Python random number generator?
    steps = rand.normal(scale=scale, size=(n-1, dim))
    path = np.concatenate((
        np.zeros((1, dim)),
        np.cumsum(steps, axis=0),
    ))
    return path

def random_walk_laplace(n, dim, rand, scale=1.0):
    # TODO: Use Python random number generator?
    normal_steps = rand.normal(size=(n-1, dim))
    direction = normal_steps / np.linalg.norm(normal_steps, axis=-1)
    radius = rand.laplace(scale=scale, size=(n-1, 1))
    steps = radius * direction
    path = np.concatenate((
        np.zeros((1, dim)),
        np.cumsum(steps, axis=0),
    ))
    return path


def sample_random_path(n, dim, rand, kind, params=None):
    return get_random_walk_func(kind)(n, dim, rand, **(params or {}))

def get_random_walk_func(kind):
    if kind == 'normal':
        return random_walk_normal
    elif kind == 'laplace':
        return random_walk_laplace
    else:
        raise ValueError('unknown kind of random walk: {}'.format(kind))
