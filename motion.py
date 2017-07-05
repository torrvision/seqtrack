import math
import numpy as np
import random

import geom
import geom_np

def add_gaussian_random_walk(trajectory,
                             sequence_len,
                             sigma_translate=0.5,
                             sigma_scale=1.1,
                             min_diameter=0.1,
                             max_diameter=0.5,
                             max_attempts=20):
    '''
    Args:
        trajectory -- Dictionary that maps frame number to rectangle.
            Rectangle is in normalized co-ordinates in (0, 1).
            TODO: Incorporate aspect ratio of image.

    Returns:
        A sequence of windows such that the trajectory is augmented in those windows.
    '''

    eps = 0.01

    times = trajectory.keys()
    rects = np.array(trajectory.values())
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
        if num_attempts > max_attempts:
            print 'random walk: exceeded maximum number of attempts'
            return np.array([[0.0, 0.0, 1.0, 1.0]] * sequence_len)
        # Sample size of object.
        # TODO: Maximum scale.
        base_diameter = math.exp(random.uniform(math.log(min_diameter), math.log(max_diameter)))
        rel_center = sigma_translate * sample_random_walk(sequence_len, dim=2)
        scale = np.exp(math.log(sigma_scale) * sample_random_walk(sequence_len, dim=1))
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
        ok = np.all(gap > 0.0)
        num_attempts += 1

    # Choose global translation of track.
    offset = -region_extent_min + np.random.uniform(size=(2,)) * gap
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
    window = geom_np.crop_solve(orig_region, region)
    # geom_np.crop_rect(orig_region, window) == region
    return window


def sample_random_walk(n, dim):
    # TODO: Use Python random number generator?
    steps = np.random.normal(size=(n-1, dim))
    path = np.concatenate((
        np.zeros((1, dim)),
        np.cumsum(steps, axis=0),
    ))
    return path
