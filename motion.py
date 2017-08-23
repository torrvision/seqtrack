import functools
import math
import numpy as np
import scipy.interpolate

import geom
import geom_np


def make_motion_augmenter(kind, rand, **kwargs):
    '''
    Returns:
        Function that maps (is_valid, labels) to viewports.
        Function that maps sequence to sequence, modifying viewports.
    '''
    if kind == 'none':
        return no_augment
    elif kind == 'add_gaussian_random_walk':
        return functools.partial(add_gaussian_random_walk, rand=rand, **kwargs)
    elif kind == 'gaussian_random_walk':
        return functools.partial(gaussian_random_walk, rand=rand, **kwargs)
    else:
        raise ValueError('unknown motion augmentation: {}'.format(kind))


def no_augment(is_valid, rects):
    return np.array([geom_np.rect_identity()] * len(rects))


def gaussian_random_walk(is_valid, rects, rand,
                         translate_kind='laplace',
                         translate_amount=0.5,
                         scale_kind='laplace',
                         scale_amount=1.1,
                         min_diameter=0.1,
                         max_diameter=0.5,
                         min_scale=0.5,
                         max_scale=4,
                         max_attempts=20):
    '''
    Args:
        rects: List of rectangles, or None if object is not present.
        rand: Numpy random object.

    Returns:
        A list of viewports for the object.
    '''

    EPSILON = 0.01

    orig_rect = rects
    # Fill missing elements.
    orig_rect = _interpolate_missing(is_valid, orig_rect)

    orig_min, orig_max = geom_np.rect_min_max(orig_rect)
    orig_size = np.maximum(0.0, orig_max - orig_min)
    orig_diameter = np.exp(np.mean(np.log(np.maximum(EPSILON, orig_size)), axis=-1))
    orig_mean_diameter = np.exp(np.mean(np.log(orig_diameter)))

    ok = False
    num_attempts = 0
    while not ok:
        if num_attempts > max_attempts:
            return None

        # Sample (average) size of object.
        base_diameter = math.exp(rand.uniform(math.log(min_diameter), math.log(max_diameter)))
        # Impose limits on scale.
        # This may override min_diameter and max_diameter!
        # TODO: Consider aspect ratio here?
        abs_scale = base_diameter / orig_mean_diameter
        abs_scale = np.minimum(max_scale, np.maximum(min_scale, abs_scale))
        base_diameter = abs_scale * orig_mean_diameter

        position_rel, scale = _sample_scale_walk(len(orig_rect), dim=2, rand=rand,
            translate_kind=translate_kind,
            translate_amount=translate_amount,
            scale_kind=scale_kind,
            scale_amount=scale_amount)

        # Now move from object co-ordinates to world co-ordinates.
        position = base_diameter * position_rel
        diameter = base_diameter * scale
        # Preserve original size variation.
        size = orig_size / orig_mean_diameter * diameter

        out_min = position - 0.5*size
        out_max = position + 0.5*size
        out_extent_min = np.amin(out_min, axis=0)
        out_extent_max = np.amax(out_max, axis=0)
        out_extent_size = out_extent_max - out_extent_min

        gap = 1.0 - out_extent_size
        ok = np.all(gap > 0.0)
        num_attempts += 1

    # Choose global translation of track.
    offset = -out_extent_min + rand.uniform(size=(2,)) * gap
    out_rect = geom_np.make_rect(out_min + offset, out_max + offset)

    viewport = geom_np.crop_solve(orig_rect, out_rect)
    # geom_np.crop_rect(orig_rect, viewport) == out_rect
    return viewport


def add_gaussian_random_walk(is_valid, rects, rand,
                             translate_kind='laplace',
                             translate_amount=0.5,
                             scale_kind='laplace',
                             scale_amount=1.1,
                             min_diameter=0.1,
                             max_diameter=0.5,
                             min_scale=0.5,
                             max_scale=4,
                             max_attempts=20):
    EPSILON = 0.01

    orig_rect = rects
    orig_rect_valid = orig_rect[is_valid]
    orig_min, orig_max = geom_np.rect_min_max(orig_rect_valid)
    orig_size = np.maximum(0.0, orig_max - orig_min)
    orig_diameter = np.exp(np.mean(np.log(np.maximum(EPSILON, orig_size)), axis=-1))
    orig_mean_diameter = np.exp(np.mean(np.log(orig_diameter)))
    # Treat the bounding box of the object's path as a static object in an image.
    # Scaling preserves the center of this "object"?
    orig_region_min = np.amin(orig_min, axis=0)
    orig_region_max = np.amax(orig_max, axis=0)
    orig_region_size = orig_region_max - orig_region_min
    orig_region = geom_np.make_rect(orig_region_min, orig_region_max)

    # Now we need to find a random walk
    # such that the extent does not go out of the image frame.
    ok = False
    num_attempts = 0
    while not ok:
        if num_attempts > max_attempts:
            return None
        # Sample (average) size of object.
        base_diameter = math.exp(rand.uniform(math.log(min_diameter), math.log(max_diameter)))
        # Do not want to shrink or grow the image too much.
        # This may override min_diameter and max_diameter!
        # TODO: Consider aspect ratio here?
        abs_scale = base_diameter / orig_mean_diameter
        abs_scale = np.minimum(max_scale, np.maximum(min_scale, abs_scale))
        base_diameter = abs_scale * orig_mean_diameter

        position_rel, scale = _sample_scale_walk(len(orig_rect), dim=2, rand=rand,
            translate_kind=translate_kind,
            translate_amount=translate_amount,
            scale_kind=scale_kind,
            scale_amount=scale_amount)

        # Check whether this object can fit in the frame with this path.
        # TODO: Use aspect ratio.
        position = base_diameter * position_rel
        # Size of original extent after scaling in each frame.
        # (Object diameter in each frame is simply base_diameter * scale.)
        region_size = scale * abs_scale * orig_region_size
        region_min = position - 0.5 * region_size
        region_max = position + 0.5 * region_size
        # Ensure that region will fit inside image frame.
        region_extent_min = np.amin(region_min, axis=0)
        region_extent_max = np.amax(region_max, axis=0)
        region_extent_size = region_extent_max - region_extent_min
        gap = 1.0 - region_extent_size
        ok = np.all(gap > 0.0)
        num_attempts += 1

    # Choose global translation of track.
    offset = -region_extent_min + rand.uniform(size=(2,)) * gap
    region = geom_np.make_rect(region_min + offset, region_max + offset)

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


def _sample_step(n, dim, rand, kind='gaussian', scale=1.0):
    if kind == 'gaussian':
        sample = rand.normal
    elif kind == 'laplace':
        sample = rand.laplace
    else:
        raise ValueError('unknown distribution: {}'.format(kind))
    magnitude = sample(size=(n, 1), scale=scale)
    direction = rand.normal(size=(n, dim))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return magnitude * direction

def _sample_walk(n, dim, rand, kind='gaussian', scale=1.0):
    return _cumsum(_sample_step(n-1, dim, rand, kind, scale))

def _cumsum(steps):
    dim = steps.shape[-1]
    return np.concatenate((
        np.zeros((1, dim)),
        np.cumsum(steps, axis=0),
    ))

def _interpolate_missing(is_valid, rects):
    is_valid, rects = _extrapolate_missing(is_valid, rects)
    n = len(rects)
    t = np.arange(n, dtype=np.float32)
    t_valid = t[is_valid]
    rects_valid = rects[is_valid]
    # TODO: Extrapolate is not good.
    f = scipy.interpolate.interp1d(t_valid, rects_valid, axis=0, kind='linear', bounds_error=True)
    return f(t)

def _extrapolate_missing(is_valid, rects):
    n = len(rects)
    t = np.arange(n)
    t_valid = t[is_valid]
    a = t_valid[0]
    b = t_valid[-1] + 1
    rects = np.concatenate((
        np.tile(rects[a], (a, 1)),
        rects[a:b],
        np.tile(rects[b-1], (n - b, 1))), axis=0)
    is_valid = np.array((
        [True] * a +
        list(is_valid[a:b]) +
        [True] * (n - b)))
    return is_valid, rects

def _sample_scale_walk(n, dim, rand,
                       translate_kind='laplace',
                       translate_amount=0.5,
                       scale_kind='laplace',
                       scale_amount=1.1):
    delta_position = _sample_step(n-1, dim, rand,
        kind=translate_kind, scale=translate_amount)
    delta_scale = np.exp(_sample_step(n-1, 1, rand,
        kind=scale_kind, scale=math.log(scale_amount)))
    # Construct path relative to first frame.
    scale = [np.array([1.0], np.float32)]
    position = [np.zeros((dim,), np.float32)]
    for i in range(n-1):
        position_t = position[-1] + scale[-1]*delta_position[i]
        scale_t = scale[-1] * delta_scale[i]
        position.append(position_t)
        scale.append(scale_t)
    return np.array(position), np.array(scale)

# def _combine_translate_scale(delta_position, delta_scale):
#     # Construct path relative to first frame.
#     n = len(delta_position)
#     scale = [1.0]
#     position = [0.0]
#     for t in range(1, n):
#         position_t = position[-1] + scale[-1]*delta_position[t]
#         scale_t = scale[-1] * delta_scale[t]
#         position.append(position_t)
#         scale.append(scale_t)
#     return np.array(position), np.array(scale)
