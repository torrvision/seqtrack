import functools
import math
import numpy as np
import scipy.interpolate

import geom
import geom_np


def augment(sequence,
            rand=None,
            max_attempts=20,
            enable_global_scale=True,
            enable_global_translate=True,
            # Choose scale and translation to keep object in frame?
            keep_inside=True,
            # Limit scale to avoid upsampling/downsampling too much.
            min_scale=0.1,
            max_scale=10.,
            # Range of object sizes in image.
            min_diam=0.1,
            max_diam=0.5,
            # Motion augmentation:
            translate_kind='normal',   # normal, laplace
            translate_amount=0.0,      # Default is none.
            scale_kind='normal',       # normal, laplace
            scale_exp_amount=1.0,      # Default is none.
            keep_original_motion=True, # Add motion or over-ride?
            ):
    if rand is None:
        rand = np.random

    def attempt():
        sequence_len = len(sequence['image_files'])
        # Get object size in every frame.
        obj_rect = np.copy(sequence['labels'])
        # Fill in missing data to make the problem easier.
        obj_rect = _impute_missing(sequence['label_is_valid'], obj_rect)
        # assert obj_rect(sequence['label_is_valid']) == sequence['labels'][sequence['label_is_valid']]

        obj_min, obj_max = geom_np.rect_min_max(obj_rect)
        obj_center, obj_size = 0.5 * (obj_min + obj_max), obj_max - obj_min
        # Get typical object size (median).
        diam = np.mean(obj_size, axis=-1) # 0.5*(width+height)
        typical_diam = np.median(diam[sequence['label_is_valid']])

        # Sample random walk.
        rel_translate, scale = _sample_scale_walk(sequence_len-1, 2, rand,
            translate_kind=translate_kind, translate_amount=translate_amount,
            scale_kind=scale_kind, scale_exp_amount=scale_exp_amount)
        # Let `translate` be the position of the center of the object rectangle
        # relative to the initial position.
        translate = rel_translate * typical_diam
        if keep_original_motion:
            translate += obj_center - obj_center[0]
        # Have object displacement from initial position and scale.
        # Need to choose global scale and initial position.
        #   center = initial_center + global_scale * translate
        #   size = global_scale * obj_size * scale
        # Construct tentative rectangle without initial position and scale.
        tmp_center, tmp_size = translate, obj_size * scale
        tmp = geom_np.make_rect(tmp_center-0.5*tmp_size, tmp_center+0.5*tmp_size)

        if enable_global_scale:
            assert enable_global_translate
            # Establish size of extent.
            tmp_min, tmp_max = geom_np.rect_min_max(tmp)
            tmp_extent_min = np.amin(tmp_min[sequence['label_is_valid']], axis=0)
            tmp_extent_max = np.amax(tmp_max[sequence['label_is_valid']], axis=0)
            tmp_extent_size = tmp_extent_max - tmp_extent_min
            # Want: min_diam <= scale * typical_diam <= max_diam
            # Equivalent: min_diam / typical_diam <= scale <= max_diam / typical_diam
            scale_low = min_diam/typical_diam
            scale_high = max_diam/typical_diam
            if min_scale is not None:
                scale_low = max(scale_low, min_scale)
            if max_scale is not None:
                scale_high = min(scale_high, max_scale)
            if keep_inside:
                # Want: all(scale * extent_size <= 1)
                # Equivalent: scale <= min(1 / extent_size)
                scale_high = min(scale_high, min(1./tmp_extent_size))
            if not scale_low <= scale_high:
                # Could not find scale to satisfy requirements.
                return None
            init_scale = math.exp(rand.uniform(math.log(scale_low), math.log(scale_high)))
            translate *= init_scale
            scale *= init_scale
            # Update tentative rectangle with new scale.
            tmp_center, tmp_size = translate, obj_size * scale
            tmp = geom_np.make_rect(tmp_center-0.5*tmp_size, tmp_center+0.5*tmp_size)

        if enable_global_translate:
            tmp_min, tmp_max = geom_np.rect_min_max(tmp)
            # Establish size of extent.
            tmp_extent_min = np.amin(tmp_min[sequence['label_is_valid']], axis=0)
            tmp_extent_max = np.amax(tmp_max[sequence['label_is_valid']], axis=0)
            tmp_extent_size = tmp_extent_max - tmp_extent_min
            if keep_inside:
                # Choose translation such that object remains inside frame.
                gap = 1. - tmp_extent_size
                if not all(gap >= 0.):
                    # Could not find translation to satisfy requirements.
                    return None
                initial_center = rand.uniform(0., gap) - tmp_extent_min
            else:
                # Could pick a random location in the image?
                # We probably won't use this anyway.
                raise ValueError('no implementation')
        else:
            initial_center = obj_center[0]

        out_center = initial_center + translate
        out_size = obj_size * scale

        # Create viewport such that object center is at origin
        # and object size is multiplied by scale.
        viewport = geom_np.make_rect(obj_center, obj_center+1./scale)
        # Translate viewport such that object center appears at `out_center` in viewport.
        viewport = geom_np.rect_translate(viewport, -out_center/scale)
        # JV: Neater but less clear?
        # viewport = np.array([geom_np.unit_rect()] * sequence_len)
        # viewport = geom_np.rect_translate(viewport, -out_center)
        # viewport = geom_np.rect_mul(viewport, 1./scale)
        # viewport = geom_np.rect_translate(viewport, obj_center)

        # Sanity check.
        # TODO: Move this to a unit test? Would require to expose internals?
        out_rect = geom_np.make_rect(out_center-0.5*out_size, out_center+0.5*out_size)
        assert np.allclose(geom_np.crop_rect(obj_rect, viewport), out_rect)

        return viewport

    viewport = None
    for i in range(max_attempts):
        viewport = attempt()
        if viewport is not None:
            break
    if viewport is None:
        # print 'exhausted attempts: {}'.format(max_attempts)
        return sequence

    output = dict(sequence)
    output['viewports'] = geom_np.crop_image_viewport(sequence['viewports'], viewport)
    output['labels'] = geom_np.crop_rect(sequence['labels'], viewport)
    return output


def _sample_scale_walk(num_steps, dim, rand,
                       translate_kind,
                       translate_amount,
                       scale_kind,
                       scale_exp_amount):
    # TODO: Make zero mean?
    delta_scale = np.exp(_sample_step(num_steps, 1, rand,
        kind=scale_kind, scale=math.log(scale_exp_amount)))
    delta_position = _sample_step(num_steps, dim, rand,
        kind=translate_kind, scale=translate_amount)
    # Construct path relative to first frame.
    scale = [np.array([1.0], np.float32)]
    position = [np.zeros((dim,), np.float32)]
    for i in range(num_steps):
        position_t = position[-1] + scale[-1]*delta_position[i]
        scale_t = scale[-1] * delta_scale[i]
        position.append(position_t)
        scale.append(scale_t)
    return np.array(position), np.array(scale)


def _sample_step(num_steps, dim, rand, kind, scale):
    if kind == 'normal':
        sample = rand.normal
    elif kind == 'laplace':
        sample = rand.laplace
    else:
        raise ValueError('unknown distribution: {}'.format(kind))
    magnitude = sample(size=(num_steps, 1), scale=scale)
    direction = rand.normal(size=(num_steps, dim))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    return magnitude * direction


def _impute_missing(is_valid, rects):
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
