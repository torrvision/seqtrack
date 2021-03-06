from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

EPSILON = 1e-3


def make_rect(min_pt, max_pt):
    x_min, y_min = np.split(np.asarray(min_pt), 2, axis=-1)
    x_max, y_max = np.split(np.asarray(max_pt), 2, axis=-1)
    rect = np.concatenate((x_min, y_min, x_max, y_max), axis=-1)
    return rect


def rect_min_max(rect):
    x_min, y_min, x_max, y_max = np.split(np.asarray(rect), 4, axis=-1)
    min_pt = np.concatenate((x_min, y_min), axis=-1)
    max_pt = np.concatenate((x_max, y_max), axis=-1)
    return min_pt, max_pt


def rect_center_size(rect):
    min_pt, max_pt = rect_min_max(rect)
    return 0.5 * (min_pt + max_pt), max_pt - min_pt


def rect_size(rect):
    min_pt, max_pt = rect_min_max(rect)
    return max_pt - min_pt


def crop_solve(original, result):
    '''Finds window such that result = crop(original, window).'''
    original_min, original_max = rect_min_max(original)
    result_min, result_max = rect_min_max(result)
    # If original is x, result is r and window is w, then
    #   (x_max - x_min) / (w_max - w_min) = r_max - r_min
    original_size = original_max - original_min
    result_size = result_max - result_min
    window_size = original_size / result_size
    # and
    #   (x_min - w_min) / (w_max - w_min) = r_min
    #   x_min - w_min = r_min * (w_max - w_min)
    #   x_min - r_min * (w_max - w_min) = w_min
    window_min = original_min - result_min * window_size
    window_max = window_min + window_size
    window = make_rect(window_min, window_max)
    return window


def crop_rect(rects, window_rect):
    window_min, window_max = rect_min_max(window_rect)
    window_size = window_max - window_min
    window_size = np.sign(window_size) * np.maximum(np.abs(window_size), EPSILON)
    rects_min, rects_max = rect_min_max(rects)
    out_min = (rects_min - window_min) / window_size
    out_max = (rects_max - window_min) / window_size
    return make_rect(out_min, out_max)


def crop_inverse(rect):
    rect_min, rect_max = rect_min_max(rect)
    # TODO: Support reversed rectangle.
    rect_size = np.maximum(np.abs(rect_max - rect_min), EPSILON)
    inv_min = -rect_min / rect_size
    # u_min = -x_min / x_size
    # v_min = -y_min / y_size
    inv_max = (1 - rect_min) / rect_size
    # inv_max = inv_min + 1 / rect_size
    # u_max = u_min + 1 / x_size
    # v_max = v_min + 1 / y_size
    return make_rect(inv_min, inv_max)


def unit_rect():
    min_pt = np.array([0.0, 0.0], dtype=np.float32)
    max_pt = np.array([1.0, 1.0], dtype=np.float32)
    return make_rect(min_pt, max_pt)


def make_rect_center_size(center, size):
    return make_rect(center - 0.5 * size, center + 0.5 * size)


def rect_iou(a_rect, b_rect):
    intersect_min, intersect_max = rect_min_max(rect_intersect(a_rect, b_rect))
    intersect_vol = np.prod(np.maximum(0.0, intersect_max - intersect_min), axis=-1)
    a_min, a_max = rect_min_max(a_rect)
    b_min, b_max = rect_min_max(b_rect)
    a_vol = np.prod(np.maximum(0.0, a_max - a_min), axis=-1)
    b_vol = np.prod(np.maximum(0.0, b_max - b_min), axis=-1)
    union_vol = a_vol + b_vol - intersect_vol
    iou = np.where(intersect_vol == 0.0, 0.0, intersect_vol / union_vol)
    iou = np.where(np.isnan(iou), 0.0, iou)
    return iou


def rect_intersect(a_rect, b_rect):
    # Assumes that rectangles are valid (min <= max).
    a_min, a_max = rect_min_max(a_rect)
    b_min, b_max = rect_min_max(b_rect)
    intersect_min = np.maximum(a_min, b_min)
    intersect_max = np.minimum(a_max, b_max)
    return make_rect(intersect_min, intersect_max)


def rect_center_dist(a_rect, b_rect):
    a_min, a_max = rect_min_max(a_rect)
    b_min, b_max = rect_min_max(b_rect)
    a_center = 0.5 * (a_min + a_max)
    b_center = 0.5 * (b_min + b_max)
    dist = np.linalg.norm(a_center - b_center, axis=-1)
    # If the rectangles were nan, return infinite distance.
    dist = np.where(np.isnan(dist), float('inf'), dist)
    return dist


def rect_translate(rect, delta):
    min_pt, max_pt = rect_min_max(rect)
    return make_rect(min_pt + delta, max_pt + delta)


def rect_mul(rect, size):
    min_pt, max_pt = rect_min_max(rect)
    return make_rect(size * min_pt, size * max_pt)


def grow_rect(scale, rect):
    min_pt, max_pt = rect_min_max(rect)
    center, size = 0.5 * (min_pt + max_pt), max_pt - min_pt
    size *= scale
    return make_rect(center - 0.5 * size, center + 0.5 * size)
