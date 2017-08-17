import numpy as np

def rect_min_max(rect):
    x_min, y_min, x_max, y_max = np.split(rect, 4, axis=-1)
    min_pt = np.concatenate((x_min, y_min), axis=-1)
    max_pt = np.concatenate((x_max, y_max), axis=-1)
    return min_pt, max_pt

def make_rect(min_pt, max_pt):
    x_min, y_min = np.split(min_pt, 2, axis=-1)
    x_max, y_max = np.split(max_pt, 2, axis=-1)
    rect = np.concatenate((x_min, y_min, x_max, y_max), axis=-1)
    return rect

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
    eps = 0.01
    window_min, window_max = rect_min_max(window_rect)
    window_size = window_max - window_min
    window_size = np.sign(window_size) * (np.abs(window_size) + eps)
    rects_min, rects_max = rect_min_max(rects)
    out_min = (rects_min - window_min) / window_size
    out_max = (rects_max - window_min) / window_size
    return make_rect(out_min, out_max)

def rect_identity():
    return [0.0, 0.0, 1.0, 1.0]
