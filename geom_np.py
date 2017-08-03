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

def crop_rect(rects, window_rect):
    '''Gives the rectangle relative to the window.'''
    eps = 0.01
    window_min, window_max = rect_min_max(window_rect)
    window_size = window_max - window_min
    window_size = np.sign(window_size) * (np.abs(window_size) + eps)
    rects_min, rects_max = rect_min_max(rects)
    out_min = (rects_min - window_min) / window_size
    out_max = (rects_max - window_min) / window_size
    return make_rect(out_min, out_max)

'''
Cropping is an operation that maps two rectangles to a rectangle.

If v = crop(u, w), then:
    v_min = (u_min - w_min) / w_size
    v_max = (u_max - w_min) / w_size
We will write this as:
    v = (u - w_min) / w_size
Also note that:
    v_size = u_size / w_size

Is this operation associative?
Let p = crop(crop(a, b), c) with x = crop(a, b)
and q = crop(a, crop(b, c)) with y = crop(b, c).
p_size = x_size / c_size = (a_size / b_size) / c_size = a_size / (b_size c_size)
q_size = a_size / y_size = a_size / (b_size / c_size) = (a_size c_size) / b_size
Therefore p_size = q_size when c_size = 1 or a_size = 0, but not in general.
This operation does not define a semi-group.

There is a right-identity e such that
    crop(u, e) = u
    u = (u - e_min) / e_size
This can be satisfied with e_min = 0, e_size = 1.
However, there does not seem to be a left-identity e such that
    crop(e, u) = u
    u = (e - u_min) / u_size

Consider inverse problems.

Find u such that v = crop(u, w).
First:
    v_size = u_size / w_size
    u_size = v_size * w_size
Then:
    v = (u - w_min) / w_size
    u = v * w_size + w_min
      = (v + w_min / w_size) * w_size
      = (v - (-w_min) / w_size) / (1 / w_size)
This can be represented:
    u = crop(v, x)
where:
    x_size = 1 / w_size
    x_min = -w_min / w_size
    x_max = -w_min / w_size + 1 / w_size
          = (1 - w_min) / w_size
and we say x = inv(w).
Therefore we have a right identity and a right inverse.

Find w such that v = crop(u, w).
First:
    v_size = u_size / w_size
    w_size = u_size / v_size
Then:
    v = (u - w_min) / w_size
    v * w_size = u - w_min
    w_min = u - v * w_size (for any corresponding pair u, v)
          = u - v * w_size
And also:
    w_max = w_min + w_size
          = u + (1 - v) * w_size
'''

def crop_identity():
    '''Returns the window e such that crop(u, e) = e.'''
    return np.array([0.0, 0.0, 1.0, 1.0])

def crop_inverse(rect):
    '''Returns the window b such that crop(a, b) = e.

    To find v such that u = crop(v, w), compute v = crop(u, inv(w)).
    '''
    eps = 0.01
    rect_min, rect_max = rect_min_max(rect)
    # TODO: Support reversed rectangle.
    rect_size = np.abs(rect_max - rect_min) + eps
    inv_min = -rect_min / rect_size
    # u_min = -x_min / x_size
    # v_min = -y_min / y_size
    inv_max = (1 - rect_min) / rect_size
    # inv_max = inv_min + 1 / rect_size
    # u_max = u_min + 1 / x_size
    # v_max = v_min + 1 / y_size
    return make_rect(inv_min, inv_max)

def crop_solve_window(original, result):
    '''Finds window such that result = crop(original, window).
    
    That is, if v = crop(u, w), then w = crop_solve_window(u, v).
    '''
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
