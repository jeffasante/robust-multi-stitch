import numpy as np


def bounding_box_from_mask(mask, pad=10):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = max(0, xs.min() - pad)
    y0 = max(0, ys.min() - pad)
    x1 = xs.max() + pad + 1
    y1 = ys.max() + pad + 1
    return x0, y0, x1, y1


def crop_array(arr, bbox):
    x0, y0, x1, y1 = bbox
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    return arr[y0:y1, x0:x1, ...]


def paste_array(dst, src, bbox):
    x0, y0, x1, y1 = bbox
    dst[y0:y1, x0:x1] = src
    return dst
