import cv2
import numpy as np


def image_corners(width, height):
    return np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]).reshape(-1, 1, 2)


def compute_global_canvas(ref_img, src_img, homographies):
    h_ref, w_ref = ref_img.shape[:2]
    h_src, w_src = src_img.shape[:2]

    ref_corners = image_corners(w_ref, h_ref)
    all_corners = [ref_corners]

    src_corners = image_corners(w_src, h_src)

    for H in homographies:
        warped = cv2.perspectiveTransform(src_corners, H)
        all_corners.append(warped)

    all_corners = np.concatenate(all_corners, axis=0)

    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    width = xmax - xmin
    height = ymax - ymin

    return T, width, height


def place_reference_on_global_canvas(ref_img, T, width, height):
    canvas = np.zeros((height, width, 3), dtype=ref_img.dtype)
    mask = np.zeros((height, width), dtype=np.uint8)

    tx = int(T[0, 2])
    ty = int(T[1, 2])

    h_ref, w_ref = ref_img.shape[:2]
    canvas[ty:ty + h_ref, tx:tx + w_ref] = ref_img
    mask[ty:ty + h_ref, tx:tx + w_ref] = 255

    return canvas, mask


def warp_source_to_global_canvas(src_img, H, T, width, height):
    H_total = T @ H
    warped = cv2.warpPerspective(src_img, H_total, (width, height))

    src_mask = np.ones(src_img.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(src_mask, H_total, (width, height))
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255

    return warped, warped_mask, H_total