import cv2
import numpy as np


def compute_output_canvas(ref_img, src_img, H_src_to_ref):
    h_ref, w_ref = ref_img.shape[:2]
    h_src, w_src = src_img.shape[:2]

    ref_corners = np.float32([
        [0, 0],
        [w_ref, 0],
        [w_ref, h_ref],
        [0, h_ref]
    ]).reshape(-1, 1, 2)

    src_corners = np.float32([
        [0, 0],
        [w_src, 0],
        [w_src, h_src],
        [0, h_src]
    ]).reshape(-1, 1, 2)

    warped_src_corners = cv2.perspectiveTransform(src_corners, H_src_to_ref)
    all_corners = np.concatenate([ref_corners, warped_src_corners], axis=0)

    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

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


def warp_image_and_mask(src_img, H_total, width, height):
    warped = cv2.warpPerspective(src_img, H_total, (width, height))
    mask = np.ones(src_img.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H_total, (width, height))
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255
    return warped, warped_mask


def place_reference_on_canvas(ref_img, T, width, height):
    canvas = np.zeros((height, width, 3), dtype=ref_img.dtype)
    ref_mask = np.zeros((height, width), dtype=np.uint8)

    tx = int(T[0, 2])
    ty = int(T[1, 2])
    h_ref, w_ref = ref_img.shape[:2]

    canvas[ty:ty + h_ref, tx:tx + w_ref] = ref_img
    ref_mask[ty:ty + h_ref, tx:tx + w_ref] = 255

    return canvas, ref_mask


def warp_pair(ref_img, src_img, H_src_to_ref):
    T, width, height = compute_output_canvas(ref_img, src_img, H_src_to_ref)
    H_total = T @ H_src_to_ref

    warped_src, _ = warp_image_and_mask(src_img, H_total, width, height)
    canvas, _ = place_reference_on_canvas(ref_img, T, width, height)

    stitched = warped_src.copy()
    ref_pixels = canvas.sum(axis=2) > 0
    stitched[ref_pixels] = canvas[ref_pixels]

    return stitched, warped_src, T


def warp_candidates(ref_img, src_img, homographies):
    results = []

    for idx, H in enumerate(homographies):
        T, width, height = compute_output_canvas(ref_img, src_img, H)
        H_total = T @ H

        warped_src, warped_mask = warp_image_and_mask(src_img, H_total, width, height)
        ref_canvas, ref_mask = place_reference_on_canvas(ref_img, T, width, height)

        overlay = warped_src.copy()
        ref_pixels = ref_mask > 0
        overlay[ref_pixels] = ref_canvas[ref_pixels]

        results.append(
            {
                "id": idx,
                "H": H,
                "T": T,
                "H_total": H_total,
                "width": width,
                "height": height,
                "warped_src": warped_src,
                "warped_mask": warped_mask,
                "ref_canvas": ref_canvas,
                "ref_mask": ref_mask,
                "overlay": overlay,
            }
        )

    return results