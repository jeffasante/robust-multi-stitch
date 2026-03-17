import cv2
import numpy as np


def to_gray_float(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def local_absdiff_cost(ref_canvas, warped_img, valid_mask, blur_ksize=9):
    ref_gray = to_gray_float(ref_canvas)
    warped_gray = to_gray_float(warped_img)

    diff = np.abs(ref_gray - warped_gray)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)

    invalid = (valid_mask == 0)
    diff[invalid] = 1.0

    return diff


def normalize_map(x, eps=1e-8):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def build_unary_cost_volume(
    ref_canvas,
    ref_mask,
    warped_candidates,
    motion_conf_maps,
    alpha_motion=0.7,
    alpha_photo=0.3,
    invalid_cost=1.0,
    ref_bias=0.35,
):
    h, w = ref_mask.shape
    num_labels = 1 + len(warped_candidates)

    costs = np.full((num_labels, h, w), invalid_cost, dtype=np.float32)

    overlap_any = np.zeros((h, w), dtype=bool)
    for cand in warped_candidates:
        valid_support = cand.get("support_mask", cand["warped_mask"])
        overlap_any |= (valid_support > 0)

    # Reference label:
    # - invalid outside reference mask
    # - small positive bias inside reference mask so candidates can win
    ref_cost = np.full((h, w), invalid_cost, dtype=np.float32)
    ref_cost[ref_mask > 0] = ref_bias

    # If only the reference exists at a pixel, let it win cheaply
    ref_only = (ref_mask > 0) & (~overlap_any)
    ref_cost[ref_only] = 0.0

    costs[0] = ref_cost

    for i, (cand, motion_conf) in enumerate(zip(warped_candidates, motion_conf_maps), start=1):
        photo_cost = local_absdiff_cost(ref_canvas, cand["warped_img"], cand["warped_mask"])
        photo_cost = normalize_map(photo_cost)

        motion_cost = 1.0 - motion_conf
        motion_cost = normalize_map(motion_cost)

        total = alpha_motion * motion_cost + alpha_photo * photo_cost
        valid_support = cand.get("support_mask", cand["warped_mask"])
        total[valid_support == 0] = invalid_cost

        costs[i] = total

    return costs