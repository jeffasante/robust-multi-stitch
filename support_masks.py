import cv2
import numpy as np


def largest_connected_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + np.argmax(areas)

    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def build_candidate_support_mask(
    motion_conf,
    warped_mask,
    conf_thresh=0.08,
    min_area=600,
    dilate_iters=4,
):
    support = ((motion_conf >= conf_thresh) & (warped_mask > 0)).astype(np.uint8) * 255

    if support.sum() == 0:
        return support

    kernel = np.ones((5, 5), np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, kernel)
    support = cv2.dilate(support, kernel, iterations=dilate_iters)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    cleaned = np.zeros_like(support)

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == lab] = 255

    if cleaned.sum() == 0:
        return cleaned

    return cleaned
