import numpy as np


def compute_overlap_region(ref_mask, warped_candidates):
    overlap_any = np.zeros(ref_mask.shape, dtype=np.uint8)
    for cand in warped_candidates:
        support = cand.get("support_mask", cand["warped_mask"])
        overlap_any = np.maximum(overlap_any, support)

    overlap_with_ref = ((ref_mask > 0) & (overlap_any > 0)).astype(np.uint8) * 255
    return overlap_any, overlap_with_ref
