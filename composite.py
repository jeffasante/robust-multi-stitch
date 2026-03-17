import numpy as np
import cv2


def composite_from_labels(ref_canvas, warped_candidates, label_map, blend_radius=20):
    h, w = label_map.shape
    out = np.zeros_like(ref_canvas, dtype=np.float32)
    ref_float = ref_canvas.astype(np.float32)

    # Base starts exactly as the reference
    out[:] = ref_float[:]

    for i, cand in enumerate(warped_candidates, start=1):
        mask = (label_map == i).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
            
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        alpha = np.clip(dist / blend_radius, 0, 1.0)
        alpha = np.dstack([alpha, alpha, alpha])
        
        cand_float = cand["warped_img"].astype(np.float32)
        
        active = (mask > 0)
        out[active] = cand_float[active] * alpha[active] + ref_float[active] * (1.0 - alpha[active])

    return out.astype(np.uint8)