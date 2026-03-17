import cv2
import numpy as np
from numba import njit


def majority_vote_filter(label_map, valid_mask, kernel_size=9):
    labels = np.unique(label_map)
    h, w = label_map.shape

    scores = np.zeros((len(labels), h, w), dtype=np.float32)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    for i, lab in enumerate(labels):
        binary = (label_map == lab).astype(np.float32)
        scores[i] = cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_REFLECT)

    smoothed = labels[np.argmax(scores, axis=0)]

    out = label_map.copy()
    out[valid_mask > 0] = smoothed[valid_mask > 0]
    return out.astype(label_map.dtype)


def compute_boundary_strength(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 1e-8:
        mag /= mag.max()
    return mag


@njit
def _numba_edge_aware_filter(label_map, boundary, valid_mask, labels, padded_labels, padded_valid, radius, edge_power, spatial):
    h, w = label_map.shape
    out = label_map.copy()
    num_labels = len(labels)
    
    for y in range(h):
        for x in range(w):
            if valid_mask[y, x] == 0:
                continue

            y0 = y
            x0 = x
            
            best_label = label_map[y, x]
            best_score = -1.0

            for l_idx in range(num_labels):
                lab = labels[l_idx]
                score = 0.0
                
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        py = y0 + radius + dy
                        px = x0 + radius + dx
                        
                        if padded_valid[py, px] == 0:
                            continue
                        if padded_labels[py, px] != lab:
                            continue
                            
                        sy = y + dy
                        sx = x + dx
                        if sy < 0: sy = 0
                        if sy >= h: sy = h - 1
                        if sx < 0: sx = 0
                        if sx >= w: sx = w - 1
                            
                        b = boundary[sy, sx]
                        s_weight = spatial[dy + radius, dx + radius]
                        score += s_weight * np.exp(-edge_power * b)

                if score > best_score:
                    best_score = score
                    best_label = lab
            out[y, x] = best_label
    return out


def edge_aware_mode_filter(label_map, guide_image, valid_mask, radius=4, edge_power=8.0):
    boundary = compute_boundary_strength(guide_image)

    labels = np.unique(label_map)
    padded_labels = np.pad(label_map, radius, mode="edge")
    padded_valid = np.pad(valid_mask, radius, mode="constant")
    
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial = np.exp(-(xx * xx + yy * yy) / (2.0 * max(radius, 1) ** 2))
    
    return _numba_edge_aware_filter(
        label_map, boundary, valid_mask, labels, padded_labels, padded_valid,
        radius, edge_power, spatial
    )
