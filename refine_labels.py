import numpy as np
import cv2
from numba import njit


def grayscale_float(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def build_label_images(ref_canvas, warped_candidates):
    images = [ref_canvas]
    for cand in warped_candidates:
        images.append(cand["warped_img"])
    return images


def build_label_masks(ref_mask, warped_candidates):
    masks = [ref_mask]
    for cand in warped_candidates:
        masks.append(cand.get("support_mask", cand["warped_mask"]))
    return masks


@njit
def _numba_refine_icm(labels, unary_costs, images, masks, valid_region, smooth_lambda, num_iters):
    h, w = labels.shape
    num_labels = unary_costs.shape[0]

    for _ in range(num_iters):
        for y in range(h):
            for x in range(w):
                if valid_region[y, x] == 0:
                    continue

                best_label = labels[y, x]
                best_energy = np.inf

                for lab in range(num_labels):
                    if masks[lab, y, x] == 0:
                        continue
                    
                    energy = unary_costs[lab, y, x]
                    
                    neighbors_y = (y-1, y+1, y, y)
                    neighbors_x = (x, x, x-1, x+1)
                    
                    for idx in range(4):
                        ny = neighbors_y[idx]
                        nx = neighbors_x[idx]
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if valid_region[ny, nx] == 0:
                            continue
                            
                        nlab = labels[ny, nx]
                        if lab != nlab:
                            a0 = images[lab, y, x, 0]
                            a1 = images[lab, y, x, 1]
                            a2 = images[lab, y, x, 2]

                            b0 = images[nlab, y, x, 0]
                            b1 = images[nlab, y, x, 1]
                            b2 = images[nlab, y, x, 2]

                            c0 = images[lab, ny, nx, 0]
                            c1 = images[lab, ny, nx, 1]
                            c2 = images[lab, ny, nx, 2]

                            d0 = images[nlab, ny, nx, 0]
                            d1 = images[nlab, ny, nx, 1]
                            d2 = images[nlab, ny, nx, 2]
                            
                            dist_ab = np.sqrt((a0-b0)**2 + (a1-b1)**2 + (a2-b2)**2)
                            dist_cd = np.sqrt((c0-d0)**2 + (c1-d1)**2 + (c2-d2)**2)
                            
                            penalty = (dist_ab + dist_cd) / (2.0 * 255.0 * np.sqrt(3.0))
                            energy += smooth_lambda * penalty

                    if energy < best_energy:
                        best_energy = energy
                        best_label = lab
                
                labels[y, x] = best_label
                
    return labels

def refine_labels_icm(
    initial_labels,
    unary_costs,
    ref_canvas,
    ref_mask,
    warped_candidates,
    valid_region,
    smooth_lambda=0.25,
    num_iters=3,
):
    labels = initial_labels.copy()

    images = build_label_images(ref_canvas, warped_candidates)
    masks = build_label_masks(ref_mask, warped_candidates)
    
    images_arr = np.stack(images).astype(np.float32)
    masks_arr = np.stack(masks)
    
    return _numba_refine_icm(labels, unary_costs, images_arr, masks_arr, valid_region, smooth_lambda, num_iters)
