import cv2
import numpy as np


def rasterize_inlier_support(points_xy, width, height, sigma=25.0):
    support = np.zeros((height, width), dtype=np.float32)

    pts = np.round(points_xy).astype(int)
    for x, y in pts:
        if 0 <= x < width and 0 <= y < height:
            support[y, x] += 1.0

    ksize = int(max(3, 2 * round(3 * sigma) + 1))
    support = cv2.GaussianBlur(support, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    if support.max() > 1e-8:
        support /= support.max()

    return support


def project_points(H_total, pts_src):
    return cv2.perspectiveTransform(pts_src, H_total).reshape(-1, 2)


def compute_motion_confidence_map(candidate, pts_src, width, height, sigma=25.0):
    inlier_pts_src = pts_src[candidate["inlier_mask"]]
    if len(inlier_pts_src) == 0:
        return np.zeros((height, width), dtype=np.float32)

    if candidate.get("is_apap", False):
        # fallback: use support from warped image content itself for APAP path
        valid = (candidate["warped_mask"] > 0).astype(np.uint8)
        conf = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        if conf.max() > 1e-8:
            conf /= conf.max()
        return conf

    warped_pts = project_points(candidate["H_total"], inlier_pts_src)
    return rasterize_inlier_support(warped_pts, width, height, sigma=sigma)