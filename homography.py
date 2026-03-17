import cv2
import numpy as np


def estimate_homography(pts_src, pts_ref, ransac_thresh=4.0):
    if len(pts_src) < 4 or len(pts_ref) < 4:
        raise ValueError("Need at least 4 correspondences for homography.")
    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H, mask


def count_inliers(mask):
    if mask is None:
        return 0
    return int(mask.ravel().sum())