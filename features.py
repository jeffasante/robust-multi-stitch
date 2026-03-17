import cv2
import numpy as np


def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(
            nfeatures=8000,
            contrastThreshold=0.01,
            edgeThreshold=10
        )
    raise RuntimeError("SIFT is not available in this OpenCV build.")


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = create_sift()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio_test=0.8):
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(desc1, desc2, k=2)

    ratio_good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            ratio_good.append(m)

    cross_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    cross_good = cross_matcher.match(desc1, desc2)

    merged = {}
    for m in ratio_good:
        key = (m.queryIdx, m.trainIdx)
        merged[key] = m

    for m in cross_good:
        key = (m.queryIdx, m.trainIdx)
        if key not in merged or m.distance < merged[key].distance:
            merged[key] = m

    matches = list(merged.values())
    matches.sort(key=lambda m: m.distance)
    return matches

def matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts1, pts2