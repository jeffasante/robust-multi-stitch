import cv2
import numpy as np


def point_set_area(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3:
        return 0.0

    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def has_good_spatial_support(points, min_area=500.0):
    area = point_set_area(points)
    return area >= min_area


def flatten_points(pts):
    return pts.reshape(-1, 2)


def warp_corners(H, width, height):
    corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]).reshape(-1, 1, 2)

    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return warped


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return float(cv2.contourArea(hull))


def is_convex_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        return False
    hull = cv2.convexHull(pts.reshape(-1, 1, 2), returnPoints=True)
    hull = hull.reshape(-1, 2)
    return len(hull) == 4


def quad_edge_lengths(points):
    pts = np.asarray(points, dtype=np.float32)
    lengths = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        lengths.append(float(np.linalg.norm(a - b)))
    return lengths


def is_plausible_warped_image(H, src_width, src_height,
                              min_area_ratio=0.2,
                              max_area_ratio=5.0,
                              max_edge_ratio=8.0):
    warped = warp_corners(H, src_width, src_height)

    if not np.all(np.isfinite(warped)):
        return False

    if not is_convex_quad(warped):
        return False

    src_area = float(src_width * src_height)
    warped_area = polygon_area(warped)

    if warped_area < min_area_ratio * src_area:
        return False
    if warped_area > max_area_ratio * src_area:
        return False

    edges = quad_edge_lengths(warped)
    if min(edges) < 1e-6:
        return False

    edge_ratio = max(edges) / min(edges)
    if edge_ratio > max_edge_ratio:
        return False

    return True


def compute_homography_inliers(H, pts_src, pts_ref, threshold=4.0):
    pts_src_flat = flatten_points(pts_src)
    pts_ref_flat = flatten_points(pts_ref)

    projected = cv2.perspectiveTransform(
        pts_src_flat.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - pts_ref_flat, axis=1)
    inliers = errors < threshold
    return inliers, errors


def local_match_indices(points, center_idx, radius):
    center = points[center_idx]
    dists = np.linalg.norm(points - center, axis=1)
    return np.where(dists <= radius)[0]


def normalize_homography(H):
    if H is None:
        return None
    if abs(H[2, 2]) < 1e-8:
        return H
    return H / H[2, 2]


def homography_distance(H1, H2):
    H1 = normalize_homography(H1)
    H2 = normalize_homography(H2)
    return np.linalg.norm(H1 - H2)


def is_reasonable_homography(H, max_scale=4.0, min_det=1e-6):
    if H is None:
        return False

    A = H[:2, :2]
    det = np.linalg.det(A)
    if abs(det) < min_det:
        return False

    sx = np.linalg.norm(A[:, 0])
    sy = np.linalg.norm(A[:, 1])

    if sx > max_scale or sy > max_scale:
        return False
    if sx < 0.1 or sy < 0.1:
        return False

    return True


def deduplicate_candidates(candidates, matrix_thresh=1.0, overlap_thresh=0.8):
    kept = []

    for cand in sorted(candidates, key=lambda x: x["num_inliers"], reverse=True):
        keep = True

        for prev in kept:
            mat_dist = homography_distance(cand["H"], prev["H"])

            a = cand["inlier_mask"]
            b = prev["inlier_mask"]
            inter = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            iou = inter / max(union, 1)

            if mat_dist < matrix_thresh or iou > overlap_thresh:
                keep = False
                break

        if keep:
            kept.append(cand)

    return kept


def generate_candidate_homographies(
    pts_src,
    pts_ref,
    src_image_shape,
    num_trials=200,
    local_radius=120.0,
    min_local_matches=8,
    ransac_thresh=4.0,
    max_candidates=8,
    random_seed=42,
):
    rng = np.random.default_rng(random_seed)
    src_h, src_w = src_image_shape[:2]

    pts_src_flat = flatten_points(pts_src)
    num_matches = len(pts_src_flat)

    if num_matches < 4:
        raise ValueError("Need at least 4 matches.")

    candidates = []

    for _ in range(num_trials):
        seed_idx = int(rng.integers(0, num_matches))
        local_idx = local_match_indices(pts_src_flat, seed_idx, local_radius)

        if len(local_idx) < min_local_matches:
            continue

        if not has_good_spatial_support(pts_src_flat[local_idx], min_area=300.0):
            continue

        local_src = pts_src[local_idx]
        local_ref = pts_ref[local_idx]

        H, local_mask = cv2.findHomography(
            local_src, local_ref, cv2.RANSAC, ransac_thresh
        )

        if H is None:
            continue

        if not is_reasonable_homography(H):
            continue

        if not is_plausible_warped_image(H, src_w, src_h):
            continue

        global_inliers, global_errors = compute_homography_inliers(
            H, pts_src, pts_ref, threshold=ransac_thresh
        )

        global_src_inliers = pts_src_flat[global_inliers]
        global_ref_inliers = flatten_points(pts_ref)[global_inliers]

        if not has_good_spatial_support(global_src_inliers, min_area=500.0):
            continue

        if not has_good_spatial_support(global_ref_inliers, min_area=500.0):
            continue

        num_inliers = int(global_inliers.sum())
        if num_inliers < 10:
            continue

        candidates.append(
            {
                "H": H,
                "seed_idx": seed_idx,
                "local_idx": local_idx,
                "inlier_mask": global_inliers,
                "errors": global_errors,
                "num_inliers": num_inliers,
            }
        )

    candidates = deduplicate_candidates(candidates)

    candidates = sorted(candidates, key=lambda x: x["num_inliers"], reverse=True)
    return candidates[:max_candidates]