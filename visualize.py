import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches, max_matches=80):
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis


def pad_to_same_height(img_a, img_b):
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]

    h = max(ha, hb)

    def pad(img, target_h):
        hh, ww = img.shape[:2]
        if hh == target_h:
            return img
        pad_bottom = target_h - hh
        return cv2.copyMakeBorder(
            img,
            0, pad_bottom, 0, 0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

    return pad(img_a, h), pad(img_b, h)


def draw_candidate_inliers(img_src, img_ref, pts_src, pts_ref, inlier_mask, max_lines=100):
    src_vis = img_src.copy()
    ref_vis = img_ref.copy()

    pts_src = pts_src.reshape(-1, 2)
    pts_ref = pts_ref.reshape(-1, 2)

    src_vis, ref_vis = pad_to_same_height(src_vis, ref_vis)

    inlier_idx = np.where(inlier_mask)[0]
    if len(inlier_idx) > max_lines:
        inlier_idx = inlier_idx[:max_lines]

    joined = np.concatenate([src_vis, ref_vis], axis=1)
    offset_x = src_vis.shape[1]

    rng = np.random.default_rng(0)

    for i in inlier_idx:
        color = tuple(int(x) for x in rng.integers(0, 255, size=3))
        p1 = tuple(np.round(pts_src[i]).astype(int))
        p2 = tuple(np.round(pts_ref[i]).astype(int) + np.array([offset_x, 0]))

        cv2.circle(joined, p1, 4, color, -1)
        cv2.circle(joined, p2, 4, color, -1)
        cv2.line(joined, p1, p2, color, 2)

    return joined



def colorize_heatmap(x):
    x = np.clip(x, 0.0, 1.0)
    x_uint8 = (255 * x).astype(np.uint8)
    return cv2.applyColorMap(x_uint8, cv2.COLORMAP_JET)


def colorize_label_map(label_map, num_labels):
    h, w = label_map.shape
    colors = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 255],
        [255, 128, 128],
        [128, 255, 128],
        [200, 200, 200],
        [80, 180, 255],
    ], dtype=np.uint8)

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label in range(num_labels):
        color = colors[label % len(colors)]
        vis[label_map == label] = color
    return vis


def overlay_mask_on_image(image, mask):
    out = image.copy()
    out[mask == 0] = (0.3 * out[mask == 0]).astype(out.dtype)
    return out