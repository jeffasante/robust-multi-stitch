import cv2
import numpy as np


def flatten_points(pts):
    return pts.reshape(-1, 2).astype(np.float32)


def make_grid(width, height, grid_rows=18, grid_cols=24):
    xs = np.linspace(0, width - 1, grid_cols).astype(np.float32)
    ys = np.linspace(0, height - 1, grid_rows).astype(np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y


def weighted_dlt_homography(src_pts, ref_pts, weights, eps=1e-8):
    src_pts = np.asarray(src_pts, dtype=np.float64)
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = weights > eps
    src_pts = src_pts[valid]
    ref_pts = ref_pts[valid]
    weights = weights[valid]

    if len(src_pts) < 4:
        return None

    A = []
    for (x, y), (u, v), w in zip(src_pts, ref_pts, weights):
        s = np.sqrt(w)
        A.append(s * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u], dtype=np.float64))
        A.append(s * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v], dtype=np.float64))

    A = np.stack(A, axis=0)

    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    h = vt[-1]
    if abs(h[-1]) < eps:
        return None

    H = h.reshape(3, 3)
    H /= H[2, 2]
    return H.astype(np.float32)


def compute_apap_weights(grid_point, src_pts, sigma=120.0, gamma=1e-3):
    diff = src_pts - grid_point[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    w = np.exp(-dist2 / (2.0 * sigma * sigma))
    w = np.maximum(w, gamma)
    return w.astype(np.float32)


def build_apap_field(
    pts_src,
    pts_ref,
    src_shape,
    global_H=None,
    grid_rows=18,
    grid_cols=24,
    sigma=120.0,
    min_weighted_points=8,
):
    src_h, src_w = src_shape[:2]
    src_pts = flatten_points(pts_src)
    ref_pts = flatten_points(pts_ref)

    grid_x, grid_y = make_grid(src_w, src_h, grid_rows=grid_rows, grid_cols=grid_cols)

    field = np.zeros((grid_rows, grid_cols, 3, 3), dtype=np.float32)
    valid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    for r in range(grid_rows):
        for c in range(grid_cols):
            gp = np.array([grid_x[r, c], grid_y[r, c]], dtype=np.float32)
            w = compute_apap_weights(gp, src_pts, sigma=sigma)

            if np.sum(w > 1e-2) < min_weighted_points:
                if global_H is not None:
                    field[r, c] = global_H
                    valid[r, c] = 1
                continue

            H_local = weighted_dlt_homography(src_pts, ref_pts, w)
            if H_local is None:
                if global_H is not None:
                    field[r, c] = global_H
                    valid[r, c] = 1
                continue

            field[r, c] = H_local
            valid[r, c] = 1

    if global_H is not None:
        for r in range(grid_rows):
            for c in range(grid_cols):
                if valid[r, c] == 0:
                    field[r, c] = global_H
                    valid[r, c] = 1

    return {
        "field": field,
        "valid": valid,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "src_w": src_w,
        "src_h": src_h,
    }


def apply_homography(H, xy):
    x, y = float(xy[0]), float(xy[1])
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    if abs(q[2]) < 1e-8:
        return np.array([np.nan, np.nan], dtype=np.float32)
    return np.array([q[0] / q[2], q[1] / q[2]], dtype=np.float32)


def bilinear_blend_homographies(H00, H01, H10, H11, tx, ty):
    H = (
        (1 - tx) * (1 - ty) * H00 +
        tx * (1 - ty) * H01 +
        (1 - tx) * ty * H10 +
        tx * ty * H11
    )
    if abs(H[2, 2]) > 1e-8:
        H = H / H[2, 2]
    return H.astype(np.float32)


def local_homography_at(field_data, x, y):
    grid_x = field_data["grid_x"]
    grid_y = field_data["grid_y"]
    field = field_data["field"]
    rows = field_data["grid_rows"]
    cols = field_data["grid_cols"]

    xs = grid_x[0]
    ys = grid_y[:, 0]

    c = np.searchsorted(xs, x) - 1
    r = np.searchsorted(ys, y) - 1

    c = int(np.clip(c, 0, cols - 2))
    r = int(np.clip(r, 0, rows - 2))

    x0, x1 = xs[c], xs[c + 1]
    y0, y1 = ys[r], ys[r + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    H00 = field[r, c]
    H01 = field[r, c + 1]
    H10 = field[r + 1, c]
    H11 = field[r + 1, c + 1]

    return bilinear_blend_homographies(H00, H01, H10, H11, tx, ty)


def warp_points_apap(field_data, pts):
    pts = flatten_points(pts)
    out = np.zeros_like(pts, dtype=np.float32)

    for i, (x, y) in enumerate(pts):
        H = local_homography_at(field_data, x, y)
        out[i] = apply_homography(H, (x, y))

    return out


def warp_image_apap(src_img, field_data, T, out_w, out_h):
    src_h, src_w = src_img.shape[:2]

    map_x = np.full((out_h, out_w), -1, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1, dtype=np.float32)

    T_inv = np.linalg.inv(T).astype(np.float32)

    step = 2
    for sy in range(0, src_h, step):
        for sx in range(0, src_w, step):
            H = local_homography_at(field_data, sx, sy)
            dst_xy = apply_homography(H, (sx, sy))
            if np.any(np.isnan(dst_xy)):
                continue

            dst_xy = apply_homography(T, dst_xy)
            dx = int(round(dst_xy[0]))
            dy = int(round(dst_xy[1]))

            if 0 <= dx < out_w and 0 <= dy < out_h:
                map_x[dy, dx] = sx
                map_y[dy, dx] = sy

    missing = (map_x < 0) | (map_y < 0)
    if np.any(~missing):
        valid_x = map_x.copy()
        valid_y = map_y.copy()

        valid_x[missing] = 0
        valid_y[missing] = 0

        map_x = cv2.inpaint((valid_x * 255 / max(src_w - 1, 1)).astype(np.uint8), missing.astype(np.uint8), 3, cv2.INPAINT_NS)
        map_y = cv2.inpaint((valid_y * 255 / max(src_h - 1, 1)).astype(np.uint8), missing.astype(np.uint8), 3, cv2.INPAINT_NS)

        map_x = map_x.astype(np.float32) * max(src_w - 1, 1) / 255.0
        map_y = map_y.astype(np.float32) * max(src_h - 1, 1) / 255.0

    warped = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    src_mask = np.ones((src_h, src_w), dtype=np.uint8) * 255
    warped_mask = cv2.remap(
        src_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255

    return warped, warped_mask, map_x, map_y
