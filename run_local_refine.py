import argparse
from pathlib import Path
import cv2
import numpy as np

from io_utils import load_image, ensure_dir
from features import detect_and_describe, match_descriptors, matched_points
from candidate_generation import (
    generate_candidate_homographies,
    compute_homography_inliers,
    is_reasonable_homography,
    is_plausible_warped_image,
)
from canvas import compute_global_canvas, place_reference_on_global_canvas, warp_source_to_global_canvas
from local_warp import build_apap_field, warp_image_apap
from visualize import draw_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--outdir", default="data/outputs/local_refine")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    ref_img = load_image(args.ref)
    src_img = load_image(args.src)

    kp_ref, desc_ref = detect_and_describe(ref_img)
    kp_src, desc_src = detect_and_describe(src_img)

    matches = match_descriptors(desc_src, desc_ref, ratio_test=0.75)
    pts_src, pts_ref = matched_points(kp_src, kp_ref, matches)

    H_global, global_mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 4.0)

    candidates = generate_candidate_homographies(
        pts_src=pts_src,
        pts_ref=pts_ref,
        src_image_shape=src_img.shape,
        num_trials=800,
        local_radius=60.0,
        min_local_matches=5,
        ransac_thresh=4.0,
        max_candidates=10,
        random_seed=42,
    )

    if H_global is not None and is_reasonable_homography(H_global):
        if is_plausible_warped_image(H_global, src_img.shape[1], src_img.shape[0]):
            global_inliers, global_errors = compute_homography_inliers(
                H_global, pts_src, pts_ref, threshold=4.0
            )
            candidates.insert(0, {
                "H": H_global,
                "seed_idx": -1,
                "local_idx": list(range(len(pts_src))),
                "inlier_mask": global_inliers,
                "errors": global_errors,
                "num_inliers": int(global_inliers.sum()),
            })

    candidates = candidates[:3]
    homographies = [c["H"] for c in candidates]

    T, out_w, out_h = compute_global_canvas(ref_img, src_img, homographies)
    ref_canvas, ref_mask = place_reference_on_global_canvas(ref_img, T, out_w, out_h)

    cv2.imwrite(str(Path(args.outdir) / "matches.jpg"), draw_matches(src_img, kp_src, ref_img, kp_ref, matches, max_matches=200))

    for i, cand in enumerate(candidates):
        H = cand["H"]

        warped_h, warped_mask_h, H_total = warp_source_to_global_canvas(src_img, H, T, out_w, out_h)

        overlay_before = warped_h.copy()
        overlay_before[ref_mask > 0] = ref_canvas[ref_mask > 0]

        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_before.jpg"), warped_h)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_before_overlay.jpg"), overlay_before)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_before_mask.jpg"), warped_mask_h)

        field = build_apap_field(
            pts_src=pts_src,
            pts_ref=pts_ref,
            src_shape=src_img.shape,
            global_H=H,
            grid_rows=18,
            grid_cols=24,
            sigma=120.0,
            min_weighted_points=8,
        )

        warped_apap, warped_mask_apap, map_x, map_y = warp_image_apap(
            src_img,
            field,
            T=T,
            out_w=out_w,
            out_h=out_h,
        )

        overlay_after = warped_apap.copy()
        overlay_after[ref_mask > 0] = ref_canvas[ref_mask > 0]

        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_after.jpg"), warped_apap)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_after_overlay.jpg"), overlay_after)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_after_mask.jpg"), warped_mask_apap)

        diff_before = cv2.absdiff(ref_canvas, warped_h)
        diff_after = cv2.absdiff(ref_canvas, warped_apap)

        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_diff_before.jpg"), diff_before)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_diff_after.jpg"), diff_after)

        print(f"candidate {i}: saved before/after APAP-style refinement")

    print(f"Saved outputs to {args.outdir}")


if __name__ == "__main__":
    main()
