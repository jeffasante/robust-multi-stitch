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

from warp import warp_candidates
from visualize import draw_matches, draw_candidate_inliers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--outdir", default="data/outputs/candidates")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    ref_img = load_image(args.ref)
    src_img = load_image(args.src)

    kp_ref, desc_ref = detect_and_describe(ref_img)
    kp_src, desc_src = detect_and_describe(src_img)

    matches = match_descriptors(desc_src, desc_ref, ratio_test=0.75)
    if len(matches) < 4:
        raise RuntimeError(f"Not enough good matches: {len(matches)}")

    print(f"Total keypoints ref: {len(kp_ref)}")
    print(f"Total keypoints src: {len(kp_src)}")
    print(f"Total good matches: {len(matches)}")

    pts_src, pts_ref = matched_points(kp_src, kp_ref, matches)

    H_global, global_mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 4.0)
    if H_global is not None:
        print(f"Global inliers: {int(global_mask.ravel().sum())}")

    match_vis = draw_matches(src_img, kp_src, ref_img, kp_ref, matches)
    cv2.imwrite(str(Path(args.outdir) / "matches.jpg"), match_vis)

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

    print(f"Found {len(candidates)} candidate homographies")

    homographies = []
    for i, cand in enumerate(candidates):
        print(
            f"[{i}] inliers={cand['num_inliers']} "
            f"seed={cand['seed_idx']} "
            f"local_support={len(cand['local_idx'])}"
        )
        homographies.append(cand["H"])

        inlier_vis = draw_candidate_inliers(
            src_img,
            ref_img,
            pts_src,
            pts_ref,
            cand["inlier_mask"],
            max_lines=120,
        )
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_inliers.jpg"), inlier_vis)

    warped_results = warp_candidates(ref_img, src_img, homographies)

    for item in warped_results:
        idx = item["id"]
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{idx:02d}_warped.jpg"), item["warped_src"])
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{idx:02d}_overlay.jpg"), item["overlay"])
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{idx:02d}_mask.jpg"), item["warped_mask"])

    print(f"Saved candidate outputs to {args.outdir}")


if __name__ == "__main__":
    main()