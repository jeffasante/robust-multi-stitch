import argparse
from pathlib import Path
import cv2

from io_utils import load_image, ensure_dir
from features import detect_and_describe, match_descriptors, matched_points
from homography import estimate_homography, count_inliers
from warp import warp_pair
from visualize import draw_matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Reference image path")
    parser.add_argument("--src", required=True, help="Source image path")
    parser.add_argument("--outdir", default="data/outputs/baseline")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    ref_img = load_image(args.ref)
    src_img = load_image(args.src)

    kp_ref, desc_ref = detect_and_describe(ref_img)
    kp_src, desc_src = detect_and_describe(src_img)

    matches = match_descriptors(desc_src, desc_ref, ratio_test=0.75)
    if len(matches) < 4:
        raise RuntimeError(f"Not enough good matches: {len(matches)}")

    pts_src, pts_ref = matched_points(kp_src, kp_ref, matches)
    H, inlier_mask = estimate_homography(pts_src, pts_ref)

    print(f"Matches: {len(matches)}")
    print(f"Inliers: {count_inliers(inlier_mask)}")
    print("Homography:")
    print(H)

    match_vis = draw_matches(src_img, kp_src, ref_img, kp_ref, matches)
    stitched, warped_src, _ = warp_pair(ref_img, src_img, H)

    cv2.imwrite(str(Path(args.outdir) / "matches.jpg"), match_vis)
    cv2.imwrite(str(Path(args.outdir) / "warped_src.jpg"), warped_src)
    cv2.imwrite(str(Path(args.outdir) / "stitched_baseline.jpg"), stitched)

    print(f"Saved outputs to {args.outdir}")


if __name__ == "__main__":
    main()