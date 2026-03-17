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
from canvas import (
    compute_global_canvas,
    place_reference_on_global_canvas,
    warp_source_to_global_canvas,
)
from confidence import compute_motion_confidence_map
from unary_costs import build_unary_cost_volume
from label_assignment import argmin_label_map
from composite import composite_from_labels
from visualize import draw_matches, colorize_heatmap, colorize_label_map, overlay_mask_on_image
from smooth_labels import majority_vote_filter, edge_aware_mode_filter
from regions import compute_overlap_region
from roi import bounding_box_from_mask, crop_array, paste_array
from refine_labels import refine_labels_icm
from support_masks import build_candidate_support_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--outdir", default="data/outputs/unary")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    ref_img = load_image(args.ref)
    src_img = load_image(args.src)

    kp_ref, desc_ref = detect_and_describe(ref_img)
    kp_src, desc_src = detect_and_describe(src_img)

    matches = match_descriptors(desc_src, desc_ref, ratio_test=0.75)
    print(f"Total keypoints ref: {len(kp_ref)}")
    print(f"Total keypoints src: {len(kp_src)}")
    print(f"Total good matches: {len(matches)}")

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

    print(f"Using {len(candidates)} initial candidates before support mask generation")

    homographies = [cand["H"] for cand in candidates]
    T, width, height = compute_global_canvas(ref_img, src_img, homographies)

    ref_canvas, ref_mask = place_reference_on_global_canvas(ref_img, T, width, height)

    warped_candidates = []
    motion_conf_maps = []

    from local_warp import build_apap_field, warp_image_apap

    for i, cand in enumerate(candidates):
        use_apap = False  # Keep APAP disabled in main pipeline for stability

        if use_apap:
            field = build_apap_field(
                pts_src=pts_src,
                pts_ref=pts_ref,
                src_shape=src_img.shape,
                global_H=cand["H"],
                grid_rows=18,
                grid_cols=24,
                sigma=120.0,
                min_weighted_points=8,
            )

            warped_img, warped_mask, map_x, map_y = warp_image_apap(
                src_img,
                field,
                T=T,
                out_w=width,
                out_h=height,
            )

            H_total = None
        else:
            warped_img, warped_mask, H_total = warp_source_to_global_canvas(
                src_img, cand["H"], T, width, height
            )

        cand_runtime = {
            "id": i,
            "H": cand["H"],
            "H_total": H_total,
            "warped_img": warped_img,
            "warped_mask": warped_mask,
            "inlier_mask": cand["inlier_mask"],
            "is_apap": use_apap,
        }

        motion_conf = compute_motion_confidence_map(
            cand_runtime,
            pts_src.reshape(-1, 1, 2),
            width,
            height,
            sigma=25.0,
        )

        support_mask = build_candidate_support_mask(
            motion_conf,
            warped_mask,
            conf_thresh=0.08,
            min_area=600,
            dilate_iters=4,
        )

        cand_runtime["support_mask"] = support_mask
        warped_candidates.append(cand_runtime)
        motion_conf_maps.append(motion_conf)

        support_overlay = overlay_mask_on_image(warped_img, support_mask)

        tag = "apap" if use_apap else "rigid"
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_{tag}_warped.jpg"), warped_img)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_{tag}_mask.jpg"), warped_mask)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_motion_conf.jpg"), colorize_heatmap(motion_conf))
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_{tag}_support.jpg"), support_mask)
        cv2.imwrite(str(Path(args.outdir) / f"candidate_{i:02d}_support_overlay.jpg"), support_overlay)

    if not warped_candidates:
        print("WARNING: No valid candidate homographies found. Proceeding with reference-only.")
        keep_idx = []
    else:
        candidate_scores = []
        for cand in warped_candidates:
            support = cand["support_mask"]
            area = int(np.count_nonzero(support))
            candidate_scores.append(area)

        keep_idx = [0]
        rest = list(range(1, len(warped_candidates)))
        rest = sorted(rest, key=lambda i: candidate_scores[i], reverse=True)

        max_local_keep = 3
        keep_idx.extend(rest[:max_local_keep])

        warped_candidates = [warped_candidates[i] for i in keep_idx]
        motion_conf_maps = [motion_conf_maps[i] for i in keep_idx]
        candidates = [candidates[i] for i in keep_idx]

        print("Kept candidate indices:", keep_idx)
    for i, cand in enumerate(warped_candidates):
        area = int(np.count_nonzero(cand["support_mask"]))
        print(f"candidate {i} support area: {area}")

    overlap_any, overlap_with_ref = compute_overlap_region(ref_mask, warped_candidates)

    cv2.imwrite(str(Path(args.outdir) / "overlap_any.jpg"), overlap_any)
    cv2.imwrite(str(Path(args.outdir) / "overlap_with_ref.jpg"), overlap_with_ref)

    bbox = bounding_box_from_mask(overlap_with_ref, pad=20)
    if bbox is None:
        raise RuntimeError("No overlap region found.")

    cost_volume = build_unary_cost_volume(
        ref_canvas=ref_canvas,
        ref_mask=ref_mask,
        warped_candidates=warped_candidates,
        motion_conf_maps=motion_conf_maps,
        alpha_motion=0.7,
        alpha_photo=0.3,
        invalid_cost=1.0,
    )

    label_map = argmin_label_map(cost_volume)

    ref_canvas_roi = crop_array(ref_canvas, bbox)
    ref_mask_roi = crop_array(ref_mask, bbox)
    overlap_with_ref_roi = crop_array(overlap_with_ref, bbox)
    cost_volume_roi = crop_array(cost_volume.transpose(1, 2, 0), bbox).transpose(2, 0, 1)

    warped_candidates_roi = []
    for cand in warped_candidates:
        warped_candidates_roi.append({
            "warped_img": crop_array(cand["warped_img"], bbox),
            "warped_mask": crop_array(cand["warped_mask"], bbox),
            "support_mask": crop_array(cand["support_mask"], bbox),
        })

    label_map_roi = crop_array(label_map, bbox)

    label_map_majority_roi = majority_vote_filter(
        label_map_roi,
        valid_mask=overlap_with_ref_roi,
        kernel_size=11,
    )

    label_map_smooth_roi = edge_aware_mode_filter(
        label_map_majority_roi,
        guide_image=ref_canvas_roi,
        valid_mask=overlap_with_ref_roi,
        radius=4,
        edge_power=8.0,
    )

    label_map_refined_roi = refine_labels_icm(
        initial_labels=label_map_smooth_roi,
        unary_costs=cost_volume_roi,
        ref_canvas=ref_canvas_roi,
        ref_mask=ref_mask_roi,
        warped_candidates=warped_candidates_roi,
        valid_region=overlap_with_ref_roi,
        smooth_lambda=0.35,
        num_iters=4,
    )

    label_map_majority = label_map.copy()
    label_map_smooth = label_map.copy()
    label_map_refined = label_map.copy()

    paste_array(label_map_majority, label_map_majority_roi, bbox)
    paste_array(label_map_smooth, label_map_smooth_roi, bbox)
    paste_array(label_map_refined, label_map_refined_roi, bbox)

    composite_unary = composite_from_labels(ref_canvas, warped_candidates, label_map)
    composite_smooth = composite_from_labels(ref_canvas, warped_candidates, label_map_smooth)
    composite_refined = composite_from_labels(ref_canvas, warped_candidates, label_map_refined)

    print("Label usage raw:")
    unique, counts = np.unique(label_map, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c} pixels")

    print("Label usage smooth:")
    unique_s, counts_s = np.unique(label_map_smooth, return_counts=True)
    for u, c in zip(unique_s, counts_s):
        print(f"  label {u}: {c} pixels")

    print("Label usage refined:")
    unique_r, counts_r = np.unique(label_map_refined, return_counts=True)
    for u, c in zip(unique_r, counts_r):
        print(f"  label {u}: {c} pixels")

    cv2.imwrite(str(Path(args.outdir) / "ref_canvas.jpg"), ref_canvas)
    cv2.imwrite(str(Path(args.outdir) / "ref_mask.jpg"), ref_mask)
    cv2.imwrite(str(Path(args.outdir) / "label_map_raw.png"), colorize_label_map(label_map, 1 + len(warped_candidates)))
    cv2.imwrite(str(Path(args.outdir) / "label_map_majority.png"), colorize_label_map(label_map_majority, 1 + len(warped_candidates)))
    cv2.imwrite(str(Path(args.outdir) / "label_map_smooth.png"), colorize_label_map(label_map_smooth, 1 + len(warped_candidates)))
    cv2.imwrite(str(Path(args.outdir) / "label_map_refined.png"), colorize_label_map(label_map_refined, 1 + len(warped_candidates)))
    cv2.imwrite(str(Path(args.outdir) / "composite_unary.jpg"), composite_unary)
    cv2.imwrite(str(Path(args.outdir) / "composite_smooth.jpg"), composite_smooth)
    cv2.imwrite(str(Path(args.outdir) / "composite_refined.jpg"), composite_refined)

    match_vis = draw_matches(src_img, kp_src, ref_img, kp_ref, matches, max_matches=200)
    cv2.imwrite(str(Path(args.outdir) / "matches.jpg"), match_vis)

    print(f"Saved unary-selection outputs to {args.outdir}")


if __name__ == "__main__":
    main()