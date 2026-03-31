import numpy as np
import argparse
import os
import glob
import json

# Define your crop coordinates here.
# Format: view_type: (left_crop_orig, top_crop_orig, right_crop_orig, bottom_crop_orig)
# These are absolute pixel coordinates on the ORIGINAL image.
OMNIDET_CROPPED_COORDS = {
    "FV": (114, 110, 1176, 610),
    "MVL": (343, 5, 1088, 411),
    "MVR": (185, 5, 915, 425),
    "RV": (186, 203, 1105, 630)
    # Add other views if necessary.
    # If a view is not in this map, or get_view_type_from_filename returns None,
    # boxes from those files will be processed as if on an uncropped original image
    # that is then resized to network input.
}


def get_view_type_from_filename(filename):
    """
    Determines the view type (e.g., "FV", "MVL") from the filename.
    This needs to be robust for your specific filename conventions.
    Example: if filename is like '00001_FV.txt'
    """
    if "_FV" in filename:  # Check for _FV first to avoid matching FV in _FVL or _FVR if they exist
        return "FV"
    elif "_MVL" in filename:
        return "MVL"
    elif "_MVR" in filename:
        return "MVR"
    elif "_RV" in filename:
        return "RV"
    # Add more specific conditions if needed
    print(f"Warning: Could not determine view type for filename: {filename}")
    return None


def calc_iou_omni(ann, centroids):
    w, h = ann
    similarities = []
    for centroid in centroids:
        c_w, c_h = centroid
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h + eps)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h + eps)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h) + eps)
        else:
            intersection = np.minimum(w, c_w) * np.minimum(h, c_h)
            union = (w * h) + (c_w * c_h) - intersection
            similarity = intersection / (union + eps)
        similarities.append(similarity)
    return np.array(similarities)


eps = 1e-6


def run_kmeans_omni(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    if ann_num == 0:
        print("Error: No bounding box dimensions found to run K-Means.")
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    # Initialize centroids robustly
    if ann_num < anchor_num:
        print(f"  Initializing {anchor_num} centroids by sampling WITH REPLACEMENT from {ann_num} boxes.")
        indices = np.random.choice(ann_num, size=anchor_num, replace=True)
    else:
        print(f"  Initializing {anchor_num} centroids by sampling WITHOUT REPLACEMENT from {ann_num} boxes.")
        indices = np.random.choice(ann_num, size=anchor_num, replace=False)

    centroids = ann_dims[indices].astype(np.float64)

    if centroids.shape[0] != anchor_num:
        # This should ideally not happen if np.random.choice works as expected
        # and ann_num is sufficient.
        if ann_num > 0 and ann_num < anchor_num:  # If fewer unique boxes than anchors requested
            print(
                f"Warning: Could only initialize {centroids.shape[0]} centroids due to insufficient unique data points ({ann_num}) for {anchor_num} clusters. Using these {centroids.shape[0]} centroids.")
            anchor_num = centroids.shape[0]  # Adjust anchor_num to what was actually initialized
            if anchor_num == 0:  # No centroids could be initialized
                return np.empty((0, 2), dtype=np.float64), np.empty((0, 0), dtype=np.float64)
        else:
            raise ValueError(
                f"Centroid initialization failed. Expected {anchor_num} centroids, got {centroids.shape[0]}")

    print(
        f"  Initial centroids shape: {centroids.shape}. Example centroid0: {centroids[0] if anchor_num > 0 else 'N/A'}")
    prev_assignments = np.full(ann_num, -1, dtype=np.int32)
    iterations = 0
    max_iterations = 300

    while True:
        distances = np.zeros((ann_num, anchor_num), dtype=np.float64)
        for i in range(ann_num):
            distances[i, :] = 1 - calc_iou_omni(ann_dims[i], centroids)

        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all() or iterations >= max_iterations:
            if iterations >= max_iterations:
                print(f"K-Means reached max_iterations ({max_iterations}).")
            else:
                print(f"K-Means converged after {iterations} iterations.")
            return centroids, distances

        new_centroids_accumulator = np.zeros((anchor_num, 2), dtype=np.float64)
        counts = np.zeros(anchor_num, dtype=np.int32)

        for ann_idx, assignment_idx in enumerate(assignments):
            new_centroids_accumulator[assignment_idx] += ann_dims[ann_idx]
            counts[assignment_idx] += 1

        for c_idx in range(anchor_num):
            if counts[c_idx] > 0:
                centroids[c_idx] = new_centroids_accumulator[c_idx] / counts[c_idx]
            else:  # Handle empty cluster
                print(
                    f"    Warning: Cluster {c_idx} became empty at iteration {iterations}. Re-initializing centroid to a random point.")
                if ann_num > 0:  # Ensure there are points to sample from
                    centroids[c_idx] = ann_dims[np.random.choice(ann_num, 1)]
                # If ann_num is 0, this state shouldn't be reached due to earlier checks.
                # If it does, centroids[c_idx] remains unchanged from previous iteration or initial.

        prev_assignments = assignments.copy()
        iterations += 1


def get_objects_width_and_height_omni_cropped(annot_dir, crop_coords_map,
                                              net_input_w, net_input_h,
                                              original_w, original_h):  # Added original dimensions
    annot_files_list = glob.glob(os.path.join(annot_dir, "*.txt"))
    if not annot_files_list:
        print(f"Warning: No annotation files found in directory: {annot_dir}")
        return [], []
    objects_width_list = []
    objects_height_list = []
    print(f"Using crop_coords_map: {crop_coords_map}")
    print(f"Original image dimensions for scaling: W={original_w}, H={original_h}")
    print(f"Network input dimensions for final scaling: W={net_input_w}, H={net_input_h}")

    for file_counter, annot_file_path in enumerate(annot_files_list):
        filename = os.path.basename(annot_file_path)
        view_type = get_view_type_from_filename(filename)

        crop_details_for_file = None
        if view_type and view_type in crop_coords_map:
            crop_details_for_file = crop_coords_map[view_type]
        # else: # Decided to process uncropped if view_type not found, then resize full original
        #     print(f"Warning: No crop defined for view_type '{view_type}' (from file {filename}). Processing boxes relative to full original image, then resizing.")

        with open(annot_file_path, 'r') as annotation_file:
            for line_num, annot_object_line in enumerate(annotation_file):
                tokens = annot_object_line.strip().split(",")
                try:
                    if len(tokens) >= 6:
                        x1_o = int(float(tokens[2]))
                        y1_o = int(float(tokens[3]))
                        x2_o = int(float(tokens[4]))
                        y2_o = int(float(tokens[5]))

                        w_box_orig_scale = 0
                        h_box_orig_scale = 0

                        # Effective dimensions of the region that will be resized to network input
                        effective_source_w = original_w
                        effective_source_h = original_h

                        if crop_details_for_file:
                            cx1_o, cy1_o, cx2_o, cy2_o = crop_details_for_file

                            # Calculate intersection of box with crop window
                            ix1 = max(x1_o, cx1_o)
                            iy1 = max(y1_o, cy1_o)
                            ix2 = min(x2_o, cx2_o)
                            iy2 = min(y2_o, cy2_o)

                            w_box_orig_scale = ix2 - ix1
                            h_box_orig_scale = iy2 - iy1

                            effective_source_w = cx2_o - cx1_o
                            effective_source_h = cy2_o - cy1_o
                        else:  # No crop for this view, use original box dimensions
                            w_box_orig_scale = x2_o - x1_o
                            h_box_orig_scale = y2_o - y1_o
                            # effective_source_w and _h remain original_w, original_h

                        if w_box_orig_scale > 0 and h_box_orig_scale > 0 and \
                                effective_source_w > 0 and effective_source_h > 0:
                            # Scale these dimensions to the network input size
                            scale_w_factor = net_input_w / effective_source_w
                            scale_h_factor = net_input_h / effective_source_h

                            final_w = w_box_orig_scale * scale_w_factor
                            final_h = h_box_orig_scale * scale_h_factor

                            objects_width_list.append(final_w)
                            objects_height_list.append(final_h)

                except ValueError:
                    print(
                        f"Warning: Could not parse coordinates in {annot_file_path}, line {line_num + 1}: {annot_object_line.strip()}")
        if (file_counter + 1) % 100 == 0 or file_counter == len(annot_files_list) - 1:
            print(f"Processed {file_counter + 1}/{len(annot_files_list)} annotation files.")
    return objects_width_list, objects_height_list


def generate_anchors_omni_style(results_dir, annot_dir, num_anchors, k_mean_runs,
                                crop_coords_map, net_input_w, net_input_h, original_w, original_h):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")

    print("Parsing object dimensions from annotations (with cropping and resizing)...")
    objects_w, objects_h = get_objects_width_and_height_omni_cropped(
        annot_dir, crop_coords_map, net_input_w, net_input_h, original_w, original_h
    )

    if not objects_w or not objects_h:
        print("Error: No valid object dimensions found. Cannot generate anchors.")
        return None

    objects_wh_np = np.array(list(zip(objects_w, objects_h)), dtype=np.float64)
    print(f"Found {objects_wh_np.shape[0]} bounding boxes for K-Means (after cropping and resizing).")

    if objects_wh_np.shape[0] == 0:
        print("Error: No bounding boxes to run K-Means on.")
        return None

    best_distance_sum = float('inf')
    best_centroids = None

    print(f"Running K-Means {k_mean_runs} time(s) with {num_anchors} clusters using IoU distance...")
    for i in range(k_mean_runs):
        print(f"  K-Means Run {i + 1}/{k_mean_runs}")
        current_centroids, current_distances = run_kmeans_omni(objects_wh_np, num_anchors)
        if current_centroids is None or current_centroids.shape[0] == 0:
            print(f"    Run {i + 1} failed to produce centroids.")
            continue
        min_distances_for_run = np.min(current_distances, axis=1)
        current_sum_dist = np.sum(min_distances_for_run)
        print(f"    Run {i + 1}: Sum of (1-IoU) distances = {current_sum_dist:.4f}")
        if current_sum_dist < best_distance_sum:
            best_distance_sum = current_sum_dist
            best_centroids = current_centroids
            print(f"    Run {i + 1}: Updated best centroids.")

    if best_centroids is None:
        print("Error: K-Means failed to generate anchors after all runs.")
        return None

    best_centroids = best_centroids[np.argsort(best_centroids[:, 0] * best_centroids[:, 1])]

    print("\nGenerated Anchors (width, height) - Sorted by Area (scaled to network input):")
    for i, anchor in enumerate(best_centroids):
        print(f"  Anchor {i + 1}: [{anchor[0]:.2f}, {anchor[1]:.2f}]")

    out_file = os.path.join(results_dir, "anchor_boxes_final_processed.json")
    anchors_for_json = [[int(round(dim[0])), int(round(dim[1]))] for dim in best_centroids]

    data = dict(anchors=anchors_for_json)
    with open(out_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"\nAnchors (integer, sorted by area, based on processed dimensions) dumped to: {out_file}")

    return best_centroids


def main():
    parser = argparse.ArgumentParser(
        description='Argument parser for anchor generation (OmniDet style, with cropping & resizing)')
    parser.add_argument('-a', '--anchors_num', help='number of anchors', default=9, type=int)
    parser.add_argument('-kr', '--k_mean_runs', help='number of k-means runs for best result', default=10, type=int)
    parser.add_argument('-rd', '--results_dir', help='directory for output json file', type=str, required=True)
    parser.add_argument('-ad', '--annot_dir',
                        help='directory for bounding boxes annotation files (*.txt, comma-separated x1,y1,x2,y2)',
                        type=str, required=True)
    parser.add_argument('--original_width', type=int, required=True,
                        help="Original width of images before any processing.")
    parser.add_argument('--original_height', type=int, required=True,
                        help="Original height of images before any processing.")
    parser.add_argument('--net_input_width', type=int, required=True,
                        help="Network input width (after crop and resize).")
    parser.add_argument('--net_input_height', type=int, required=True,
                        help="Network input height (after crop and resize).")

    script_args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Annotation directory: {script_args.annot_dir}")
    print(f"  Number of anchors: {script_args.anchors_num}")
    print(f"  K-Means runs: {script_args.k_mean_runs}")
    print(f"  Results directory: {script_args.results_dir}")
    print(f"  Original Image Size: {script_args.original_width}x{script_args.original_height}")
    print(f"  Network Input Size: {script_args.net_input_width}x{script_args.net_input_height}")
    print(f"  Using OMNIDET_CROPPED_COORDS: {OMNIDET_CROPPED_COORDS}")
    print("-" * 30)

    generated_anchors = generate_anchors_omni_style(
        results_dir=script_args.results_dir,
        annot_dir=script_args.annot_dir,
        num_anchors=script_args.anchors_num,
        k_mean_runs=script_args.k_mean_runs,
        crop_coords_map=OMNIDET_CROPPED_COORDS,
        net_input_w=script_args.net_input_width,
        net_input_h=script_args.net_input_height,
        original_w=script_args.original_width,
        original_h=script_args.original_height
    )

    if generated_anchors is not None:
        print("\n--- Suggested Anchor Distribution for YOLO (3 heads, 3 anchors each if num_anchors=9) ---")
        print("Anchors are sorted by area. Assign smallest to P3, medium to P4, largest to P5.")

        num_heads = 3
        if generated_anchors.shape[0] == script_args.anchors_num and script_args.anchors_num % num_heads == 0:
            anchors_per_head = script_args.anchors_num // num_heads

            print("\nFor P3 head (e.g., args.anchors3 in JSON - smallest areas):")
            for i in range(anchors_per_head):
                print(f"  [{int(round(generated_anchors[i, 0]))}, {int(round(generated_anchors[i, 1]))}]")

            print("\nFor P4 head (e.g., args.anchors2 in JSON - medium areas):")
            for i in range(anchors_per_head, 2 * anchors_per_head):
                print(f"  [{int(round(generated_anchors[i, 0]))}, {int(round(generated_anchors[i, 1]))}]")

            print("\nFor P5 head (e.g., args.anchors1 in JSON - largest areas):")
            for i in range(2 * anchors_per_head, 3 * anchors_per_head):
                print(f"  [{int(round(generated_anchors[i, 0]))}, {int(round(generated_anchors[i, 1]))}]")

            print("\nCopy these integer values into your JSON configuration.")
        else:
            print(
                f"Could not automatically distribute {generated_anchors.shape[0]} anchors into {num_heads} heads. Please distribute manually.")


if __name__ == "__main__":
    main()
