"""
Detection utils for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

Parts of the code adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3
Please refer to the license of the above repo.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def to_cpu(tensor):
    return tensor.detach().cpu()


def get_tensor_value(tensor):
    if isinstance(tensor, torch.Tensor):
        return to_cpu(tensor).item()
    else:
        return tensor


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = (list(), list(), list())
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    :param recall:    The recall curve (list).
    :param precision: The precision curve (list).
    :return The average precision.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# --- Utility Functions (Ensure these are robust in your project) ---
def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert nx4 boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    where xy1=top-left, xy2=bottom-right.
    """
    # Create a new tensor with the same properties as x
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)

    # Calculate coordinates using broadcasting-safe indexing
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates IoU (Intersection over Union) between box1 and box2.
    box1: (N, 4) tensor, predicted boxes
    box2: (M, 4) tensor, target boxes
    x1y1x2y2: If True, boxes are in [x1, y1, x2, y2] format, otherwise [cx, cy, w, h]
    eps: Small epsilon to prevent division by zero.
    Returns: (N, M) tensor of IoUs between all pairs of boxes.
    """
    # Ensure inputs are on the same device
    device = box1.device
    box2 = box2.to(device)

    # Ensure inputs are at least 2D for consistent processing
    if box1.ndim == 1: box1 = box1.unsqueeze(0) # (4,) -> (1,4)
    if box2.ndim == 1: box2 = box2.unsqueeze(0) # (4,) -> (1,4)

    if not x1y1x2y2: # If boxes are cxcywh, convert to x1y1x2y2
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    # Get the coordinates of bounding boxes
    # .unbind(-1) splits along the last dimension, so each is (N,) or (M,)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    # Intersection area
    # b1_x1 is (N,), b2_x1 is (M,). Unsqueeze b1 to (N,1) for broadcasting with (M,) to get (N,M)
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # Shape (N,)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # Shape (M,)

    union_area = b1_area.unsqueeze(1) + b2_area - inter_area + eps # (N,1) + (M,) - (N,M) -> (N,M)
    iou = inter_area / union_area
    return iou


def get_batch_statistics(outputs: List[Optional[torch.Tensor]],
                         targets: torch.Tensor,
                         iou_threshold: float,
                         args: Any) -> List[List[np.ndarray]]:
    """
    Compute true positives, predicted scores, and predicted labels per sample.
    Args:
        outputs: List of tensors from NMS, one per image in the batch.
                 Each tensor is (num_preds, 7) with (x1, y1, x2, y2, obj_conf, cls_conf, cls_pred_idx),
                 typically on CPU. cls_conf is the confidence of the predicted class.
        targets: Tensor of shape (total_num_gt_in_batch, 6) with
                 (batch_img_idx, class_id, cx, cy, w, h),
                 typically on GPU. cx,cy,w,h are assumed to be scaled to network input size.
        iou_threshold (float): IoU threshold for a match to be considered a True Positive.
        args (object): Configuration object, used for device and debug flags.

    Returns:
        batch_metrics: List of lists, where each inner list contains:
                       [true_positives_for_sample (np.array),
                        pred_scores_for_sample (np.array),
                        pred_labels_for_sample (np.array)]
    """
    batch_metrics = []
    debug_val_metrics = getattr(args, 'debug_val_metrics', getattr(args, 'debug_nms', False))

    # Ensure targets is on CPU for processing if outputs are on CPU
    # NMS usually outputs to CPU.
    # This assumes 'outputs' list elements are on CPU.
    targets_cpu = targets.cpu()

    for sample_i in range(len(outputs)):  # len(outputs) is batch size
        output_for_sample = outputs[sample_i]

        if output_for_sample is None or output_for_sample.numel() == 0:
            if debug_val_metrics: print(f"  [DEBUG get_batch_statistics] Sample {sample_i}: No NMS outputs or empty predictions.")
            batch_metrics.append([np.array([]), np.array([]), np.array([])])
            continue

        # Predictions for the current sample (from NMS, assumed to be on CPU)
        pred_boxes_x1y1x2y2 = output_for_sample[:, :4]
        pred_obj_conf = output_for_sample[:, 4]
        pred_cls_conf = output_for_sample[:, 5]
        pred_labels_idx = output_for_sample[:, -1]
        pred_final_scores = pred_obj_conf * pred_cls_conf

        true_positives = torch.zeros(pred_boxes_x1y1x2y2.shape[0], device=pred_boxes_x1y1x2y2.device, dtype=torch.bool)

        sample_annotations = targets_cpu[targets_cpu[:, 0] == sample_i]

        if debug_val_metrics:
            print(f"  [DEBUG get_batch_statistics] Sample {sample_i}: Num preds after NMS: {output_for_sample.size(0)}, Num GTs for this sample: {sample_annotations.size(0)}")

        if sample_annotations.numel() == 0:
            batch_metrics.append([true_positives.cpu().numpy(), pred_final_scores.cpu().numpy(), pred_labels_idx.cpu().numpy()])
            continue

        gt_class_labels_for_sample = sample_annotations[:, 1]
        target_boxes_cxcywh_net_input_scale = sample_annotations[:, 2:6]
        target_boxes_x1y1x2y2_net_input_scale = xywh2xyxy(target_boxes_cxcywh_net_input_scale)

        detected_gt_indices = set()
        sorted_indices = torch.argsort(pred_final_scores, descending=True)

        for pred_i_sorted in sorted_indices: # Iterate through predictions by decreasing confidence
            pred_box_single = pred_boxes_x1y1x2y2[pred_i_sorted]
            pred_label_single = pred_labels_idx[pred_i_sorted]

            # Filter GTs by the class of the current prediction
            same_class_gt_mask = (gt_class_labels_for_sample == pred_label_single)

            if not same_class_gt_mask.any():
                if debug_val_metrics: print(f"    Pred (sorted_idx {pred_i_sorted.item()}) class {pred_label_single.item():.0f}: No GTs of this class. FP.")
                continue

            relevant_target_boxes = target_boxes_x1y1x2y2_net_input_scale[same_class_gt_mask]
            original_indices_of_relevant_gts = torch.arange(sample_annotations.size(0))[same_class_gt_mask]

            if relevant_target_boxes.numel() == 0:
                continue

            iou_values = bbox_iou(pred_box_single.unsqueeze(0), relevant_target_boxes, x1y1x2y2=True) # (1, num_relevant_gts)

            if iou_values.numel() > 0:
                best_iou_val_for_pred, best_relevant_gt_idx_in_filtered_list = torch.max(iou_values.squeeze(0), dim=0)
                iou_scalar = best_iou_val_for_pred.item()
                original_gt_idx_of_best_match = original_indices_of_relevant_gts[best_relevant_gt_idx_in_filtered_list.item()].item()

                if iou_scalar >= iou_threshold and original_gt_idx_of_best_match not in detected_gt_indices:
                    true_positives[pred_i_sorted] = True # Mark this prediction as TP
                    detected_gt_indices.add(original_gt_idx_of_best_match)
                    if debug_val_metrics:
                        print(f"    Pred (sorted_idx {pred_i_sorted.item()}) class {pred_label_single.item():.0f}, score {pred_final_scores[pred_i_sorted]:.4f}: MATCHED with original GT idx {original_gt_idx_of_best_match}, IoU: {iou_scalar:.4f}. TP.")
                elif debug_val_metrics:
                    reason = "low IoU" if iou_scalar < iou_threshold else "GT already detected"
                    print(f"    Pred (sorted_idx {pred_i_sorted.item()}) class {pred_label_single.item():.0f}, score {pred_final_scores[pred_i_sorted]:.4f}: No valid GT match ({reason}). Best IoU: {iou_scalar:.4f} with GT orig_idx {original_gt_idx_of_best_match}. FP.")
            elif debug_val_metrics:
                 print(f"    Pred (sorted_idx {pred_i_sorted.item()}) class {pred_label_single.item():.0f}, score {pred_final_scores[pred_i_sorted]:.4f}: No overlapping GTs of same class. FP.")

        batch_metrics.append([true_positives.cpu().numpy(), pred_final_scores.cpu().numpy(), pred_labels_idx.cpu().numpy()])
    return batch_metrics
import numpy as np
import torch  # Add if not already imported in that file


# Assuming bbox_iou is defined in this file or imported correctly
# def bbox_iou(box1, box2, x1y1x2y2=True):
# ... your bbox_iou implementation ...

def get_batch_statistics(outputs, targets, iou_threshold, args):
    """
    Compute true positives, predicted scores and predicted labels per sample.
    outputs: List of tensors from NMS. Each tensor: (x1, y1, x2, y2, object_conf, class_score, class_pred) or None.
    targets: Full batch targets tensor (batch_idx, class_id, x1, y1, x2, y2) - ensure this format.
             Coordinates should be absolute pixel values.
    iou_threshold (float): Minimum IoU for a match to be considered a True Positive.
    args (object): Arguments object, used for debug flag.
    """
    debug_mode = getattr(args, 'debug_decoder', False)
    batch_metrics = []

    # Determine the actual batch size processed by NMS
    # This could be different from args.batch_size if it's the last, smaller batch.
    num_images_in_batch = len(outputs)

    for sample_i in range(num_images_in_batch):
        if debug_mode:
            current_epoch_for_debug = getattr(args, 'current_epoch', 'N/A')
            current_step_for_debug = getattr(args, 'current_step', 'N/A')
            print(
                f"\n--- [DEBUG get_batch_statistics] Epoch: {current_epoch_for_debug}, Step: {current_step_for_debug} ---")
            print(f"--- [DEBUG get_batch_statistics] Processing Sample {sample_i} ---")

        if outputs[sample_i] is None:
            if debug_mode:
                print(f"  [DEBUG get_batch_statistics] Sample {sample_i}: No detections after NMS.")
            # Still append empty stats for this sample to maintain batch structure for zipping later
            batch_metrics.append(
                [np.array([], dtype=np.int8), np.array([], dtype=np.float32), np.array([], dtype=np.int32)])
            continue

        output_per_image = outputs[sample_i]  # Detections for this sample, should be on CUDA from NMS

        # It's generally safer to move to CPU for numpy operations and if bbox_iou might expect CPU.
        # However, if bbox_iou is a pure PyTorch CUDA-enabled function, this can be optimized.
        # For robust debugging, let's work with CPU tensors here for now.
        pred_boxes = output_per_image[:, :4].cpu()
        pred_scores = output_per_image[:, 4].cpu()
        pred_labels = output_per_image[:, -1].cpu()

        true_positives = np.zeros(pred_boxes.shape[0], dtype=np.int8)

        # Get ground truths for this specific sample from the batch targets tensor
        # Ensure targets tensor is on the same device as operations or move explicitly
        sample_annotations = targets[targets[:, 0] == sample_i].cpu()  # Move to CPU for consistency here

        if debug_mode:
            print(
                f"  [DEBUG get_batch_statistics] Sample {sample_i}: Num Detections (after NMS, before matching): {output_per_image.shape[0]}")
            print(
                f"  [DEBUG get_batch_statistics] Sample {sample_i}: Num GT Annotations: {sample_annotations.shape[0]}")

        if sample_annotations.shape[0] == 0:  # No GT for this image
            if debug_mode:
                print(
                    f"  [DEBUG get_batch_statistics] Sample {sample_i}: No GT annotations. All {output_per_image.shape[0]} detections are False Positives.")
            batch_metrics.append([true_positives, pred_scores.numpy(), pred_labels.numpy()])
            continue

        target_sample_labels = sample_annotations[:, 1]
        target_sample_boxes = sample_annotations[:, 2:]  # These are x1,y1,x2,y2 (absolute)

        if debug_mode:
            if target_sample_labels.numel() > 0:
                print(
                    f"  [DEBUG get_batch_statistics] Sample {sample_i}: Unique GT labels: {torch.unique(target_sample_labels).tolist()}")
            else:
                print(
                    f"  [DEBUG get_batch_statistics] Sample {sample_i}: No GT labels (target_sample_labels is empty).")

        detected_gt_indices = []  # Keep track of detected GT box indices to avoid multiple matches to the same GT

        for pred_i in range(len(pred_boxes)):
            pred_box_current = pred_boxes[pred_i].unsqueeze(0)  # Shape (1, 4), on CPU
            pred_label_current = pred_labels[pred_i]

            if debug_mode:
                print(
                    f"    [DEBUG get_batch_statistics] Pred {pred_i}: Box={pred_box_current.tolist()}, Label={pred_label_current.item():.0f}, Score={pred_scores[pred_i].item():.4f}")

            if target_sample_boxes.numel() == 0:
                if debug_mode: print(f"      No GT boxes to match against for pred {pred_i}.")
                continue

            # Calculate IoU of this predicted box with all GT boxes for this sample
            # Ensure bbox_iou can handle CPU tensors if pred_box_current and target_sample_boxes are CPU
            # bbox_iou should return shape (1, num_gts_for_sample)
            ious_with_gts = bbox_iou(pred_box_current, target_sample_boxes, x1y1x2y2=True)

            if ious_with_gts.numel() == 0:
                if debug_mode: print(f"      bbox_iou returned empty for pred {pred_i}.")
                continue

            # For a single prediction, find the GT box it has the highest IoU with.
            # ious_with_gts is (1, M). max(1) reduces along the GT dimension.
            best_iou_val_for_pred, best_gt_idx_for_pred_tensor = ious_with_gts.max(1)

            # Convert to scalar values
            best_iou_scalar = best_iou_val_for_pred.item()
            best_gt_idx_scalar = best_gt_idx_for_pred_tensor.item()

            if debug_mode:
                print(f"      IoUs with all GTs for this pred: {ious_with_gts.tolist()}")  # Will be [[iou1, iou2, ...]]
                print(
                    f"      Best IoU: {best_iou_scalar:.4f} with GT (original index in sample_annotations) {best_gt_idx_scalar}")

            if best_iou_scalar >= iou_threshold:
                matched_gt_original_label = target_sample_labels[best_gt_idx_scalar]
                if debug_mode:
                    print(
                        f"        IoU ({best_iou_scalar:.4f}) >= threshold ({iou_threshold}). Matched GT label: {matched_gt_original_label.item():.0f}")

                if pred_label_current.item() == matched_gt_original_label.item():
                    if debug_mode: print(
                        f"        Pred label ({pred_label_current.item():.0f}) == Matched GT label ({matched_gt_original_label.item():.0f}).")
                    if best_gt_idx_scalar not in detected_gt_indices:
                        true_positives[pred_i] = 1
                        detected_gt_indices.append(best_gt_idx_scalar)
                        if debug_mode: print(
                            f"        ----> MARKED AS TRUE POSITIVE. GT index {best_gt_idx_scalar} now detected.")
                    else:
                        if debug_mode: print(
                            f"        ----> NOT TP: GT index {best_gt_idx_scalar} was already detected by a higher scoring pred.")
                else:
                    if debug_mode: print(
                        f"        ----> NOT TP: Label mismatch (Pred: {pred_label_current.item():.0f}, GT: {matched_gt_original_label.item():.0f}).")
            else:
                if debug_mode: print(
                    f"        ----> NOT TP: Best IoU ({best_iou_scalar:.4f}) < threshold ({iou_threshold}).")

        batch_metrics.append([true_positives, pred_scores.numpy(), pred_labels.numpy()])
        if debug_mode:
            print(
                f"  [DEBUG get_batch_statistics] Sample {sample_i} results: TPs={true_positives.sum()}/{len(true_positives)} preds, {len(detected_gt_indices)} unique GTs matched.")
            print(f"--- [DEBUG get_batch_statistics] END Sample {sample_i} ---")

    return batch_metrics
'''
def get_batch_statistics(outputs, targets, iou_threshold, args):
    """
    Compute true positives, predicted scores and predicted labels per sample
    outputs: (x1, y1, x2, y2, conf, cls_conf, cls_pred)
    """
    batch_metrics = list()
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_thetas = output[:, 4]  # this is dummy value will not be used
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label, pred_theta) in enumerate(zip(pred_boxes, pred_labels, pred_thetas)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
'''

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


import torch , math


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU (Intersection over Union) between box1 and box2.
    box1: (N, 4) tensor, predicted boxes
    box2: (M, 4) tensor, target boxes
    x1y1x2y2: If True, boxes are in [x1, y1, x2, y2] format, otherwise [cx, cy, w, h]
    GIoU, DIoU, CIoU: Flags for different IoU variants (set to False for standard IoU)
    eps: Small epsilon to prevent division by zero.

    Returns:
        (N, M) tensor of IoUs between all pairs of boxes.
    """
    # Ensure both boxes are on the same device
    device = box1.device
    box2 = box2.to(device)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
    else:  # transform from cxcywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Transpose box2 to allow broadcasting for intersection calculation
    # b1: (N, 4) -> b1_x1: (N, 1)
    # b2: (M, 4) -> b2_x1: (M, 1) -> b2_x1.T: (1, M)

    # Intersection area
    inter_rect_x1 = torch.max(b1_x1, b2_x1.T)  # Error was likely here if devices mismatched
    inter_rect_y1 = torch.max(b1_y1, b2_y1.T)
    inter_rect_x2 = torch.min(b1_x2, b2_x2.T)
    inter_rect_y2 = torch.min(b1_y2, b2_y2.T)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area.T - inter_area + eps

    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        # Enclosing box
        cw = torch.max(b1_x2, b2_x2.T) - torch.min(b1_x1, b2_x1.T)  # convex width
        ch = torch.max(b1_y2, b2_y2.T) - torch.min(b1_y1, b2_y1.T)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1.T + b2_x2.T - b1_x1 - b1_x2) ** 2 +
                    (b2_y1.T + b2_y2.T - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # Complete IoU
                v = (4 / math.pi ** 2) * torch.pow(torch.atan((b2_x2.T - b2_x1.T) / (b2_y2.T - b2_y1.T + eps)) -
                                                   torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union_area) / c_area  # GIoU
    return iou  # Standard IoU

'''
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou
'''

def get_contour(b_box, theta):
    # b_box in shape of (x_min, y_min, x_max, y_max)
    cont = shapely.geometry.box(b_box[0], b_box[1], b_box[2], b_box[3])
    rot_cont = shapely.affinity.rotate(cont, theta * -1, use_radians=False)
    return rot_cont

'''
#old nms , works well
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    :return Detections with the shape (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output
'''
'''
#new NSM - yolo7, works fine _> mAP = 0.38
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, args=None):  # Added args for debug flag
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections. (User's version with merging)

    Args:
        prediction (torch.Tensor): Tensor of shape (batch_size, num_total_predictions, 5 + num_classes)
                                   Format: (x_center, y_center, width, height, object_conf, class_probs...)
        conf_thres (float): Confidence threshold for filtering initial detections.
        nms_thres (float): IoU threshold for Non-Maximum Suppression.
        args (object, optional): Arguments object, used for debug_nms flag.

    Returns:
        list: A list of tensors, one for each image in the batch.
              Each tensor contains filtered detections with the shape (num_filtered_detections, 7).
              Format: (x1, y1, x2, y2, object_conf, class_score, class_pred)
              Returns None for images with no detections after filtering.
    """
    debug_nms = False
    if args and hasattr(args, 'debug_nms'):
        debug_nms = args.debug_nms
    elif args and hasattr(args, 'debug_decoder'):  # Fallback to general decoder debug flag
        debug_nms = args.debug_decoder

    if prediction is None:
        return []

    prediction_clone = prediction.clone()
    prediction_clone[..., :4] = xywh2xyxy(prediction_clone[..., :4])

    output = [None for _ in range(prediction_clone.size(0))]

    if debug_nms:
        print(f"\n--- [DEBUG NMS Start] --- conf_thres: {conf_thres}, nms_thres: {nms_thres}")
        print(f"[DEBUG NMS] Input prediction shape: {prediction.shape}")

    for image_i, image_pred_init in enumerate(prediction_clone):
        if debug_nms:
            print(f"\n[DEBUG NMS] Processing Image: {image_i}, Initial image_pred shape: {image_pred_init.shape}")
            if image_pred_init.numel() > 0:
                print(
                    f"[DEBUG NMS] Initial image_pred confs (top 5): {torch.topk(image_pred_init[:, 4].flatten(), min(5, image_pred_init.size(0))).values.tolist()}")

        # Filter out confidence scores below threshold
        image_pred_conf_filtered = image_pred_init[image_pred_init[:, 4] >= conf_thres]

        if debug_nms:
            print(
                f"[DEBUG NMS] Image {image_i}: After conf_thres ({conf_thres}), shape: {image_pred_conf_filtered.shape}")

        if not image_pred_conf_filtered.size(0):
            if debug_nms: print(f"[DEBUG NMS] Image {image_i}: No detections after conf_thres. Skipping.")
            continue

        # Object confidence times class confidence
        # image_pred_conf_filtered[:, 5:] are class probabilities
        max_class_scores, _ = image_pred_conf_filtered[:, 5:].max(1)
        score = image_pred_conf_filtered[:, 4] * max_class_scores

        if debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: Calculated scores (obj_conf * class_conf) shape: {score.shape}")
            if score.numel() > 0:
                print(
                    f"[DEBUG NMS] Image {image_i}: Top 5 combined scores: {torch.topk(score.flatten(), min(5, score.numel())).values.tolist()}")

        # Sort by it
        sorted_indices = (-score).argsort()
        image_pred_sorted = image_pred_conf_filtered[sorted_indices]

        class_confs_sorted, class_preds_sorted = image_pred_sorted[:, 5:].max(1, keepdim=True)

        detections = torch.cat((image_pred_sorted[:, :5], class_confs_sorted.float(), class_preds_sorted.float()), 1)

        if debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: Detections tensor for NMS loop (shape {detections.shape}):")
            # print(detections.cpu().numpy()) # Potentially very verbose

        keep_boxes = []
        loop_count = 0
        while detections.size(0):
            if debug_nms:
                print(
                    f"  [DEBUG NMS Loop Iter {loop_count}] Image {image_i}: Detections remaining: {detections.size(0)}")
                print(
                    f"  [DEBUG NMS Loop Iter {loop_count}] Current best detection (score: {detections[0, 4].item() * detections[0, 5].item():.4f}): {detections[0].tolist()}")

            iou_scores_for_best = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4], x1y1x2y2=True)
            if debug_nms:
                print(f"    [DEBUG NMS Loop Iter {loop_count}] IoU scores with best: {iou_scores_for_best.tolist()}")

            large_overlap = iou_scores_for_best > nms_thres  # Shape (1, num_remaining_detections)
            label_match = detections[0, -1] == detections[:, -1]  # Shape (num_remaining_detections,)

            invalid_2d = large_overlap & label_match.unsqueeze(0)
            invalid_1d = invalid_2d.squeeze(0)

            if debug_nms:
                print(
                    f"    [DEBUG NMS Loop Iter {loop_count}] large_overlap (raw from iou > thres): {large_overlap.tolist()}")
                print(f"    [DEBUG NMS Loop Iter {loop_count}] label_match: {label_match.tolist()}")
                print(
                    f"    [DEBUG NMS Loop Iter {loop_count}] invalid_1d (to be merged/removed): {invalid_1d.tolist()} (sum: {invalid_1d.sum().item()})")

            # --- Box Merging / Weighted Averaging ---
            if invalid_1d.any():
                selected_for_merging = detections[invalid_1d]

                if debug_nms and selected_for_merging.numel() > 0:
                    print(
                        f"      [DEBUG NMS Loop Iter {loop_count}] Selected for merging (shape {selected_for_merging.shape}): {selected_for_merging[:, :4].tolist()}")

                if selected_for_merging.numel() > 0:
                    weights = selected_for_merging[:, 4:5]  # obj_conf of these boxes
                    if debug_nms:
                        print(f"        [DEBUG NMS Loop Iter {loop_count}] Weights for merging: {weights.T.tolist()}")

                    if weights.sum() > 1e-7:
                        merged_box_coords = (weights * selected_for_merging[:, :4]).sum(0) / weights.sum()
                        detections[0, :4] = merged_box_coords
                        if debug_nms:
                            print(
                                f"        [DEBUG NMS Loop Iter {loop_count}] Box merged. New coords for detections[0]: {merged_box_coords.tolist()}")
                    elif debug_nms:
                        print(f"        [DEBUG NMS Loop Iter {loop_count}] Weights sum too small, no merge for coords.")
                elif debug_nms:
                    print(
                        f"      [DEBUG NMS Loop Iter {loop_count}] No boxes actually selected by invalid_1d for merging (selected_for_merging is empty).")

            # --- End of Box Merging ---

            keep_boxes.append(detections[0].clone())  # Clone to avoid modification if detections is further sliced
            if debug_nms:
                print(f"    [DEBUG NMS Loop Iter {loop_count}] Added to keep_boxes: {keep_boxes[-1].tolist()}")

            # Remove all boxes that were marked in invalid_1d for the next iteration.
            detections = detections[~invalid_1d]
            loop_count += 1
            if loop_count > 1000:  # Safety break for very long NMS loops during debugging
                if debug_nms: print(
                    f"    [DEBUG NMS WARNING] Exiting NMS loop due to excessive iterations for image {image_i}")
                break

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
            if debug_nms:
                print(f"[DEBUG NMS] Image {image_i}: Final kept boxes: {output[image_i].shape[0]}")
        elif debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: No boxes kept after NMS loop.")

    if debug_nms:
        print(f"--- [DEBUG NMS End] ---")

    return output
'''
import torch
import math  # For bbox_iou CIoU if used, though not directly in NMS here


# Assuming xywh2xyxy and bbox_iou are defined elsewhere and imported correctly
# from .utils import xywh2xyxy, bbox_iou # Or your specific import path

# --- Placeholder for utility functions if they are not in the provided snippet ---
# You should have your actual implementations for these.
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # Assumes x is a torch tensor
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU (Intersection over Union) between box1 and box2.
    box1: (N, 4) tensor, predicted boxes
    box2: (M, 4) tensor, target boxes
    x1y1x2y2: If True, boxes are in [x1, y1, x2, y2] format, otherwise [cx, cy, w, h]
    GIoU, DIoU, CIoU: Flags for different IoU variants (set to False for standard IoU)
    eps: Small epsilon to prevent division by zero.

    Returns:
        (N, M) tensor of IoUs between all pairs of boxes.
    """
    # Ensure both boxes are on the same device
    device = box1.device
    box2 = box2.to(device)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
    else:  # transform from cxcywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1.T)
    inter_rect_y1 = torch.max(b1_y1, b2_y1.T)
    inter_rect_x2 = torch.min(b1_x2, b2_x2.T)
    inter_rect_y2 = torch.min(b1_y2, b2_y2.T)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area.T - inter_area + eps
    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2.T) - torch.min(b1_x1, b2_x1.T)
        ch = torch.max(b1_y2, b2_y2.T) - torch.min(b1_y1, b2_y1.T)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1.T + b2_x2.T - b1_x1 - b1_x2) ** 2 +
                    (b2_y1.T + b2_y2.T - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan((b2_x2.T - b2_x1.T) / (b2_y2.T - b2_y1.T + eps)) -
                                                   torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps
            return iou - (c_area - union_area) / c_area
    return iou


# --- End Placeholder ---


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, args=None):  # Added args for debug flag
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.

    Args:
        prediction (torch.Tensor): Tensor of shape (batch_size, num_total_predictions, 5 + num_classes)
                                   Format: (x_center, y_center, width, height, object_conf, class_probs...)
        conf_thres (float): Confidence threshold for filtering initial detections.
        nms_thres (float): IoU threshold for Non-Maximum Suppression.
        args (object, optional): Arguments object, used for debug_nms flag.

    Returns:
        list: A list of tensors, one for each image in the batch.
              Each tensor contains filtered detections with the shape (num_filtered_detections, 7).
              Format: (x1, y1, x2, y2, object_conf, class_score, class_pred)
              Returns None for images with no detections after filtering.
    """
    debug_nms = False
    if args and hasattr(args, 'debug_nms'):
        debug_nms = args.debug_nms
    elif args and hasattr(args, 'debug_decoder'):  # Fallback to general decoder debug flag
        debug_nms = args.debug_decoder

    if prediction is None:
        return []

    prediction_clone = prediction.clone()
    prediction_clone[..., :4] = xywh2xyxy(prediction_clone[..., :4])

    output = [None for _ in range(prediction_clone.size(0))]

    if debug_nms:
        print(f"\n--- [DEBUG NMS Start] --- conf_thres: {conf_thres}, nms_thres: {nms_thres}")
        print(f"[DEBUG NMS] Input prediction shape: {prediction.shape}")

    for image_i, image_pred_init in enumerate(prediction_clone):
        if debug_nms:
            print(f"\n[DEBUG NMS] Processing Image: {image_i}, Initial image_pred shape: {image_pred_init.shape}")
            if image_pred_init.numel() > 0:
                print(
                    f"[DEBUG NMS] Initial image_pred confs (top 5): {torch.topk(image_pred_init[:, 4].flatten(), min(5, image_pred_init.size(0))).values.tolist()}")

        # Filter out confidence scores below threshold
        image_pred_conf_filtered = image_pred_init[image_pred_init[:, 4] >= conf_thres]

        if debug_nms:
            print(
                f"[DEBUG NMS] Image {image_i}: After conf_thres ({conf_thres}), shape: {image_pred_conf_filtered.shape}")

        if not image_pred_conf_filtered.size(0):
            if debug_nms: print(f"[DEBUG NMS] Image {image_i}: No detections after conf_thres. Skipping.")
            continue

        # Object confidence times class confidence
        max_class_scores, _ = image_pred_conf_filtered[:, 5:].max(1)
        score = image_pred_conf_filtered[:, 4] * max_class_scores

        if debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: Calculated scores (obj_conf * class_conf) shape: {score.shape}")
            if score.numel() > 0:
                print(
                    f"[DEBUG NMS] Image {image_i}: Top 5 combined scores: {torch.topk(score.flatten(), min(5, score.numel())).values.tolist()}")

        # Sort by it
        sorted_indices = (-score).argsort()
        image_pred_sorted = image_pred_conf_filtered[sorted_indices]

        class_confs_sorted, class_preds_sorted = image_pred_sorted[:, 5:].max(1, keepdim=True)

        detections = torch.cat((image_pred_sorted[:, :5], class_confs_sorted.float(), class_preds_sorted.float()), 1)

        if debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: Detections tensor for NMS loop (shape {detections.shape}):")

        keep_boxes = []
        loop_count = 0
        while detections.size(0):
            if debug_nms:
                print(
                    f"  [DEBUG NMS Loop Iter {loop_count}] Image {image_i}: Detections remaining: {detections.size(0)}")
                print(
                    f"  [DEBUG NMS Loop Iter {loop_count}] Current best detection (score: {detections[0, 4].item() * detections[0, 5].item():.4f}): {detections[0].tolist()}")

            iou_scores_for_best = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4], x1y1x2y2=True)
            if debug_nms:
                print(f"    [DEBUG NMS Loop Iter {loop_count}] IoU scores with best: {iou_scores_for_best.tolist()}")

            large_overlap = iou_scores_for_best > nms_thres
            label_match = detections[0, -1] == detections[:, -1]

            invalid_2d = large_overlap & label_match.unsqueeze(0)
            invalid_1d = invalid_2d.squeeze(0)

            if debug_nms:
                print(
                    f"    [DEBUG NMS Loop Iter {loop_count}] large_overlap (raw from iou > thres): {large_overlap.tolist()}")
                print(f"    [DEBUG NMS Loop Iter {loop_count}] label_match: {label_match.tolist()}")
                print(
                    f"    [DEBUG NMS Loop Iter {loop_count}] invalid_1d (to be merged/removed): {invalid_1d.tolist()} (sum: {invalid_1d.sum().item()})")

            # --- Box Merging / Weighted Averaging (User's Original Logic) ---
            # This section is now COMMENTED OUT to perform standard greedy NMS
            """
            if invalid_1d.any(): 
                selected_for_merging = detections[invalid_1d] 

                if debug_nms and selected_for_merging.numel() > 0:
                    print(f"      [DEBUG NMS Loop Iter {loop_count}] Selected for merging (shape {selected_for_merging.shape}): {selected_for_merging[:, :4].tolist()}")

                if selected_for_merging.numel() > 0:
                    weights = selected_for_merging[:, 4:5] 
                    if debug_nms:
                        print(f"        [DEBUG NMS Loop Iter {loop_count}] Weights for merging: {weights.T.tolist()}")

                    if weights.sum() > 1e-7: 
                        merged_box_coords = (weights * selected_for_merging[:, :4]).sum(0) / weights.sum()
                        detections[0, :4] = merged_box_coords
                        if debug_nms:
                            print(f"        [DEBUG NMS Loop Iter {loop_count}] Box merged. New coords for detections[0]: {merged_box_coords.tolist()}")
                    elif debug_nms:
                        print(f"        [DEBUG NMS Loop Iter {loop_count}] Weights sum too small, no merge for coords.")
                elif debug_nms:
                     print(f"      [DEBUG NMS Loop Iter {loop_count}] No boxes actually selected by invalid_1d for merging (selected_for_merging is empty).")
            """
            # --- End of Box Merging ---

            keep_boxes.append(detections[0].clone())
            if debug_nms:
                print(f"    [DEBUG NMS Loop Iter {loop_count}] Added to keep_boxes: {keep_boxes[-1].tolist()}")

            # For standard NMS, we remove the current best box (detections[0]) and all other boxes
            # that had high IoU with it AND the same label.
            # The `invalid_1d` mask already marks detections[0] as True (due to self-overlap and label match)
            # and other overlapping boxes. So, `~invalid_1d` keeps the ones that are NOT these.
            detections = detections[~invalid_1d]
            loop_count += 1
            if loop_count > 1000:
                if debug_nms: print(
                    f"    [DEBUG NMS WARNING] Exiting NMS loop due to excessive iterations for image {image_i}")
                break

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
            if debug_nms:
                print(f"[DEBUG NMS] Image {image_i}: Final kept boxes: {output[image_i].shape[0]}")
        elif debug_nms:
            print(f"[DEBUG NMS] Image {image_i}: No boxes kept after NMS loop.")

    if debug_nms:
        print(f"--- [DEBUG NMS End] ---")

    return output

#new implementation for rawbits , it works and stabilize after long time ( yolo v7)
def build_targets(raw_predictions, target, anchors, ignore_thres, args):
    """
    Builds target tensors and extracts raw predictions for YOLO loss calculation.

    Args:
        raw_predictions (torch.Tensor): Raw predictions from the final convolution of YOLOLayer.
                                        Shape: (batch_size, num_anchors * (5 + num_classes), grid_h, grid_w)
        target (torch.Tensor): Ground truth targets.
                               Shape: (num_targets, 6)
                               Format: [batch_index, class_id, x_center, y_center, width, height]
                               Coordinates are relative to the image size (0-1).
        anchors (torch.Tensor): Anchor box dimensions (scaled to grid size).
                                Shape: (num_anchors, 2)
        ignore_thres (float): IoU threshold for ignoring negative samples.
        args (object): Input arguments from params file (must contain 'device', 'input_width', 'input_height').

    Returns:
        tuple: A tuple containing the target tensors, raw predictions for loss, and auxiliary metrics:
               (obj_mask, noobj_mask, tx, ty, tw, th, tcls,
                raw_pred_x_obj, raw_pred_y_obj, raw_pred_w_obj, raw_pred_h_obj, raw_pred_obj_conf_obj, raw_pred_cls_obj,
                raw_pred_obj_conf_noobj,
                iou_scores, class_mask)
    """
    device = args.device

    nB = raw_predictions.size(0)  # Batch size
    nA = anchors.size(0) # Number of anchors per scale
    nG_y = raw_predictions.size(2)  # Grid size vertical
    nG_x = raw_predictions.size(3)  # Grid size horizontal
    nC = (raw_predictions.size(1) // nA) - 5 # Number of classes

    # Reshape raw predictions to (B, nA, nG_y, nG_x, 5 + nC)
    raw_predictions_reshaped = raw_predictions.view(nB, nA, 5 + nC, nG_y, nG_x).permute(0, 1, 3, 4, 2).contiguous()

    # Decode raw predictions to get pred_boxes and pred_cls for IoU/mask calculations
    # These are decoded values, used *only* for target assignment logic (IoU, class_mask)
    pred_x = torch.sigmoid(raw_predictions_reshaped[..., 0])
    pred_y = torch.sigmoid(raw_predictions_reshaped[..., 1])
    pred_w_raw = raw_predictions_reshaped[..., 2] # Keep raw for decoding w/ anchors
    pred_h_raw = raw_predictions_reshaped[..., 3] # Keep raw for decoding h/ anchors
    pred_conf = torch.sigmoid(raw_predictions_reshaped[..., 4])
    pred_cls = torch.sigmoid(raw_predictions_reshaped[..., 5:])

    # Compute decoded pred_boxes in grid scale for IoU calculation
    grid_x = torch.arange(nG_x, device=device).view(1, 1, 1, nG_x).expand(nB, nA, nG_y, nG_x)
    grid_y = torch.arange(nG_y, device=device).view(1, 1, nG_y, 1).expand(nB, nA, nG_y, nG_x)
    anchor_w = anchors[:, 0].view(1, nA, 1, 1).expand(nB, nA, nG_y, nG_x)
    anchor_h = anchors[:, 1].view(1, nA, 1, 1).expand(nB, nA, nG_y, nG_x)

    # Decoded pred_boxes in grid scale (aligned with YOLOv7 decoding)
    pred_boxes_grid_scale = torch.empty_like(raw_predictions_reshaped[..., :4], device=device) # Ensure on device
    pred_boxes_grid_scale[..., 0] = (pred_x * 2. - 0.5 + grid_x)
    pred_boxes_grid_scale[..., 1] = (pred_y * 2. - 0.5 + grid_y)
    pred_boxes_grid_scale[..., 2] = (torch.sigmoid(pred_w_raw) * 2).pow(2) * anchor_w
    pred_boxes_grid_scale[..., 3] = (torch.sigmoid(pred_h_raw) * 2).pow(2) * anchor_h


    # Output tensors initialized on the correct device
    obj_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.bool, device=device)
    noobj_mask = torch.ones([nB, nA, nG_y, nG_x], dtype=torch.bool, device=device)
    class_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    iou_scores = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    tx = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    ty = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    tw = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    th = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float, device=device)
    tcls = torch.zeros([nB, nA, nG_y, nG_x, nC], dtype=torch.float, device=device)

    # Handle negative samples (targets with zero or negative width/height)
    # Filter targets where width or height is zero or less
    valid_targets = target[(target[:, 4] > 0) & (target[:, 5] > 0)]

    if valid_targets.nelement() == 0:
        # If no valid targets, return masks and zero tensors
        # Also return empty raw prediction tensors for obj and noobj
        empty_raw_obj = torch.empty(0, 4, device=device) # x, y, w, h
        empty_raw_obj_conf = torch.empty(0, device=device)
        empty_raw_cls_obj = torch.empty(0, nC, device=device)
        raw_pred_obj_conf_noobj = raw_predictions_reshaped[..., 4][noobj_mask] # Still get noobj conf

        return (obj_mask, noobj_mask, tx, ty, tw, th, tcls,
                empty_raw_obj[:, 0], empty_raw_obj[:, 1], empty_raw_obj[:, 2], empty_raw_obj[:, 3], empty_raw_obj_conf, empty_raw_cls_obj,
                raw_pred_obj_conf_noobj,
                iou_scores, class_mask)

    # Scale target box coordinates from image relative (0-1) to grid scale
    # target_boxes shape: (num_valid_targets, 4) - [gx, gy, gw, gh] in grid scale
    # Need image dimensions to calculate stride and convert target boxes to grid scale
    img_w, img_h = args.input_width, args.input_height
    stride_w = img_w / nG_x
    stride_h = img_h / nG_y

    target_boxes_grid_scale = torch.cat(((valid_targets[:, 2] * nG_x).unsqueeze(1),
                                         (valid_targets[:, 3] * nG_y).unsqueeze(1),
                                         (valid_targets[:, 4] * nG_x).unsqueeze(1),
                                         (valid_targets[:, 5] * nG_y).unsqueeze(1)), 1)


    # Get grid coordinates and dimensions for valid targets
    gxy = target_boxes_grid_scale[:, :2] # (num_valid_targets, 2) - [gx, gy] in grid scale
    gwh = target_boxes_grid_scale[:, 2:] # (num_valid_targets, 2) - [gw, gh] in grid scale
    gi, gj = gxy.long().t() # (2, num_valid_targets) - integer grid cell indices [gi, gj]

    # Clamp grid cell indices to boundaries
    gi = torch.clamp(gi, 0, nG_x - 1)
    gj = torch.clamp(gj, 0, nG_y - 1)

    # Find the best anchor for each ground truth box based on IoU of width/height
    # ious shape is (num_anchors, num_valid_targets)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0) # best_n shape: (num_valid_targets,) - index of best anchor for each target

    # Separate target values for assigned objects
    # Use valid_targets[:, :2].long().t() to get batch index and class label
    b, target_labels = valid_targets[:, :2].long().t() # b shape: (num_valid_targets,), target_labels shape: (num_valid_targets,)

    # Set obj_mask and noobj_mask for assigned objects
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold for any anchor in the cell
    # Iterate through valid targets and their corresponding grid cells
    for i, target_iou_with_all_anchors in enumerate(ious.t()):
         # target_iou_with_all_anchors shape: (num_anchors,) - IoU of current target with all anchors
         # For the grid cell (b[i], gj[i], gi[i]), set noobj_mask to 0 for anchors where IoU > ignore_thres
         noobj_mask[b[i], target_iou_with_all_anchors > ignore_thres, gj[i], gi[i]] = 0

    # --- Target Value Calculations (YOLOv7 Decoding Alignment) ---
    # Calculate target x, y offsets relative to the grid cell center (0-1)
    # gx, gy are target centers in grid scale. gi, gj are integer grid cell indices.
    # gx - gi gives the offset from the top-left of the cell. Add 0.5 to shift to center.
    # Divide by 2 because the decoding is sigmoid(tx)*2, mapping 0-1 to 0-2.
    # Clamp argument to logit to prevent infinity
    tx[b, best_n, gj, gi] = torch.logit(torch.clamp((gxy[:, 0] - gi + 0.5) / 2, min=1e-6, max=1.0 - 1e-6)) # Adjusted clamping
    ty[b, best_n, gj, gi] = torch.logit(torch.clamp((gxy[:, 1] - gj + 0.5) / 2, min=1e-6, max=1.0 - 1e-6)) # Adjusted clamping

    # Calculate target w, h scaling relative to anchor dimensions
    # gw, gh are target dimensions in grid scale. anchors[best_n] are best anchor dimensions in grid scale.
    # gw / anchors[best_n][:, 0] is the ratio. Take sqrt because decoding is (sigmoid(tw)*2)^2.
    # Divide by 2 because decoding is sigmoid(tw)*2.
    # Clamp argument to logit to prevent infinity
    tw[b, best_n, gj, gi] = torch.logit(torch.clamp(torch.sqrt(gwh[:, 0] / anchors[best_n][:, 0]) / 2, min=1e-6, max=1.0 - 1e-6)) # Adjusted clamping
    th[b, best_n, gj, gi] = torch.logit(torch.clamp(torch.sqrt(gwh[:, 1] / anchors[best_n][:, 1]) / 2, min=1e-6, max=1.0 - 1e-6)) # Adjusted clamping

    # One-hot encoding of target class
    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness (for metrics)
    # Check if the predicted class (argmax of sigmoid) matches the target class for assigned objects
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    # Compute IoU between predicted box and target box for assigned objects (for metrics/IoU loss)
    # Use the decoded pred_boxes_grid_scale calculated earlier
    # Ensure target_boxes_grid_scale used here are the grid-scaled ones calculated earlier
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes_grid_scale[b, best_n, gj, gi], target_boxes_grid_scale[torch.arange(len(b)), :], x1y1x2y2=False)

    # --- Extract Raw Predictions for Loss Calculation ---
    # Get the raw predictions corresponding to the assigned objects (obj_mask)
    raw_pred_obj_samples = raw_predictions_reshaped[obj_mask]
    raw_pred_x_obj = raw_pred_obj_samples[:, 0]
    raw_pred_y_obj = raw_pred_obj_samples[:, 1]
    raw_pred_w_obj = raw_pred_obj_samples[:, 2]
    raw_pred_h_obj = raw_pred_obj_samples[:, 3]
    raw_pred_obj_conf_obj = raw_pred_obj_samples[:, 4]
    raw_pred_cls_obj = raw_pred_obj_samples[:, 5:]

    # Get the raw objectness predictions corresponding to the non-assigned objects (noobj_mask)
    raw_pred_obj_conf_noobj = raw_predictions_reshaped[..., 4][noobj_mask]


    return (obj_mask, noobj_mask, tx, ty, tw, th, tcls,
            raw_pred_x_obj, raw_pred_y_obj, raw_pred_w_obj, raw_pred_h_obj, raw_pred_obj_conf_obj, raw_pred_cls_obj,
            raw_pred_obj_conf_noobj,
            iou_scores, class_mask)



'''''''''
#old implementation + normalization, works fine and better 
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, args):
    """
    :param pred_boxes:
    :param pred_cls:
    :param target: (img_id, Class_type, x, y ,w ,h, yaw [if exist])
    :param anchors:
    :param ignore_thres:
    :param args: input arguments from params file
    :return:
    """
    device = args.device


    nB = pred_boxes.size(0)  # Batch size
    nA = pred_boxes.size(1)  # Anchor size
    nC = pred_cls.size(-1)  # Number of classes
    nG_y = pred_boxes.size(2)  # Grid size vertical
    nG_x = pred_boxes.size(3)  # Grid size horizontal


    # Output tensors
    obj_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device)
    noobj_mask = torch.ones([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device=obj_mask.device)
    class_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    iou_scores = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tx = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    ty = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tw = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    th = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tcls = torch.zeros([nB, nA, nG_y, nG_x, nC], dtype=torch.float).to(device=obj_mask.device)


    # Handel negative samples
    target = target[target[:, 4] > 0]
    if target.nelement() == 0:
        tconf = obj_mask.float()
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


    # Normalize pred_boxes to 0-1 range
    pred_boxes_normalized = pred_boxes.clone()
    pred_boxes_normalized[..., 0] /= nG_x
    pred_boxes_normalized[..., 1] /= nG_y
    pred_boxes_normalized[..., 2] /= nG_x
    pred_boxes_normalized[..., 3] /= nG_y


    # Assign each object to corresponding grid cell
    target_boxes = torch.cat(((target[:, 2] * nG_x).unsqueeze(1), (target[:, 3] * nG_y).unsqueeze(1),
                              (target[:, 4] * nG_x).unsqueeze(1), (target[:, 5] * nG_y).unsqueeze(1)), 1)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    # ious shape is (3, number_of_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()


    # Handel corner cases(boundaries cases)
    gi = torch.clamp(gi, 0, nG_x - 1)
    gj = torch.clamp(gj, 0, nG_y - 1)


    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0


    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0


    # Normalize coordinates relative to the cell (x & y in range [0,1])
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # Assign the class type
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes_normalized[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)


    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

'''''''''

'''''''''
#original implementation , works fine
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, args):
    """
    :param pred_boxes:
    :param pred_cls:
    :param target: (img_id, Class_type, x, y ,w ,h, yaw [if exist])
    :param anchors:
    :param ignore_thres:
    :param args: input arguments from params file
    :return:
    """
    device = args.device

    nB = pred_boxes.size(0)  # Batch size
    nA = pred_boxes.size(1)  # Anchor size
    nC = pred_cls.size(-1)  # Number of classes
    nG_y = pred_boxes.size(2)  # Grid size vertical
    nG_x = pred_boxes.size(3)  # Grid size horizontal

    # Output tensors
    obj_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device)
    noobj_mask = torch.ones([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device=obj_mask.device)
    class_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    iou_scores = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tx = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    ty = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tw = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    th = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tcls = torch.zeros([nB, nA, nG_y, nG_x, nC], dtype=torch.float).to(device=obj_mask.device)

    # Handel negative samples
    target = target[target[:, 4] > 0]
    if target.nelement() == 0:
        tconf = obj_mask.float()
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    # Assign each object to corresponding grid cell
    target_boxes = torch.cat(((target[:, 2] * nG_x).unsqueeze(1), (target[:, 3] * nG_y).unsqueeze(1),
                              (target[:, 4] * nG_x).unsqueeze(1), (target[:, 5] * nG_y).unsqueeze(1)), 1)
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    # ious shape is (3, number_of_boxes)
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Handel corner cases(boundaries cases)
    gi = torch.clamp(gi, 0, nG_x - 1)
    gj = torch.clamp(gj, 0, nG_y - 1)

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Normalize coordinates relative to the cell (x & y in range [0,1])
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    # Assign the class type
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

"""
Enhanced build_targets function for object detection.

This version uses a matrix-capable IoU function for efficient ignore region
calculation and includes additional debug prints. It incorporates CIoU loss
(calculated outside this function) for regression targets.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>
# author: Varun Ravi Kumar <rvarun7777@gmail.com>
# author: Hazem Rashed <hazem.rashed.@valeo.com>

Parts of the code adapted from https://github.com/eriklindernoren/PyTorch-YOLOv3 and YOLOv7 official implementation
Please refer to the license of the above repos.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from collections import OrderedDict

# Import necessary utility functions from your detection_utils
# Assuming bbox_wh_iou is still in train_utils.detection_utils
try:
    from train_utils.detection_utils import bbox_wh_iou
    print("Using bbox_wh_iou from train_utils.detection_utils.")
except ImportError:
    print("Error: Could not import bbox_wh_iou from train_utils.detection_utils.")
    # Define dummy function if import fails
    def bbox_wh_iou(*args, **kwargs):
        raise NotImplementedError("bbox_wh_iou not imported or defined.")

# --- Integrated Bounding Box Comparison Functions (Matrix-Capable) ---
# Standard implementations of IoU, GIoU, and CIoU.
# Assumes input boxes are in (x_center, y_center, width, height) format
# or (x1, y1, x2, y2) format if x1y1x2y2=True.
# Designed to handle batches of boxes for both box1 and box2, returning a matrix.

def calculate_iou_matrix(box1, box2, x1y1x2y2=False, giou=False, diou=False, ciou=False):
    """
    Calculates the IoU, GIoU, DIoU, or CIoU matrix between two sets of bounding boxes.
    Assumes box1 is a tensor of shape [N, 4] and box2 is a tensor of shape [M, 4].
    Returns a tensor of shape [N, M].
    """
    # Ensure inputs have the correct shape [N, 4] and [M, 4]
    # If inputs are [4], unsqueeze to [1, 4] to handle single boxes as batches
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    N = box1.size(0)
    M = box2.size(0)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4] # Shape [N, 1]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0:1], box2[:, 1:2], box2[:, 2:3], box2[:, 3:4] # Shape [M, 1]
    else:
        # x_center, y_center, width, height
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1] - box1[:, 2:3] / 2, box1[:, 1:2] - box1[:, 3:4] / 2, \
                                    box1[:, 0:1] + box1[:, 2:3] / 2, box1[:, 1:2] + box1[:, 3:4] / 2 # Shape [N, 1]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0:1] - box2[:, 2:3] / 2, box2[:, 1:2] - box2[:, 3:4] / 2, \
                                    box2[:, 0:1] + box2[:, 2:3] / 2, box2[:, 1:2] + box2[:, 3:4] / 2 # Shape [M, 1]

    # Get the coordinates of the intersection rectangle using broadcasting
    # Shapes: [N, 1] vs [1, M] -> [N, M]
    inter_rect_x1 = torch.max(b1_x1, b2_x1.t())
    inter_rect_y1 = torch.max(b1_y1, b2_y1.t())
    inter_rect_x2 = torch.min(b1_x2, b2_x2.t())
    inter_rect_y2 = torch.min(b1_y2, b2_y2.t())

    # Intersection area
    inter = (inter_rect_x2 - inter_rect_x1).clamp(0) * (inter_rect_y2 - inter_rect_y1).clamp(0) # Shape [N, M]

    # Union Area using broadcasting
    # Shapes: [N, 1] vs [1, M] -> [N, M]
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 # Shape [N, 1]
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 # Shape [M, 1]
    union = (w1 * h1 + (w2 * h2).t() - inter) + 1e-16 # Shape [N, M]

    iou = inter / union # IoU shape is [N, M]

    if giou or diou or ciou:
        # convex (bounding) box coordinates using broadcasting
        c_x1, c_y1, c_x2, c_y2 = torch.min(b1_x1, b2_x1.t()), torch.min(b1_y1, b2_y1.t()), \
                                 torch.max(b1_x2, b2_x2.t()), torch.max(b1_y2, b2_y2.t()) # Shape [N, M]
        # convex area
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + 1e-16 # Shape [N, M]
        # GIoU
        if giou:
            return iou - (c_area - union) / c_area # GIoU shape is [N, M]
        # DIoU
        if diou or ciou:
            # center distance squared using broadcasting
            c_dist_sq = ((b1_x1 + b1_x2 - b2_x1.t() - b2_x2.t()) ** 2 + (b1_y1 + b1_y2 - b2_y1.t() - b2_y2.t()) ** 2) / 4 # Shape [N, M]
            # outer diagonal squared
            c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-16 # Shape [N, M]
            diou = iou - c_dist_sq / c_diag_sq # DIoU shape is [N, M]
            if ciou:
                # aspect ratio consistency term
                # Add a small epsilon to denominator in atan for numerical stability
                v = (4 / math.pi ** 2) * torch.pow(torch.atan((w1 / (h1 + 1e-16))) - torch.atan((w2 / (h2 + 1e-16)).t()), 2) # Shape [N, M]
                # alpha
                # Add a small epsilon to the denominator for numerical stability
                alpha = v / (v - iou + 1 + diou + 1e-16) # Shape [N, M]
                # CIoU
                return diou - alpha * v # CIoU shape is [N, M]
            return diou
    return iou # IoU shape is [N, M]


def calculate_ciou_matrix(box1, box2):
    """Calculates the CIoU matrix between box1 and box2."""
    # calculate_iou_matrix function with ciou=True returns the CIoU value directly
    return calculate_iou_matrix(box1, box2, ciou=True)


def calculate_giou_matrix(box1, box2):
    """Calculates the GIoU matrix between box1 and box2."""
    # calculate_iou_matrix function with giou=True returns the GIoU value directly
    return calculate_iou_matrix(box1, box2, giou=True)


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, args):
    """
    Builds targets for YOLO detection loss.

    Args:
        pred_boxes (tensor): Decoded predicted bounding boxes from the YOLOLayer
                             (batch_size, num_anchors, grid_h, grid_w, 4).
                             Format: [x_center, y_center, width, height] in grid cell scale.
        pred_cls (tensor): Sigmoid activated class probabilities from the YOLOLayer
                           (batch_size, num_anchors, grid_h, grid_w, num_classes).
        target (tensor): Ground truth targets (num_targets, 6).
                         Format: [img_id, class_type, x_center, y_center, width, height, (yaw)].
                         Coordinates are normalized [0, 1]. Yaw is optional.
        anchors (tensor): Scaled anchors for the current detection layer (num_anchors, 2).
                          Format: [anchor_width, anchor_height] in grid cell scale.
        ignore_thres (float): IoU threshold to ignore predictions as negative samples.
        args (object): Input arguments from params file (e.g., contains device).

    Returns:
        tuple: Tensors containing target values and masks for loss calculation:
               (iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf)
    """
    device = args.device

    nB = pred_boxes.size(0)  # Batch size
    nA = pred_boxes.size(1)  # Anchor size
    nG_y = pred_boxes.size(2)  # Grid size vertical
    nG_x = pred_boxes.size(3)  # Grid size horizontal
    nC = pred_cls.size(-1)  # Number of classes

    #print(f"--- build_targets inputs ---")
    #print(f"pred_boxes shape: {pred_boxes.shape}")
    #print(f"pred_cls shape: {pred_cls.shape}")
    #print(f"target shape: {target.shape}")
    #print(f"anchors shape: {anchors.shape}")
    #print(f"ignore_thres: {ignore_thres}")
    #print(f"nB: {nB}, nA: {nA}, nG_y: {nG_y}, nG_x: {nG_x}, nC: {nC}")
    #print("----------------------------")

    # Output tensors
    obj_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device)
    # Initialize noobj_mask to all ones, will be set to zero for positive samples and ignored samples
    noobj_mask = torch.ones([nB, nA, nG_y, nG_x], dtype=torch.bool).to(device=obj_mask.device)
    class_mask = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    iou_scores = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tx = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    ty = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tw = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    th = torch.zeros([nB, nA, nG_y, nG_x], dtype=torch.float).to(device=obj_mask.device)
    tcls = torch.zeros([nB, nA, nG_y, nG_x, nC], dtype=torch.float).to(device=obj_mask.device)

    # Handel negative samples (targets with width or height <= 0 are invalid)
    target = target[target[:, 4] > 0]
    if target.nelement() == 0:
        # If no valid targets in the batch, return masks and zero tensors
        tconf = obj_mask.float() # Objectness target is 0 for all
        print("--- build_targets: No valid targets in batch ---")
        return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

    #print(f"--- build_targets: Valid targets shape: {target.shape} ---")

    # Assign each object to corresponding grid cell and best anchor
    # Scale target box coordinates to grid cell scale
    target_boxes_grid = torch.cat(((target[:, 2] * nG_x).unsqueeze(1), (target[:, 3] * nG_y).unsqueeze(1),
                                   (target[:, 4] * nG_x).unsqueeze(1), (target[:, 5] * nG_y).unsqueeze(1)), 1)

    gxy = target_boxes_grid[:, :2] # Center x, y in grid scale
    gwh = target_boxes_grid[:, 2:] # Width, height in grid scale

    # Get anchors with best IoU based on width and height (bbox_wh_iou)
    # ious shape is (num_anchors, num_targets)
    ious_wh = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious_wh, best_n = ious_wh.max(0) # best_n is the anchor index for each target

    # Separate target values for easier indexing
    b, target_labels = target[:, :2].long().t() # batch index, class label
    gx, gy = gxy.t() # grid x, grid y (float)
    gw, gh = gwh.t() # grid w, grid h (float)
    gi, gj = gxy.long().t() # grid cell index i (x), grid cell index j (y)

    # Handel corner cases (boundaries cases) - clamp grid indices
    gi = torch.clamp(gi, 0, nG_x - 1)
    gj = torch.clamp(gj, 0, nG_y - 1)

    # Set object mask to 1 for the assigned grid cell and anchor
    obj_mask[b, best_n, gj, gi] = 1
    # Set noobj mask to 0 for the assigned grid cell and anchor
    noobj_mask[b, best_n, gj, gi] = 0

    # --- Calculate IoU for ignore regions using matrix-capable IoU ---
    img_w, img_h = args.input_width, args.input_height
    stride_w = img_w / nG_x
    stride_h = img_h / nG_y

    # Scale target boxes to image dimensions once
    target_boxes_image_scale_all = torch.cat(((target[:, 2] * img_w).unsqueeze(1),
                                              (target[:, 3] * img_h).unsqueeze(1),
                                              (target[:, 4] * img_w).unsqueeze(1),
                                              (target[:, 5] * img_h).unsqueeze(1)), 1)

    # Flatten predicted boxes across batch, anchors, and grid cells
    pred_boxes_flat = pred_boxes.view(nB * nA * nG_y * nG_x, 4)
    # Scale flattened predicted boxes to image dimensions
    pred_boxes_image_scale_flat = torch.empty_like(pred_boxes_flat)
    pred_boxes_image_scale_flat[:, 0] = pred_boxes_flat[:, 0] * stride_w
    pred_boxes_image_scale_flat[:, 1] = pred_boxes_flat[:, 1] * stride_h
    pred_boxes_image_scale_flat[:, 2] = pred_boxes_flat[:, 2] * stride_w
    pred_boxes_image_scale_flat[:, 3] = pred_boxes_flat[:, 3] * stride_h

    #print(f"--- build_targets: IoU calculation shapes for ignore regions ---")
    #print(f"pred_boxes_image_scale_flat shape: {pred_boxes_image_scale_flat.shape}")
    #print(f"target_boxes_image_scale_all shape: {target_boxes_image_scale_all.shape}")
    #print("-----------------------------------------------------------------")

    # Calculate IoU matrix between all flattened predicted boxes and all target boxes in the batch
    # Using the matrix-capable calculate_iou_matrix
    ious_all = calculate_iou_matrix(pred_boxes_image_scale_flat, target_boxes_image_scale_all, x1y1x2y2=False) # Shape (num_total_predictions_in_batch, num_total_targets_in_batch)

    #print(f"--- build_targets: ious_all matrix shape: {ious_all.shape} ---")

    # For each predicted box, get the maximum IoU with any target box in the batch
    max_ious_for_preds, _ = ious_all.max(1) # Shape (num_total_predictions_in_batch,)

    #print(f"--- build_targets: max_ious_for_preds shape: {max_ious_for_preds.shape} ---")

    # Reshape max_ious_for_preds back to (batch_size, num_anchors, grid_h, grid_w)
    max_ious_reshaped = max_ious_for_preds.view(nB, nA, nG_y, nG_x)

    #print(f"--- build_targets: max_ious_reshaped shape: {max_ious_reshaped.shape} ---")

    # Set noobj mask to zero where max IoU exceeds ignore threshold
    noobj_mask[max_ious_reshaped > ignore_thres] = 0
    # Ensure that locations with objects (obj_mask == 1) are not marked as noobj=0 by the ignore threshold
    noobj_mask[obj_mask] = 0

    #print(f"--- build_targets: obj_mask sum: {obj_mask.sum().item()}, noobj_mask sum: {noobj_mask.sum().item()} ---")


    # --- Calculate Bounding Box Regression Targets ---
    # Standard YOLOv7 regression target calculation:
    # tx = gx - gi
    # ty = gy - gj
    # tw = log(gw / anchor_w)
    # th = log(gh / anchor_h)

    # Calculate regression targets for positive samples (where obj_mask is 1)
    # Need to index gx, gy, gw, gh, and anchors using the same mask logic as obj_mask
    # Create a mask for targets based on which anchor and grid cell they were assigned to
    # NOTE: target_assignment_mask is not actually used for indexing the target values directly.
    # The indices b, best_n, gj, gi are used to set the masks and assign target values.
    # The target values (gx, gy, gw, gh, target_labels) are already filtered for assigned targets.

    # Use these indices to assign regression targets to the correct locations in the output tensors
    tx[b, best_n, gj, gi] = gx - gi.float() # Target x (offset within grid cell)
    ty[b, best_n, gj, gi] = gy - gj.float() # Target y (offset within grid cell)
    # Ensure anchors are accessed correctly based on best_n
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16) # Target w (log scale)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16) # Target h (log scale)

    #print(f"--- build_targets: Calculated regression targets for positive samples ---")


    # One-hot encoding of label for positive samples
    tcls[b, best_n, gj, gi, target_labels] = 1
    #print(f"--- build_targets: Set class targets for positive samples ---")

    # Compute label correctness for positive samples
    # Check if the predicted class with highest score matches the target class
    # Use the indices (b, best_n, gj, gi) to select the corresponding predicted class probabilities
    # pred_cls shape: (nB, nA, nG_y, nG_x, nC)
    pred_cls_pos = pred_cls[b, best_n, gj, gi] # Shape: (num_targets, nC)
    class_mask[b, best_n, gj, gi] = (pred_cls_pos.argmax(-1) == target_labels).float()
    #print(f"--- build_targets: Calculated class mask for positive samples ---")

    # Compute IoU scores between Predicted and Ground Truth Boxes for Positive Anchors
    # Use the predicted boxes (pred_boxes) and the corresponding target boxes (target_boxes_grid)
    # Ensure pred_boxes and target_boxes_grid are indexed correctly for positive samples

    # Get the predicted boxes for the positive anchors using the assignment indices
    # pred_boxes shape: (nB, nA, nG_y, nG_x, 4)
    # b, best_n, gj, gi shapes: (num_targets,)
    pred_boxes_pos = pred_boxes[b, best_n, gj, gi] # Shape: (num_targets, 4)

    # The target boxes for the positive samples are simply the target_boxes_grid
    # as target_boxes_grid is already filtered for valid targets and aligned with the assignment indices
    # REMOVED: target_boxes_grid_pos = target_boxes_grid[target_assignment_mask.view(-1)] # This line caused the error
    target_boxes_grid_pos = target_boxes_grid # Correctly use target_boxes_grid directly

    #print(f"--- build_targets: IoU calculation shapes for positive samples ---")
    #print(f"pred_boxes_pos shape: {pred_boxes_pos.shape}")
    #print(f"target_boxes_grid_pos shape: {target_boxes_grid_pos.shape}")
    #print("-----------------------------------------------------------------")

    # Calculate IoU between predicted and target boxes for positive samples
    # Use the matrix-capable calculate_iou_matrix, comparing N positive predictions to N positive targets
    # The result will be a tensor of shape (N, N). We need the diagonal elements.
    # Since pred_boxes_pos and target_boxes_grid_pos are now aligned (both shape (num_targets, 4)),
    # calculate_iou_matrix will return (num_targets, num_targets).
    # We need the IoU between each predicted box and its corresponding assigned target,
    # which are on the diagonal.
    iou_scores_pos_matrix = calculate_iou_matrix(pred_boxes_pos, target_boxes_grid_pos, x1y1x2y2=False)
    # Get the diagonal elements (IoU between each predicted positive box and its assigned target)
    iou_scores_pos = torch.diag(iou_scores_pos_matrix)

    #print(f"--- build_targets: iou_scores_pos shape: {iou_scores_pos.shape} ---")

    # Assign the calculated IoU scores back to the iou_scores tensor at the positive locations
    # Use the assignment indices (b, best_n, gj, gi) to place the iou_scores_pos (shape num_targets,)
    # into the iou_scores tensor (shape nB, nA, nG_y, nG_x)
    iou_scores[b, best_n, gj, gi] = iou_scores_pos.detach() # Use detach() as IoU is not part of the gradient flow for box loss
    #print(f"--- build_targets: Assigned IoU scores for positive samples ---")


    # Objectness confidence target (tconf)
    # tconf is 1 for positive samples (obj_mask is 1) and 0 for negative samples (noobj_mask is 1)
    # The ignore regions (where noobj_mask was set to 0 due to high IoU) will have a target of 0.
    tconf = obj_mask.float()
    #print(f"--- build_targets: Calculated objectness confidence target ---")


    # Return all generated targets and masks
    #print(f"--- build_targets: Returning targets and masks ---")
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
'''




# Helper function to concatenate tensors from a list or return an empty tensor
def _cat_or_empty(list_of_tensors, shape, device):
    """Concatenates a list of tensors along dimension 0, or returns an empty tensor if the list is empty."""
    if list_of_tensors:
        return torch.cat(list_of_tensors, dim=0)
    else:
        return torch.empty(shape, device=device)

'''''''''
def log_metrics(outputs, targets, losses):
    target = targets[0]
    output = outputs[0]
    # Metrics
    cls_acc = 100 * target["class_mask"][target["obj_mask"]].mean()
    conf_obj = output["obj_conf"].mean()
    conf_noobj = output["no_obj_conf"].mean()
    conf50 = (output["pred_conf"] > 0.5).float()
    iou50 = (target["iou_scores"] > 0.5).float()
    iou75 = (target["iou_scores"] > 0.75).float()
    detected_mask = conf50 * target["class_mask"] * target["tconf"]
    precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
    recall50 = torch.sum(iou50 * detected_mask) / (target["obj_mask"].sum() + 1e-16)
    recall75 = torch.sum(iou75 * detected_mask) / (target["obj_mask"].sum() + 1e-16)

    metrics = dict(detection_loss=get_tensor_value(losses["detection_loss"]),
                   x=get_tensor_value(losses["x"]), y=get_tensor_value(losses["y"]),
                   w=get_tensor_value(losses["w"]), h=get_tensor_value(losses["h"]),
                   conf=get_tensor_value(losses["conf"]),
                   cls=get_tensor_value(losses["cls"]),
                   cls_acc=get_tensor_value(cls_acc),
                   recall50=get_tensor_value(recall50), recall75=get_tensor_value(recall75),
                   precision=get_tensor_value(precision),
                   conf_obj=get_tensor_value(conf_obj), conf_noobj=get_tensor_value(conf_noobj))
    return metrics
'''''''''
'''''''''
# Assuming get_tensor_value is a utility function you have defined
def get_tensor_value(tensor):
    if torch.is_tensor(tensor):
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return tensor

#multiple implementation 1
def log_metrics(outputs, targets, losses):
    # Assume outputs and targets are lists of dictionaries for each head.
    # We will process the first head's output and target for simplicity,
    # matching the structure observed in your debug output.
    # A more complete implementation might aggregate metrics across all heads.

    output = outputs[0]
    target = targets[0]

    # Get the full masks from the target dictionary
    obj_mask = target["obj_mask"]
    noobj_mask = target["noobj_mask"]

    # Get the flattened tensors for positive samples from the output and target dictionaries
    # These were already masked by obj_mask in the _post_proccess_output_target method
    pred_obj_conf_pos = output["obj_conf"] # Sigmoid objectness for positive samples
    target_iou_scores_pos = target["iou_scores"] # IoU scores for positive samples
    # Get the class mask and target confidence for positive samples by applying obj_mask
    target_class_mask_pos = target["class_mask"][obj_mask]
    target_tconf_pos = target["tconf"][obj_mask] # Should be all 1.0 for positive samples

    # Get the flattened tensor for negative samples from the output dictionary
    pred_no_obj_conf_neg = output["no_obj_conf"] # Sigmoid objectness for negative samples

    # Get the full predicted confidence tensor for calculating total conf > 0.5 sum
    pred_conf_full = output["pred_conf"] # Full sigmoid objectness predictions

    # Metrics for positive samples (using flattened tensors)
    # cls_acc is calculated on positive samples
    # target["cls"] in the target_dict is already the one-hot target class for positive samples
    # output["cls"] in the output_dict is the sigmoid predicted class probs for positive samples
    if target["cls"].shape[0] > 0: # Check if there are positive samples
        predicted_class_indices_pos = output["cls"].argmax(-1)
        target_class_labels_pos = target["cls"].argmax(-1) # Assuming target["cls"] is one-hot
        cls_acc = 100 * (predicted_class_indices_pos == target_class_labels_pos).float().mean()
    else:
        cls_acc = torch.tensor(0.0, device=pred_obj_conf_pos.device) # Handle no positive samples


    conf_obj = pred_obj_conf_pos.mean()
    conf_noobj = pred_no_obj_conf_neg.mean()

    # Calculate conf > 0.5 and IoU > 0.5/0.75 for positive samples (flattened)
    conf50_pos = (pred_obj_conf_pos > 0.5).float()
    iou50_pos = (target_iou_scores_pos > 0.5).float()
    iou75_pos = (target_iou_scores_pos > 0.75).float()

    # Calculate detected_mask for positive samples (flattened)
    # This represents correctly classified positive detections with confidence > 0.5
    # target_tconf_pos should be 1.0 for all positive samples, so multiplying by it is redundant.
    detected_mask_pos = conf50_pos * target_class_mask_pos


    # Precision: TP / (TP + FP).
    # TP are positive samples correctly detected (conf > 0.5, correct class, IoU > 0.5).
    # FP are predictions with conf > 0.5 that are not TP.
    # A common approximation: sum(iou50_pos * detected_mask_pos) / sum(conf > 0.5 over all predictions).
    total_conf50_preds = (pred_conf_full > 0.5).float().sum() # Sum over the full tensor

    precision = torch.sum(iou50_pos * detected_mask_pos) / (total_conf50_preds + 1e-16)


    # Recall@50: TP / (Total Ground Truth Objects).
    # Total Ground Truth Objects is the number of positive anchors, which is the sum of the full obj_mask.
    total_gt_objects = obj_mask.sum() # Sum over the full mask

    recall50 = torch.sum(iou50_pos * detected_mask_pos) / (total_gt_objects + 1e-16)
    recall75 = torch.sum(iou75_pos * detected_mask_pos) / (total_gt_objects + 1e-16) # Use iou75_pos here


    metrics = dict(detection_loss=get_tensor_value(losses["detection_loss"]),
                   x=get_tensor_value(losses["x"]), y=get_tensor_value(losses["y"]),
                   w=get_tensor_value(losses["w"]), h=get_tensor_value(losses["h"]),
                   conf=get_tensor_value(losses["conf"]),
                   cls=get_tensor_value(losses["cls"]),
                   cls_acc=get_tensor_value(cls_acc),
                   recall50=get_tensor_value(recall50), recall75=get_tensor_value(recall75),
                   precision=get_tensor_value(precision),
                   conf_obj=get_tensor_value(conf_obj), conf_noobj=get_tensor_value(conf_noobj))
    return metrics

'''''''''
# Assuming get_tensor_value is a utility function you have defined
def get_tensor_value(tensor):
    """Extracts scalar or list value from a tensor."""
    if torch.is_tensor(tensor):
        return tensor.item() if tensor.numel() == 1 and tensor.ndim == 0 else tensor.tolist()
    return tensor
#multiple implementation 2
def log_metrics(outputs, targets, losses):
    """
    Calculates and logs detection metrics.

    Args:
        outputs (list): A list of dictionaries, one for each detection head scale.
                        Each dictionary contains raw predictions and some processed values.
                        Format: [{'x': raw_x_obj, 'y': raw_y_obj, 'w': raw_w_obj, 'h': raw_h_obj,
                                   'obj_conf': raw_obj_conf_obj, 'no_obj_conf': raw_obj_conf_noobj,
                                   'cls': raw_cls_obj, 'pred_conf': sigmoid_conf_all, ...}, ...]
        targets (list): A list of dictionaries, one for each detection head scale.
                        Each dictionary contains target values and masks.
                        Format: [{'x': target_tx_obj, 'y': target_ty_obj, 'w': target_tw_obj, 'h': target_th_obj,
                                   'obj_conf': target_obj_conf, 'no_obj_conf': target_no_obj_conf,
                                   'cls': target_tcls_obj, 'iou_scores': iou_scores_obj,
                                   'class_mask': class_mask_obj, 'obj_mask': obj_mask_full,
                                   'noobj_mask': noobj_mask_full, ...}, ...]
        losses (dict): Dictionary containing the accumulated losses from ObjectDetectionLoss.

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    # Initialize aggregated metrics
    total_cls_acc = 0
    total_conf_obj = 0
    total_conf_noobj = 0
    total_precision_numerator = 0
    total_precision_denominator = 0
    total_recall50_numerator = 0
    total_recall75_numerator = 0
    total_recall_denominator = 0
    total_num_positive_samples = 0 # To correctly average metrics over positive samples

    # Iterate through each detection head's outputs and targets
    for i in range(len(outputs)):
        output = outputs[i] # Output dictionary for scale i
        target = targets[i] # Target dictionary for scale i

        # Metrics that only require positive samples
        if target['x'].nelement() > 0: # Check if there are positive samples for this scale
            num_positive_samples = target['x'].nelement() # Number of positive samples in this head/batch
            total_num_positive_samples += num_positive_samples

            # Class Accuracy: Use the class_mask for positive samples directly
            # target['class_mask'] is already a 1D tensor for positive samples
            total_cls_acc += target["class_mask"].sum() # Sum of correct class predictions for positive samples

            # Confidence for positive samples: Use the raw obj_conf predictions for positive samples
            # output['obj_conf'] is already a 1D tensor of raw predictions for positive samples
            total_conf_obj += torch.sigmoid(output["obj_conf"]).sum() # Sum of sigmoid confidence for positive samples

            # Precision and Recall calculations
            # Need sigmoid confidence for all predictions (output['pred_conf'])
            # Need IoU scores for positive samples (target['iou_scores'])
            # Need class correctness for positive samples (target['class_mask'])
            # Need the original obj_mask to identify positive samples for filtering pred_conf

            # Get sigmoid confidence for all predictions for this head
            pred_conf_all = output["pred_conf"] # Shape (B, nA, nG_y, nG_x)

            # Get the original masks from the target dictionary
            obj_mask_full = target["obj_mask"] # Shape (B, nA, nG_y, nG_x)
            # Create a float version of the object mask for multiplication
            obj_mask_full_float = obj_mask_full.float()

            # Thresholded confidence for all predictions
            conf50_all = (pred_conf_all > 0.5).float() # Shape (B, nA, nG_y, nG_x)

            # Thresholded IoU for positive samples
            # target['iou_scores'] is already a 1D tensor of IoUs for positive samples
            iou50_obj = (target["iou_scores"] > 0.5).float() # Shape (num_positive_samples,)
            iou75_obj = (target["iou_scores"] > 0.75).float() # Shape (num_positive_samples,)

            # Detected mask for positive samples: where confidence > 0.5 AND class is correct AND it's an object
            # target['class_mask'] is 1 for correct class, 0 otherwise for positive samples
            # We need the confidence > 0.5 for the *positive* samples.
            # output['obj_conf'] are raw logits for positive samples. Apply sigmoid and threshold.
            conf50_obj = (torch.sigmoid(output["obj_conf"]) > 0.5).float() # Shape (num_positive_samples,)

            # Detected mask for positive samples (where conf > 0.5 AND class is correct)
            detected_mask_obj = conf50_obj * target["class_mask"] # Shape (num_positive_samples,)

            # Precision: True Positives (conf > 0.5, IoU > 0.5, correct class) / (Total detections with conf > 0.5)
            # True Positives (IoU > 0.5 and detected_mask_obj is 1)
            true_positives_iou50 = (iou50_obj * detected_mask_obj).sum()

            # Total detections with conf > 0.5: Sum of conf50_all over all locations/anchors
            total_detections_conf50 = conf50_all.sum()

            total_precision_numerator += true_positives_iou50
            total_precision_denominator += total_detections_conf50

            # Recall@50: True Positives (conf > 0.5, IoU > 0.5, correct class) / (Total ground truth objects)
            # Total ground truth objects for this head is num_positive_samples
            total_recall50_numerator += true_positives_iou50
            total_recall_denominator += num_positive_samples # Sum of positive samples across heads

            # Recall@75: True Positives (conf > 0.5, IoU > 0.75, correct class) / (Total ground truth objects)
            true_positives_iou75 = (iou75_obj * detected_mask_obj).sum()
            total_recall75_numerator += true_positives_iou75


        # Confidence for negative samples: Use the raw no_obj_conf predictions for negative samples
        # output['no_obj_conf'] is already a 1D tensor of raw predictions for negative samples
        # This is calculated for all negative samples across all heads
        for i in range(len(outputs)): # Need a separate loop or collect all no_obj_conf first
             if outputs[i]["no_obj_conf"].nelement() > 0:
                  total_conf_noobj += torch.sigmoid(outputs[i]["no_obj_conf"]).sum()


    # Calculate final aggregated metrics
    # Avoid division by zero
    cls_acc = (total_cls_acc / total_num_positive_samples) * 100 if total_num_positive_samples > 0 else 0.0
    conf_obj = total_conf_obj / total_num_positive_samples if total_num_positive_samples > 0 else 0.0
    # Need the total number of negative samples across all heads for conf_noobj average
    total_num_negative_samples = sum(output["no_obj_conf"].nelement() for output in outputs)
    conf_noobj = total_conf_noobj / total_num_negative_samples if total_num_negative_samples > 0 else 0.0

    precision = total_precision_numerator / total_precision_denominator if total_precision_denominator > 0 else 0.0
    recall50 = total_recall50_numerator / total_recall_denominator if total_recall_denominator > 0 else 0.0
    recall75 = total_recall75_numerator / total_recall_denominator if total_recall_denominator > 0 else 0.0


    metrics = dict(detection_loss=get_tensor_value(losses["detection_loss"]),
                   #x=get_tensor_value(losses["x"]), y=get_tensor_value(losses["y"]),
                  # w=get_tensor_value(losses["w"]), h=get_tensor_value(losses["h"]),
                   # Removed 'conf' as it was just a sum of scaled obj/no_obj conf losses
                   cls=get_tensor_value(losses["cls"]),
                   cls_acc=get_tensor_value(cls_acc),
                   recall50=get_tensor_value(recall50), recall75=get_tensor_value(recall75),
                   precision=get_tensor_value(precision),
                   conf_obj=get_tensor_value(conf_obj), conf_noobj=get_tensor_value(conf_noobj))

    return metrics


def scale_annotation(box, scaled_size, image_shape, start_box_idx):
    """
    Scale the output boxes to the desired shape (up-sampling / down-sampling)
    :param box: the predected boxes on the scale of the feed images (labels ,x1, y1, x2, y2, yaw)
    :param scaled_size: the desired shape (width, height)
    :param image_shape: the original size (height, width)
    :param start_box_idx: the start index that the box coordinates starts from
    :return:
    """
    # Parse the shape
    height = image_shape[0]
    width = image_shape[1]

    # Compute ratio --> Fraction means downsizing
    ratio_height = scaled_size[1] / height
    ratio_width = scaled_size[0] / width

    # Multiply box with scale
    box[:, start_box_idx + 1] = np.multiply(box[:, start_box_idx + 1], ratio_height)
    box[:, start_box_idx] = np.multiply(box[:, start_box_idx], ratio_width)
    box[:, start_box_idx + 3] = np.multiply(box[:, start_box_idx + 3], ratio_height)
    box[:, start_box_idx + 2] = np.multiply(box[:, start_box_idx + 2], ratio_width)

    return box


def crop_annotation(box, cropping, accepted_crop_ratio, img_size: tuple,
                    orginial_image_size, enable_scaling=False):
    """
    The function takes the cropping and applies it to the bounding boxes.
    :param box: box (labels, x, y, w ,h, yaw)
    :param cropping: desired crop from left,top,right,bottom
    :param accepted_crop_ratio: determines the percentage of accepted area after cropping.
    :param img_size: [w, h] the desired size
    :param orginial_image_size: Original image size before cropping [w,h] (as loaded from the disk)
    :param enable_scaling: scale annotation to desired size after cropping.
    :returns box: The Cropped and scaled box.
    """
    image_width = img_size[0]
    image_height = img_size[1]
    org_image_width = orginial_image_size[0]
    org_image_height = orginial_image_size[1]
    box_xyxy = box.clone()

    # Parse the shape
    height = abs(cropping["top"] - cropping["bottom"])
    width = abs(cropping["left"] - cropping["right"])

    # Compute x1,y1 and x2,y2 un-normalized
    box_xyxy[:, 1] = (box[:, 1] - (box[:, 3] / 2)) * org_image_width
    box_xyxy[:, 2] = (box[:, 2] - (box[:, 4] / 2)) * org_image_height
    box_xyxy[:, 3] = (box[:, 1] + (box[:, 3] / 2)) * org_image_width
    box_xyxy[:, 4] = (box[:, 2] + (box[:, 4] / 2)) * org_image_height

    org_box_width = box_xyxy[:, 3] - box_xyxy[:, 1]
    org_box_height = box_xyxy[:, 4] - box_xyxy[:, 2]

    # Subtract x and y from the box points and Handle the boundaries
    box_xyxy[:, 1] = box_xyxy[:, 1] - cropping["left"]
    box_xyxy[:, 3] = box_xyxy[:, 3] - cropping["left"]
    box_xyxy[:, 2] = box_xyxy[:, 2] - cropping["top"]
    box_xyxy[:, 4] = box_xyxy[:, 4] - cropping["top"]

    # Compute area of the overlapped box
    new_boxes_area = (box_xyxy[:, 3] - box_xyxy[:, 1]) * (box_xyxy[:, 4] - box_xyxy[:, 2])

    # Apply filtering according to area relative to the original box
    skipbox1 = new_boxes_area < (accepted_crop_ratio * (org_box_width * org_box_height))

    # Check area of the output
    # skipbox2 = new_boxes_area < min_accepted_area
    skipbox2_w = abs(box_xyxy[:, 1] - box_xyxy[:, 3]) < 30
    skipbox2_h = abs(box_xyxy[:, 2] - box_xyxy[:, 4]) < 20
    skipbox2 = torch.mul(skipbox2_w, skipbox2_h)

    # Filter boxes according to area conditions
    skipBox = torch.mul(skipbox1, skipbox2)
    box_xyxy_filtered = box_xyxy[skipBox == False]
    if len(box_xyxy_filtered) == 0:
        # Handle negative samples
        box_xyxy_filtered = torch.zeros(box_xyxy[0].unsqueeze(0).shape, dtype=torch.float64)
    box = box_xyxy_filtered.clone()

    # scaling the boxes to the desired image size
    if enable_scaling:
        box_xyxy_filtered = scale_annotation(box_xyxy_filtered,
                                             [image_width, image_height], [height, width],
                                             start_box_idx=1)

    # Convert xyxy to center, width and height and normalize them
    w = (box_xyxy_filtered[:, 3] - box_xyxy_filtered[:, 1])
    h = (box_xyxy_filtered[:, 4] - box_xyxy_filtered[:, 2])
    box[:, 1] = (box_xyxy_filtered[:, 1] + (w / 2)) / image_width
    box[:, 2] = (box_xyxy_filtered[:, 2] + (h / 2)) / image_height
    box[:, 3] = w / image_width
    box[:, 4] = h / image_height

    return box
