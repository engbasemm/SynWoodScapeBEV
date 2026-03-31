"""
Loss function for 2D Object Detection for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""
import torch
from torch import nn
from typing import List, Tuple, Union , Dict
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # Import math for constants like pi
from typing import List, Dict

'''
# --- ObjectDetectionLoss with CIoU - under testing ---
# Предполагаемый путь: models/detection_decoderV7_BFIN_ATTN.py (или аналогичный)
# Этот файл будет содержать YoloDecoder, YOLOLayer, CIoU loss, bbox_iou, и вспомогательные модули Conv и т.д.

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Dict, Optional, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import math
from typing import List, Dict, Any  # Ensure these are imported


# You need to ensure ciou_loss function is accessible here.
# Either define it in this file, or import it from where it's defined
# (e.g., from your models/detection_decoderV7_BFIN_ATTN.py or a utils file)

# --- CIoU Loss Function (copied from yolodecoder_for_ciou_v3 for completeness if not imported) ---
def ciou_loss(preds_cxcywh, targets_cxcywh, eps=1e-7):
    """
    Calculates Complete IoU (CIoU) Loss.
    Args:
        preds_cxcywh (torch.Tensor): Predicted bboxes of shape (N, 4).
                                     Format: (x_center_grid, y_center_grid, width_grid, height_grid)
        targets_cxcywh (torch.Tensor): Target bboxes of shape (N, 4).
                                       Format: (x_center_grid, y_center_grid, width_grid, height_grid)
        eps (float, optional): Small value to prevent division by zero. Defaults to 1e-7.
    Returns:
        torch.Tensor: CIoU loss, typically 1 - CIoU. Shape will be (N,).
    """
    if preds_cxcywh.numel() == 0 or targets_cxcywh.numel() == 0:
        return torch.tensor(0.0, device=preds_cxcywh.device)

    device = preds_cxcywh.device
    targets_cxcywh = targets_cxcywh.to(device)

    # Convert to x1y1x2y2 for IoU calculation
    p_x1 = preds_cxcywh[..., 0] - preds_cxcywh[..., 2] / 2;
    p_y1 = preds_cxcywh[..., 1] - preds_cxcywh[..., 3] / 2
    p_x2 = preds_cxcywh[..., 0] + preds_cxcywh[..., 2] / 2;
    p_y2 = preds_cxcywh[..., 1] + preds_cxcywh[..., 3] / 2
    t_x1 = targets_cxcywh[..., 0] - targets_cxcywh[..., 2] / 2;
    t_y1 = targets_cxcywh[..., 1] - targets_cxcywh[..., 3] / 2
    t_x2 = targets_cxcywh[..., 0] + targets_cxcywh[..., 2] / 2;
    t_y2 = targets_cxcywh[..., 1] + targets_cxcywh[..., 3] / 2

    inter_x1 = torch.max(p_x1, t_x1);
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2);
    inter_y2 = torch.min(p_y2, t_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0);
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection_area = inter_w * inter_h

    p_area = (p_x2 - p_x1) * (p_y2 - p_y1);
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union_area = p_area + t_area - intersection_area + eps
    iou = intersection_area / union_area

    c_x1 = torch.min(p_x1, t_x1);
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2);
    c_y2 = torch.max(p_y2, t_y2)
    convex_w = c_x2 - c_x1;
    convex_h = c_y2 - c_y1
    c_diag_sq = convex_w ** 2 + convex_h ** 2 + eps
    rho_sq = ((preds_cxcywh[..., 0] - targets_cxcywh[..., 0]) ** 2 +
              (preds_cxcywh[..., 1] - targets_cxcywh[..., 1]) ** 2)
    distance_penalty = rho_sq / c_diag_sq

    pw = preds_cxcywh[..., 2];
    ph = preds_cxcywh[..., 3]
    tw = targets_cxcywh[..., 2];
    th = targets_cxcywh[..., 3]
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps)), 2)
    with torch.no_grad(): alpha = v / (v - iou + (1 + eps))
    aspect_ratio_penalty = alpha * v
    ciou_value = iou - distance_penalty - aspect_ratio_penalty
    loss_ciou = 1.0 - ciou_value
    return loss_ciou


# --- End CIoU Loss Function ---


class ObjectDetectionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Loss component scales from JSON config
        self.object_scale = getattr(config, 'object_scale', 1.0)
        self.no_object_scale = getattr(config, 'no_object_scale', 100.0)
        self.cls_loss_scale = getattr(config, 'cls_loss_scale', 1.0)  # Using 1.0 from your JSON
        self.iou_loss_scale = getattr(config, 'iou_loss_scale', 0.05)

        self.debug_mode = getattr(config, 'debug_decoder', False)

    def forward(self, outputs_from_decoder: List[Dict[str, Any]], targets_from_decoder: List[Dict[str, Any]]):
        """
        Args:
            outputs_from_decoder (list): List of output_dicts from YOLOLayers.
                                         Each dict contains 'decoded_pred_boxes_obj', 'obj_conf', 'no_obj_conf', 'cls'.
            targets_from_decoder (list): List of target_dicts from YOLOLayers.
                                         Each dict contains 'decoded_target_boxes_obj', 'obj_conf', 'no_obj_conf', 'cls'.
        """
        loss_ciou_summed_over_all_pos = torch.tensor(0.0, device=self.config.device)
        loss_obj_conf_summed = torch.tensor(0.0, device=self.config.device)
        loss_no_obj_conf_summed = torch.tensor(0.0, device=self.config.device)
        loss_cls_summed = torch.tensor(0.0, device=self.config.device)

        num_positive_samples_total = 0

        if self.debug_mode:
            print(f"--- [DEBUG ObjectDetectionLoss forward] ---")
            print(
                f"  [DEBUG ObjectDetectionLoss] Received {len(outputs_from_decoder)} output_dicts and {len(targets_from_decoder)} target_dicts for loss calculation.")

        for i in range(len(outputs_from_decoder)):  # Iterate over scales (P3, P4, P5)
            output_dict = outputs_from_decoder[i]
            target_dict = targets_from_decoder[i]

            if not output_dict or not target_dict:
                if self.debug_mode: print(
                    f"  [DEBUG ObjectDetectionLoss] Scale {i}: Empty output_dict or target_dict, skipping.")
                continue

            # --- CIoU Loss for Bounding Boxes (only for positive samples) ---
            decoded_pred_boxes_obj = output_dict.get("decoded_pred_boxes_obj")
            decoded_target_boxes_obj = target_dict.get("decoded_target_boxes_obj")

            if self.debug_mode:
                pred_shape_str = str(decoded_pred_boxes_obj.shape) if decoded_pred_boxes_obj is not None else "None"
                tgt_shape_str = str(decoded_target_boxes_obj.shape) if decoded_target_boxes_obj is not None else "None"
                print(
                    f"  [DEBUG ObjectDetectionLoss] Scale {i}: Retrieved for CIoU - Preds shape: {pred_shape_str}, Tgts shape: {tgt_shape_str}")

            if decoded_pred_boxes_obj is not None and decoded_target_boxes_obj is not None and \
                    decoded_pred_boxes_obj.numel() > 0 and decoded_target_boxes_obj.numel() > 0 and \
                    decoded_pred_boxes_obj.shape[0] == decoded_target_boxes_obj.shape[
                0]:  # Ensure same number of positive samples

                num_pos_scale = decoded_pred_boxes_obj.size(0)
                num_positive_samples_total += num_pos_scale

                loss_ciou_unreduced_scale = ciou_loss(decoded_pred_boxes_obj, decoded_target_boxes_obj)
                loss_ciou_summed_for_scale = loss_ciou_unreduced_scale.sum()
                loss_ciou_total += loss_ciou_summed_for_scale  # Accumulate sum before averaging
                if self.debug_mode:
                    print(
                        f"    [DEBUG ObjectDetectionLoss] Scale {i}: CIoU Loss (summed for scale): {loss_ciou_summed_for_scale.item():.4f} for {num_pos_scale} positive samples. Avg per sample: {loss_ciou_summed_for_scale.item() / num_pos_scale if num_pos_scale > 0 else 0:.4f}")
            elif self.debug_mode:
                print(
                    f"    [DEBUG ObjectDetectionLoss] Scale {i}: Skipping CIoU loss. Conditions not met or no positive samples. Preds numel: {decoded_pred_boxes_obj.numel() if decoded_pred_boxes_obj is not None else 'N/A'}, Tgts numel: {decoded_target_boxes_obj.numel() if decoded_target_boxes_obj is not None else 'N/A'}")

            # --- Objectness Confidence Loss (Positive Samples) ---
            raw_pred_obj_conf = output_dict.get("obj_conf")  # Logits for positive locations
            target_obj_conf = target_dict.get("obj_conf")  # Should be all 1s for positive locations
            if raw_pred_obj_conf is not None and target_obj_conf is not None and raw_pred_obj_conf.numel() > 0 and target_obj_conf.numel() > 0:
                loss_obj_conf_scale = self.bce_with_logits_loss(raw_pred_obj_conf, target_obj_conf)
                loss_obj_conf_summed += loss_obj_conf_scale
                if self.debug_mode: print(
                    f"    [DEBUG ObjectDetectionLoss] Scale {i}: Obj Conf Loss (summed): {loss_obj_conf_scale.item():.4f}")

            # --- Objectness Confidence Loss (Negative Samples) ---
            raw_pred_no_obj_conf = output_dict.get("no_obj_conf")  # Logits for negative locations
            target_no_obj_conf = target_dict.get("no_obj_conf")  # Should be all 0s for negative locations
            if raw_pred_no_obj_conf is not None and target_no_obj_conf is not None and raw_pred_no_obj_conf.numel() > 0 and target_no_obj_conf.numel() > 0:
                loss_no_obj_conf_scale = self.bce_with_logits_loss(raw_pred_no_obj_conf, target_no_obj_conf)
                loss_no_obj_conf_summed += loss_no_obj_conf_scale
                if self.debug_mode: print(
                    f"    [DEBUG ObjectDetectionLoss] Scale {i}: NoObj Conf Loss (summed): {loss_no_obj_conf_scale.item():.4f}")

            # --- Classification Loss (only for positive samples) ---
            raw_pred_cls = output_dict.get("cls")  # Logits for positive locations
            target_cls = target_dict.get("cls")  # One-hot encoded targets for positive locations
            if raw_pred_cls is not None and target_cls is not None and \
                    raw_pred_cls.numel() > 0 and target_cls.numel() > 0 and \
                    raw_pred_cls.shape[0] == target_cls.shape[0]:  # Ensure matching number of positive samples
                loss_cls_scale = self.bce_with_logits_loss(raw_pred_cls, target_cls)
                loss_cls_summed += loss_cls_scale
                if self.debug_mode: print(
                    f"    [DEBUG ObjectDetectionLoss] Scale {i}: Class Loss (summed): {loss_cls_scale.item():.4f}")

        # Normalize CIoU loss by total number of positive samples across all scales (if any)
        if num_positive_samples_total > 0:
            avg_loss_ciou = loss_ciou_total / num_positive_samples_total
        else:
            avg_loss_ciou = torch.tensor(0.0, device=self.config.device)  # Avoid NaN if no positive samples

        scaled_loss_ciou = avg_loss_ciou * self.iou_loss_scale

        batch_size = float(self.config.batch_size if hasattr(self.config, 'batch_size') else 1.0)

        scaled_loss_obj_conf = (loss_obj_conf_summed / batch_size) * self.object_scale
        scaled_loss_no_obj_conf = (loss_no_obj_conf_summed / batch_size) * self.no_object_scale
        scaled_loss_cls = (loss_cls_summed / batch_size) * self.cls_loss_scale

        total_detection_loss = scaled_loss_ciou + scaled_loss_obj_conf + scaled_loss_no_obj_conf + scaled_loss_cls

        losses = dict(
            ciou=scaled_loss_ciou.item(),
            obj_conf=scaled_loss_obj_conf.item(),
            no_obj_conf=scaled_loss_no_obj_conf.item(),
            cls=scaled_loss_cls.item(),
            detection_loss=total_detection_loss
        )

        if self.debug_mode:
            print(f"  [DEBUG ObjectDetectionLoss] Total Pos Samples for CIoU: {num_positive_samples_total}")
            print(
                f"  [DEBUG ObjectDetectionLoss] CIoU Loss (avg'd over pos, then scaled by {self.iou_loss_scale}): {losses['ciou']:.4f} (Raw sum before avg: {loss_ciou_total.item() if isinstance(loss_ciou_total, torch.Tensor) else loss_ciou_total:.4f})")
            print(
                f"  [DEBUG ObjectDetectionLoss] ObjConf Loss (summed, div_batch, scaled by {self.object_scale}): {losses['obj_conf']:.4f} (Raw sum: {loss_obj_conf_summed.item():.4f})")
            print(
                f"  [DEBUG ObjectDetectionLoss] NoObjConf Loss (summed, div_batch, scaled by {self.no_object_scale}): {losses['no_obj_conf']:.4f} (Raw sum: {loss_no_obj_conf_summed.item():.4f})")
            print(
                f"  [DEBUG ObjectDetectionLoss] Cls Loss (summed, div_batch, scaled by {self.cls_loss_scale}): {losses['cls']:.4f} (Raw sum: {loss_cls_summed.item():.4f})")
            print(f"  [DEBUG ObjectDetectionLoss] Final Total Detection Loss: {total_detection_loss.item():.4f}")
            print(f"--- [DEBUG ObjectDetectionLoss forward END] ---")

        return losses

'''
#detection loss logbits ( yolo7) - working and provide 0.38 for mAP
class ObjectDetectionLoss(nn.Module):
    """
    This criterion combines the object detection losses (localization, confidence, class).
    Uses BCEWithLogitsLoss for confidence and class for numerical stability with raw predictions.
    """

    def __init__(self, config):
        super().__init__()
        # Use MSELoss for localization (x, y, w, h) - targets are logits
        self.mse_loss = nn.MSELoss(reduction='sum') # Use sum reduction to match original logic

        # Use BCEWithLogitsLoss for confidence and class - takes raw predictions (logits)
        # Targets for BCEWithLogitsLoss should be 0 or 1 (or one-hot for class)
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='sum') # Use sum reduction

        self.loss = dict(x=0, y=0, w=0, h=0, theta=0, obj_conf=0, no_obj_conf=0, conf=0, cls=0, total_loss=0, detection_loss=0)
        self.config = config
        # Assuming object_scale and no_object_scale are weights for positive/negative confidence losses
        self.object_scale = 1.0  #config.object_scale if hasattr(config, 'object_scale') else 1.0
        self.no_object_scale = 4.0 # config.no_object_scale if hasattr(config, 'no_object_scale') else 100.0


    def forward(self, outputs, targets):
        """
        Compute the detection losses (localization loss, confidence loss, class loss).

        Args:
            outputs (list): A list of dictionaries, one for each detection head scale.
                            Each dictionary contains raw predictions for assigned samples.
                            Format: [{'x': raw_x, 'y': raw_y, 'w': raw_w, 'h': raw_h,
                                       'obj_conf': raw_obj_conf_obj, 'no_obj_conf': raw_obj_conf_noobj,
                                       'cls': raw_cls_obj, ...}, ...]
            targets (list): A list of dictionaries, one for each detection head scale.
                            Each dictionary contains target values for assigned samples.
                            Format: [{'x': target_tx, 'y': target_ty, 'w': target_tw, 'h': target_th,
                                       'obj_conf': target_obj_conf, 'no_obj_conf': target_no_obj_conf,
                                       'cls': target_tcls, ...}, ...]

        Returns:
            dict: Dictionary containing individual and total detection losses.
        """
        # Reset losses
        self.loss = dict(x=0, y=0, w=0, h=0, theta=0, obj_conf=0, no_obj_conf=0, conf=0, cls=0, total_loss=0, detection_loss=0)

        # Accumulate losses across all detection heads
        for i in range(len(outputs)):
            output = outputs[i] # Output dictionary for scale i (contains raw predictions)
            target = targets[i] # Target dictionary for scale i (contains target values)

            # Localization Loss (x, y, w, h) - Use MSELoss on raw predictions (logits) and targets (logits)
            # Only calculate for positive samples (where objects are assigned)
            if target['x'].nelement() > 0: # Check if there are positive samples for this scale
                self.loss["x"] += self.mse_loss(output["x"], target["x"])
                self.loss["y"] += self.mse_loss(output["y"], target["y"])
                self.loss["w"] += self.mse_loss(output["w"], target["w"])
                self.loss["h"] += self.mse_loss(output["h"], target["h"])

                # Class Loss - Use BCEWithLogitsLoss on raw class predictions and one-hot targets
                self.loss["cls"] += self.bce_with_logits_loss(output["cls"], target["cls"])

                # Objectness Confidence Loss for Positive Samples - Use BCEWithLogitsLoss on raw obj_conf and target 1s
                self.loss["obj_conf"] += self.bce_with_logits_loss(output["obj_conf"], target["obj_conf"])

            # Objectness Confidence Loss for Negative Samples - Use BCEWithLogitsLoss on raw no_obj_conf and target 0s
            # This is calculated for all negative samples identified by build_targets
            if output["no_obj_conf"].nelement() > 0: # Check if there are negative samples for this scale
                 self.loss["no_obj_conf"] += self.bce_with_logits_loss(output["no_obj_conf"], target["no_obj_conf"])


            # Combine confidence losses with scaling factors
            # Note: The original code combined obj_conf and no_obj_conf with scales here.
            # Let's keep this structure for now, applying scales *after* the BCEWithLogitsLoss.
            # A more standard approach is to apply weights directly within BCEWithLogitsLoss
            # using the 'pos_weight' or 'weight' arguments, but this requires reshaping
            # the weights to match the input tensor dimensions.
            # For simplicity, we'll apply scales to the accumulated losses for now.
            # This might need adjustment depending on how you want to weight the losses per sample.

            # Accumulate scaled confidence losses
            # Note: This accumulation should happen per scale before summing across scales if weighting per scale is desired.
            # Assuming the scales are applied to the total obj_conf and no_obj_conf losses across all scales:
            pass # Scales will be applied when summing total_loss


        # Sum up all individual losses to get the total detection loss
        # Apply scaling factors to the confidence losses here
        total_detection_loss = (self.loss["x"] + self.loss["y"] + self.loss["w"] + self.loss["h"] + self.loss["cls"] +
                                self.object_scale * self.loss["obj_conf"] + self.no_object_scale * self.loss["no_obj_conf"])

        self.loss["detection_loss"] = total_detection_loss
        self.loss["total_loss"] = total_detection_loss # Assuming no other losses in this criterion

        return self.loss

