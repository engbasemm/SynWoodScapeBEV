# Предполагаемый путь: models/detection_decoderV7_BFIN_ATTN.py (или аналогичный)
# Этот файл будет содержать YoloDecoder, YOLOLayer, CIoU loss, bbox_iou, и вспомогательные модули Conv и т.д.

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Dict, Optional, Any
from collections import OrderedDict

# --- Utility Functions (bbox_iou, ciou_loss) ---
try:
    from train_utils.detection_utils import build_targets

    print("Successfully imported build_targets from train_utils.detection_utils for YOLOLayer.")
except ImportError:
    print("CRITICAL WARNING: build_targets could not be imported. YOLOLayer will fail during training.")


    def build_targets(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_targets is not available. Training will fail.")


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    device = box1.device
    box2 = box2.to(device)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    else:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1.transpose(-1, -2))
    inter_rect_y1 = torch.max(b1_y1, b2_y1.transpose(-1, -2))
    inter_rect_x2 = torch.min(b1_x2, b2_x2.transpose(-1, -2))
    inter_rect_y2 = torch.min(b1_y2, b2_y2.transpose(-1, -2))

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area.transpose(-1, -2) - inter_area + eps
    iou = inter_area / union_area

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2.transpose(-1, -2)) - torch.min(b1_x1, b2_x1.transpose(-1, -2))
        ch = torch.max(b1_y2, b2_y2.transpose(-1, -2)) - torch.min(b1_y1, b2_y1.transpose(-1, -2))
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            b1_cx = (b1_x1 + b1_x2) / 2;
            b1_cy = (b1_y1 + b1_y2) / 2
            b2_cx_T = ((b2_x1.transpose(-1, -2)) + (b2_x2.transpose(-1, -2))) / 2
            b2_cy_T = ((b2_y1.transpose(-1, -2)) + (b2_y2.transpose(-1, -2))) / 2
            rho2 = (b2_cx_T - b1_cx) ** 2 + (b2_cy_T - b1_cy) ** 2
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                b1_w = b1_x2 - b1_x1;
                b1_h = b1_y2 - b1_y1
                b2_w_T = (b2_x2.transpose(-1, -2)) - (b2_x1.transpose(-1, -2))
                b2_h_T = (b2_y2.transpose(-1, -2)) - (b2_y1.transpose(-1, -2))
                v = (4 / (math.pi ** 2)) * torch.pow(
                    torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w_T / (b2_h_T + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
            c_area = cw * ch + eps; return iou - (c_area - union_area) / c_area
    return iou


def ciou_loss(preds_cxcywh, targets_cxcywh, eps=1e-7):
    if preds_cxcywh.numel() == 0 or targets_cxcywh.numel() == 0:
        return torch.tensor(0.0, device=preds_cxcywh.device)
    device = preds_cxcywh.device
    targets_cxcywh = targets_cxcywh.to(device)
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


class Conv(nn.Module):  # Your Conv implementation
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1, act: bool = True,
                 deploy: bool = False):
        super(Conv, self).__init__();
        self.deploy = deploy
        if p is None: p = k // 2 if k > 1 else 0
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True if self.deploy else False)
        if not self.deploy: self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) if self.deploy else self.bn(self.conv(x)))

    def _fuse_kernel(self) -> None:
        if not self.deploy and hasattr(self, 'bn'):
            bn_gamma, bn_beta, bn_mean, bn_var, bn_eps, conv_w = self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps, self.conv.weight
            std = (bn_var + bn_eps).sqrt();
            t = bn_gamma / std
            fused_conv_w = conv_w * t.reshape(-1, 1, 1, 1);
            fused_conv_b = bn_beta - bn_mean * t
            fused_conv_layer = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size,
                                         self.conv.stride, self.conv.padding, groups=self.conv.groups, bias=True).to(
                conv_w.device)
            fused_conv_layer.weight.data = fused_conv_w;
            fused_conv_layer.bias.data = fused_conv_b
            self.conv = fused_conv_layer;
            del self.bn;
            self.deploy = True


class SPPCSPC(nn.Module):  # Your SPPCSPC implementation
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5,
                 k: Tuple[int, int, int] = (5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(c2 * e);
        self.cv1 = Conv(c1, c_, 1, 1);
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1);
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(c_ * (1 + len(k)), c_, 1, 1);
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)));
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        return self.cv7(torch.cat([y1, self.cv2(x)], dim=1))


class RepConv(nn.Module):  # Your RepConv implementation
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1, act: bool = True,
                 deploy: bool = False):
        super(RepConv, self).__init__();
        self.deploy = deploy;
        self.groups = g;
        self.in_channels = c1;
        self.out_channels = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        assert k == 3 and p == k // 2
        if deploy:
            self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
        else:
            self.rconv_bn = nn.BatchNorm2d(c2);
            self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.branch_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False);
            self.branch_bn = nn.BatchNorm2d(c2)
            self.branch_linear_bn = nn.BatchNorm2d(c2);
            self.branch_linear = nn.Conv2d(c1, c2, 1, s, groups=g, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy: return self.act(self.rconv(x))
        return self.act(self.rconv_bn(self.rconv(x)) + self.branch_bn(self.branch_conv(x)) + self.branch_linear_bn(
            self.branch_linear(x)))

    def _get_equivalent_kernel_bias(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        bn_gamma, bn_beta, bn_mean, bn_var, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
        std = (bn_var + bn_eps).sqrt();
        t = bn_gamma / std
        return conv.weight.data * t.reshape(-1, 1, 1, 1), bn_beta - bn_mean * t

    def _fuse_kernel(self) -> None:
        if self.deploy or not hasattr(self, 'rconv_bn'): return
        kernel_rconv, bias_rconv = self._get_equivalent_kernel_bias(self.rconv, self.rconv_bn)
        kernel_branch, bias_branch = self._get_equivalent_kernel_bias(self.branch_conv, self.branch_bn)
        kernel_linear, bias_linear = self._get_equivalent_kernel_bias(self.branch_linear, self.branch_linear_bn)
        kernel_linear_padded = F.pad(kernel_linear, [1, 1, 1, 1])
        fused_kernel = kernel_rconv + kernel_branch + kernel_linear_padded;
        fused_bias = bias_rconv + bias_branch + bias_linear
        fused_conv_layer = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.rconv.kernel_size,
                                     stride=self.rconv.stride, padding=self.rconv.padding, groups=self.groups,
                                     bias=True).to(fused_kernel.device)
        fused_conv_layer.weight.data = fused_kernel;
        fused_conv_layer.bias.data = fused_bias
        self.rconv = fused_conv_layer;
        del self.rconv_bn, self.branch_conv, self.branch_bn, self.branch_linear, self.branch_linear_bn;
        self.deploy = True


class ChannelAttention(nn.Module):  # Your ChannelAttention implementation
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__();
        self.avg_pool = nn.AdaptiveAvgPool2d(1);
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False);
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False);
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))));
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):  # Your SpatialAttention implementation
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__();
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False);
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True);
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):  # Your CBAM implementation
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__();
        self.ca = ChannelAttention(in_channels, reduction_ratio);
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor: x = x * self.ca(x); return x * self.sa(x)


class BiFPNBlock(nn.Module):  # Your BiFPNBlock implementation
    def __init__(self, in_channels_high_res: int, in_channels_low_res: int, out_channels: int, use_cbam: bool = False):
        super().__init__();
        self.conv_high_res = Conv(in_channels_high_res, out_channels, k=1, s=1)
        self.conv_low_res = Conv(in_channels_low_res, out_channels, k=1, s=1)
        self.weight_high_res = nn.Parameter(torch.ones(1));
        self.weight_low_res = nn.Parameter(torch.ones(1))
        self.relu = nn.ReLU();
        self.conv_after_fusion = Conv(out_channels, out_channels, k=3, s=1)
        self.use_cbam = use_cbam
        if self.use_cbam: self.cbam = CBAM(out_channels)

    def forward(self, feature_high_res: torch.Tensor, feature_low_res: torch.Tensor) -> torch.Tensor:
        upsampled_low_res = F.interpolate(feature_low_res, size=feature_high_res.shape[2:], mode='nearest')
        transformed_high_res = self.conv_high_res(feature_high_res);
        transformed_low_res = self.conv_low_res(upsampled_low_res)
        weight_high_res_norm = self.relu(self.weight_high_res);
        weight_low_res_norm = self.relu(self.weight_low_res)
        epsilon = 1e-4;
        fusion_weight_sum = weight_high_res_norm + weight_low_res_norm + epsilon
        fused_feature = (
                                    transformed_high_res * weight_high_res_norm + transformed_low_res * weight_low_res_norm) / fusion_weight_sum
        fused_feature = self.conv_after_fusion(fused_feature)
        if self.use_cbam: fused_feature = self.cbam(fused_feature)
        return fused_feature


# --- Detection Head (YOLOLayer) --- MODIFIED for CIoU Loss data preparation ---
class YOLOLayer(nn.Module):
    def __init__(self, anchors: List[List[float]], args: Any):
        super().__init__()
        self.args = args
        self.debug_mode = getattr(args, 'debug_decoder', False)
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_classes = self.args.num_classes_detection
        self.ignore_thres = float(self.args.ignore_thres)
        self.img_dim = [self.args.input_width, self.args.input_height]
        self.grid_size: List[int] = [0, 0]
        self.stride: List[float] = [0.0, 0.0]
        self.in_channels_head: Optional[int] = None
        self.conv1: Optional[RepConv] = None
        self.conv2: Optional[RepConv] = None
        self.prediction: Optional[nn.Conv2d] = None
        self.output: Optional[torch.Tensor] = None

        self.register_buffer('grid_x', torch.empty(0))
        self.register_buffer('grid_y', torch.empty(0))
        self.register_buffer('scaled_anchors', torch.empty(0))
        self.register_buffer('anchor_w', torch.empty(0))
        self.register_buffer('anchor_h', torch.empty(0))

    def set_in_channels(self, in_channels: int) -> None:
        self.in_channels_head = in_channels
        device = self.args.device
        self.conv1 = RepConv(self.in_channels_head, self.in_channels_head, k=3).to(device)
        self.conv2 = RepConv(self.in_channels_head, self.in_channels_head, k=3).to(device)
        self.prediction = nn.Conv2d(self.in_channels_head, self.num_anchors * (5 + self.num_classes), 1).to(device)
        if self.debug_mode:
            print(
                f"[DEBUG YOLOLayer init] Head for {self.anchors} anchors, in_channels: {in_channels}, num_classes: {self.num_classes}")

    def compute_grid_offsets(self, grid_size: List[int]) -> None:
        self.grid_size = grid_size
        g_h, g_w = self.grid_size[0], self.grid_size[1]

        if self.img_dim[0] == 0 or self.img_dim[1] == 0:
            if self.debug_mode: print("Warning: img_dim not set in YOLOLayer.compute_grid_offsets.")
            self.stride = [1.0, 1.0]
        else:
            self.stride = [self.img_dim[0] / g_w, self.img_dim[1] / g_h]

        grid_x_row = torch.arange(g_w, device=self.args.device, dtype=torch.float32)
        grid_y_col = torch.arange(g_h, device=self.args.device, dtype=torch.float32)
        self.grid_x = grid_x_row.view(1, 1, 1, g_w).expand(1, 1, g_h, g_w)
        self.grid_y = grid_y_col.view(1, 1, g_h, 1).expand(1, 1, g_h, g_w)

        _scaled_anchors = torch.tensor([(a_w / self.stride[0], a_h / self.stride[1])
                                        for a_w, a_h in self.anchors], device=self.args.device, dtype=torch.float32)
        self.scaled_anchors = _scaled_anchors
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        if self.debug_mode:
            print(
                f"[DEBUG YOLOLayer compute_grid_offsets] Grid: {g_h}x{g_w}, Stride: {self.stride}, Scaled Anchors (sum): {self.scaled_anchors.sum().item() if self.scaled_anchors.numel() > 0 else 'N/A'}")

    def _post_process_output_target(self,
                                    targets: Optional[torch.Tensor],
                                    raw_predictions_from_conv: torch.Tensor,
                                    prediction_reshaped_logits: torch.Tensor,  # Raw logits (B,nA,gH,gW,5+nC)
                                    decoded_pred_obj_conf_for_nms: torch.Tensor,  # Sigmoid applied to obj_conf logit
                                    grid_shape: List[int]) -> Tuple[
        Optional[torch.Tensor], Dict[str, Any], Dict[str, Any]]:
        device = self.args.device

        # Initialize dictionaries that will always be returned
        output_dict_for_loss = {
            "decoded_pred_boxes_obj": torch.empty(0, 4, device=device),
            "obj_conf": torch.empty(0, device=device),  # Will hold raw logits for positive samples
            "no_obj_conf": torch.empty(0, device=device),  # Will hold raw logits for negative samples
            "cls": torch.empty(0, self.num_classes, device=device)  # Will hold raw logits for positive samples
        }
        target_dict_for_loss = {
            "decoded_target_boxes_obj": torch.empty(0, 4, device=device),
            "obj_conf": torch.empty(0, device=device),  # Will hold 1s for positive samples
            "no_obj_conf": torch.empty(0, device=device),  # Will hold 0s for negative samples
            "cls": torch.empty(0, self.num_classes, device=device),  # Will hold target classes for positive samples
            "obj_mask": torch.empty(0, dtype=torch.bool, device=device),
            "noobj_mask": torch.empty(0, dtype=torch.bool, device=device)
        }

        if self.debug_mode:
            print(
                f"[DEBUG YOLOLayer _post_process_output_target] Targets shape: {targets.shape if targets is not None else 'None'}")

        if targets is None:  # Validation or inference mode
            return self.output, output_dict_for_loss, target_dict_for_loss

        # Training mode
        # build_targets returns:
        # obj_mask (B,nA,gH,gW), noobj_mask (B,nA,gH,gW),
        # tx_logit_tgt_full, ty_logit_tgt_full, tw_logit_tgt_full, th_logit_tgt_full (B,nA,gH,gW),
        # tcls_tgt_full (B,nA,gH,gW,nC) or (B,nA,gH,gW) if not one-hot
        # raw_pred_x_obj, ..., raw_pred_obj_conf_obj, raw_pred_cls_obj (1D, N_pos_total_batch) <- these are from build_targets internal logic
        # raw_pred_obj_conf_noobj (1D, N_neg_total_batch)
        # _iou_scores_full, _class_mask_full (B,nA,gH,gW)
        (obj_mask, noobj_mask,
         tx_logit_tgt_full, ty_logit_tgt_full, tw_logit_tgt_full, th_logit_tgt_full, tcls_tgt_full,
         # The following raw_pred_... items are from build_targets, using its internal selection.
         # For confidence and class loss, we need the network's actual output logits for the positive/negative locations.
         _raw_pred_x_obj_bt, _raw_pred_y_obj_bt, _raw_pred_w_obj_bt, _raw_pred_h_obj_bt,
         raw_pred_obj_conf_obj_bt, raw_pred_cls_obj_bt,
         # These are from build_targets, might not be needed directly if we re-select from prediction_reshaped_logits
         raw_pred_obj_conf_noobj_bt,  # This one IS useful
         _iou_scores_full, _class_mask_full) = build_targets(
            raw_predictions=raw_predictions_from_conv,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
            args=self.args
        )

        obj_mask = obj_mask.bool()
        target_dict_for_loss["obj_mask"] = obj_mask
        target_dict_for_loss["noobj_mask"] = noobj_mask

        # Populate common parts for loss (using raw network logits selected by masks)
        output_dict_for_loss["obj_conf"] = prediction_reshaped_logits[..., 4][obj_mask]
        output_dict_for_loss["no_obj_conf"] = prediction_reshaped_logits[..., 4][noobj_mask]
        output_dict_for_loss["cls"] = prediction_reshaped_logits[..., 5:][obj_mask]

        target_dict_for_loss["obj_conf"] = torch.ones_like(output_dict_for_loss["obj_conf"])
        target_dict_for_loss["no_obj_conf"] = torch.zeros_like(output_dict_for_loss["no_obj_conf"])
        target_dict_for_loss["cls"] = tcls_tgt_full[obj_mask]  # Assuming tcls_tgt_full is (B,nA,gH,gW,nC)

        if self.debug_mode:
            print(f"    [DEBUG YOLOLayer _post_process] obj_mask sum for current scale: {obj_mask.sum().item()}")

        if obj_mask.any():
            # Get raw logits for positive predictions from the reshaped tensor using obj_mask
            raw_x_logits_obj_for_decode = prediction_reshaped_logits[..., 0][obj_mask]
            raw_y_logits_obj_for_decode = prediction_reshaped_logits[..., 1][obj_mask]
            raw_w_logits_obj_for_decode = prediction_reshaped_logits[..., 2][obj_mask]
            raw_h_logits_obj_for_decode = prediction_reshaped_logits[..., 3][obj_mask]

            grid_x_obj = self.grid_x.expand_as(prediction_reshaped_logits[..., 0])[obj_mask]
            grid_y_obj = self.grid_y.expand_as(prediction_reshaped_logits[..., 1])[obj_mask]
            anchor_w_obj = self.anchor_w.expand_as(prediction_reshaped_logits[..., 2])[obj_mask]
            anchor_h_obj = self.anchor_h.expand_as(prediction_reshaped_logits[..., 3])[obj_mask]

            pred_cx_grid = (torch.sigmoid(raw_x_logits_obj_for_decode) * 2. - 0.5 + grid_x_obj)
            pred_cy_grid = (torch.sigmoid(raw_y_logits_obj_for_decode) * 2. - 0.5 + grid_y_obj)
            pred_w_grid = (torch.sigmoid(raw_w_logits_obj_for_decode) * 2).pow(2) * anchor_w_obj
            pred_h_grid = (torch.sigmoid(raw_h_logits_obj_for_decode) * 2).pow(2) * anchor_h_obj
            output_dict_for_loss["decoded_pred_boxes_obj"] = torch.stack(
                [pred_cx_grid, pred_cy_grid, pred_w_grid, pred_h_grid], dim=1)

            # Use the full grid target logits and mask them here for decoding
            tx_logit_tgt_obj_masked = tx_logit_tgt_full[obj_mask]
            ty_logit_tgt_obj_masked = ty_logit_tgt_full[obj_mask]
            tw_logit_tgt_obj_masked = tw_logit_tgt_full[obj_mask]
            th_logit_tgt_obj_masked = th_logit_tgt_full[obj_mask]

            target_cx_grid = (torch.sigmoid(tx_logit_tgt_obj_masked) * 2. - 0.5 + grid_x_obj)
            target_cy_grid = (torch.sigmoid(ty_logit_tgt_obj_masked) * 2. - 0.5 + grid_y_obj)
            target_w_grid = (torch.sigmoid(tw_logit_tgt_obj_masked) * 2).pow(2) * anchor_w_obj
            target_h_grid = (torch.sigmoid(th_logit_tgt_obj_masked) * 2).pow(2) * anchor_h_obj
            target_dict_for_loss["decoded_target_boxes_obj"] = torch.stack(
                [target_cx_grid, target_cy_grid, target_w_grid, target_h_grid], dim=1)

            if self.debug_mode:
                print(f"        [DEBUG YOLOLayer _post_process] INSIDE obj_mask.any():")
                print(
                    f"            decoded_pred_boxes_for_loss shape: {output_dict_for_loss['decoded_pred_boxes_obj'].shape}")
                print(
                    f"            decoded_target_boxes_for_loss shape: {target_dict_for_loss['decoded_target_boxes_obj'].shape}")

        if self.debug_mode:
            print(
                f"    [DEBUG YOLOLayer _post_process_output_target] RETURNING output_dict_for_loss keys: {list(output_dict_for_loss.keys())}")
            print(
                f"    [DEBUG YOLOLayer _post_process_output_target] RETURNING target_dict_for_loss keys: {list(target_dict_for_loss.keys())}")
        return self.output, output_dict_for_loss, target_dict_for_loss

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                img_dim: Optional[Union[List[int], torch.Tensor]] = None) -> Tuple[
        Optional[torch.Tensor], Dict[str, Any], Dict[str, Any]]:
        if self.in_channels_head is None or self.conv1 is None or self.conv2 is None or self.prediction is None:
            raise RuntimeError("YOLOLayer's layers not initialized. Call set_in_channels() first.")

        features_after_head_convs = self.conv2(self.conv1(x))
        raw_predictions_from_conv = self.prediction(features_after_head_convs)

        if img_dim is not None:
            if torch.is_tensor(img_dim):
                img_dim_list = img_dim.cpu().tolist()
            else:
                img_dim_list = img_dim
            if self.img_dim != img_dim_list: self.img_dim = img_dim_list

        num_samples = raw_predictions_from_conv.size(0)
        current_grid_shape = [raw_predictions_from_conv.size(2), raw_predictions_from_conv.size(3)]

        if self.grid_size != current_grid_shape or self.scaled_anchors.numel() == 0:
            self.compute_grid_offsets(current_grid_shape)

        num_of_output_params = 5 + self.num_classes
        prediction_reshaped_logits = (
            raw_predictions_from_conv.view(
                num_samples, self.num_anchors, num_of_output_params, current_grid_shape[0], current_grid_shape[1]
            ).permute(0, 1, 3, 4, 2).contiguous()
        )

        decoded_x_offset_nms = torch.sigmoid(prediction_reshaped_logits[..., 0])
        decoded_y_offset_nms = torch.sigmoid(prediction_reshaped_logits[..., 1])
        pred_cx_grid_nms = (decoded_x_offset_nms * 2. - 0.5 + self.grid_x)
        pred_cy_grid_nms = (decoded_y_offset_nms * 2. - 0.5 + self.grid_y)
        pred_w_grid_nms = (torch.sigmoid(prediction_reshaped_logits[..., 2]) * 2).pow(2) * self.anchor_w
        pred_h_grid_nms = (torch.sigmoid(prediction_reshaped_logits[..., 3]) * 2).pow(2) * self.anchor_h

        decoded_pred_obj_conf_nms = torch.sigmoid(prediction_reshaped_logits[..., 4])
        decoded_pred_cls_probs_nms = torch.sigmoid(prediction_reshaped_logits[..., 5:])

        output_boxes_img_scale = torch.empty_like(prediction_reshaped_logits[..., :4], device=self.args.device)
        output_boxes_img_scale[..., 0] = pred_cx_grid_nms * self.stride[0]
        output_boxes_img_scale[..., 1] = pred_cy_grid_nms * self.stride[1]
        output_boxes_img_scale[..., 2] = pred_w_grid_nms * self.stride[0]
        output_boxes_img_scale[..., 3] = pred_h_grid_nms * self.stride[1]

        self.output = torch.cat((
            output_boxes_img_scale.view(num_samples, -1, 4),
            decoded_pred_obj_conf_nms.view(num_samples, -1, 1),
            decoded_pred_cls_probs_nms.view(num_samples, -1, self.num_classes)
        ), -1)

        nms_out, out_dict, tgt_dict = self._post_process_output_target(
            targets,
            raw_predictions_from_conv,
            prediction_reshaped_logits,
            decoded_pred_obj_conf_nms,
            current_grid_shape
        )

        if self.debug_mode and targets is not None:  # Added check for targets not None
            print(f"    [DEBUG YOLOLayer.forward] ABOUT TO RETURN:")
            print(f"        type(nms_out): {type(nms_out)}")
            print(f"        type(out_dict): {type(out_dict)}")
            if isinstance(out_dict, dict): print(f"            out_dict keys: {list(out_dict.keys())}")
            print(f"        type(tgt_dict): {type(tgt_dict)}")
            if isinstance(tgt_dict, dict): print(f"            tgt_dict keys: {list(tgt_dict.keys())}")

        return nms_out, out_dict, tgt_dict


# --- YoloDecoder Class (Main Neck + Heads) ---
class YoloDecoder(nn.Module):
    def __init__(self, _out_filters: List[int], args: Any):
        super(YoloDecoder, self).__init__()
        self.args = args;
        self.debug_mode = getattr(args, 'debug_decoder', False)
        self.all_in_channels = _out_filters;
        self.target_strides = [8, 16, 32]
        p3_ch, p4_ch, p5_ch = _out_filters[-3], _out_filters[-2], _out_filters[-1]
        self.num_scales = len(self.target_strides)

        anchors_p3 = getattr(args, 'anchors3', [[10, 13], [16, 30], [33, 23]])
        anchors_p4 = getattr(args, 'anchors2', [[30, 61], [62, 45], [59, 119]])
        anchors_p5 = getattr(args, 'anchors1', [[116, 90], [156, 198], [373, 326]])
        anchors_list_for_heads = [list(anchors_p3), list(anchors_p4), list(anchors_p5)]

        img_w, img_h = self.args.input_width, self.args.input_height
        self.grid_sizes = [[img_h // s, img_w // s] for s in self.target_strides]

        spp_c = getattr(args, 'neck_sppcspc_channels', 512);
        p5p4_c = spp_c // 2
        bfp4_c = getattr(args, 'neck_bifpn_p4_channels', 512);
        p4p3_c = bfp4_c // 2
        bfp3_c = getattr(args, 'neck_bifpn_p3_channels', 256)
        use_ncb = getattr(args, 'use_neck_cbam', False);
        use_bcb = use_ncb and getattr(args, 'use_bifpn_cbam', False)

        self.sppcspc = SPPCSPC(p5_ch, spp_c)
        if use_ncb and not use_bcb: self.cbam_p5 = CBAM(spp_c)
        self.conv_p5_to_p4_branch = Conv(spp_c, p5p4_c, k=1)
        self.upsample_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bifpn_block_p4 = BiFPNBlock(p4_ch, p5p4_c, bfp4_c, use_cbam=use_bcb)
        if use_ncb and not use_bcb: self.cbam_p4 = CBAM(bfp4_c)
        self.conv_p4_to_p3_branch = Conv(bfp4_c, p4p3_c, k=1)
        self.upsample_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bifpn_block_p3 = BiFPNBlock(p3_ch, p4p3_c, bfp3_c, use_cbam=use_bcb)
        if use_ncb and not use_bcb: self.cbam_p3 = CBAM(bfp3_c)

        self.head_p3 = YOLOLayer(anchors_list_for_heads[0], self.args);
        self.head_p3.set_in_channels(bfp3_c)
        self.head_p4 = YOLOLayer(anchors_list_for_heads[1], self.args);
        self.head_p4.set_in_channels(bfp4_c)
        self.head_p5 = YOLOLayer(anchors_list_for_heads[2], self.args);
        self.head_p5.set_in_channels(spp_c)
        self.final_layers = nn.ModuleList([self.head_p3, self.head_p4, self.head_p5])
        self._initialize_grids_and_biases()

    def _initialize_grids_and_biases(self) -> None:
        for i, head in enumerate(self.final_layers):  # type: ignore
            head.compute_grid_offsets(self.grid_sizes[i])
            if head.prediction is not None and head.prediction.bias is not None:
                b = head.prediction.bias.data.view(head.num_anchors, -1);
                b[:, 4] = -math.log((1 - 0.01) / 0.01)

    def switch_to_deploy(self) -> None:
        for m_name, module in self.named_modules():
            if hasattr(module, '_fuse_kernel'): module._fuse_kernel()

    def forward(self, backbone_features: List[torch.Tensor], img_dim: Optional[Union[List[int], torch.Tensor]] = None,
                targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        current_epoch_for_debug = getattr(self.args, 'current_epoch', 'N/A')
        current_step_for_debug = getattr(self.args, 'current_step', 'N/A')
        if self.debug_mode:
            print(
                f"\n--- [DEBUG YoloDecoder forward] --- Epoch: {current_epoch_for_debug}, Step: {current_step_for_debug} ---")

        p3_in, p4_in, p5_in = backbone_features[-3], backbone_features[-2], backbone_features[-1]
        sppcspc_out = self.sppcspc(p5_in);
        if hasattr(self, 'cbam_p5'): sppcspc_out = self.cbam_p5(sppcspc_out)
        p5_b = self.conv_p5_to_p4_branch(sppcspc_out);
        x1_up = self.upsample_p5_to_p4(p5_b)
        bfp4_out = self.bifpn_block_p4(p4_in, x1_up)
        if hasattr(self, 'cbam_p4'): bfp4_out = self.cbam_p4(bfp4_out)
        p4_b = self.conv_p4_to_p3_branch(bfp4_out);
        x2_up = self.upsample_p4_to_p3(p4_b)
        bfp3_out = self.bifpn_block_p3(p3_in, x2_up)
        if hasattr(self, 'cbam_p3'): bfp3_out = self.cbam_p3(bfp3_out)

        neck_outs = [bfp3_out, bfp4_out, sppcspc_out]
        yolo_nms_outputs_list: List[Optional[torch.Tensor]] = []
        yolo_output_dicts_for_loss: List[Dict[str, Any]] = []
        yolo_target_dicts_for_loss: List[Dict[str, Any]] = []

        current_img_dim_val = img_dim
        if current_img_dim_val is None:
            current_img_dim_val = [self.args.input_width, self.args.input_height]

        for i in range(self.num_scales):
            layer_return_tuple = self.final_layers[i](
                neck_outs[i], targets, current_img_dim_val
            )

            # Explicitly unpack
            nms_output: Optional[torch.Tensor] = layer_return_tuple[0]
            output_dict_loss_scale: Dict[str, Any] = layer_return_tuple[1]  # Type hint for clarity
            target_dict_loss_scale: Dict[str, Any] = layer_return_tuple[2]  # Type hint for clarity

            if self.debug_mode and targets is not None:
                print(
                    f"  [DEBUG YoloDecoder forward] Scale {i} - YOLOLayer returned tuple type: {type(layer_return_tuple)}, len: {len(layer_return_tuple) if isinstance(layer_return_tuple, tuple) else 'N/A'}")
                print(f"    Tuple element 0 (nms_output) type: {type(nms_output)}")
                print(f"    Tuple element 1 (output_dict_loss_scale) type: {type(output_dict_loss_scale)}")
                print(f"    Tuple element 2 (target_dict_loss_scale) type: {type(target_dict_loss_scale)}")
                if output_dict_loss_scale is not None and isinstance(output_dict_loss_scale, dict):
                    print(f"    Keys in output_dict_loss_scale: {list(output_dict_loss_scale.keys())}")
                    pred_boxes_tensor = output_dict_loss_scale.get('decoded_pred_boxes_obj')
                    print(
                        f"    decoded_pred_boxes_obj shape: {pred_boxes_tensor.shape if pred_boxes_tensor is not None else 'None'}")
                else:
                    print(f"    output_dict_loss_scale is None or not a dict.")
                if target_dict_loss_scale is not None and isinstance(target_dict_loss_scale, dict):
                    print(f"    Keys in target_dict_loss_scale: {list(target_dict_loss_scale.keys())}")
                    tgt_boxes_tensor = target_dict_loss_scale.get('decoded_target_boxes_obj')
                    print(
                        f"    decoded_target_boxes_obj shape: {tgt_boxes_tensor.shape if tgt_boxes_tensor is not None else 'None'}")
                else:
                    print(f"    target_dict_loss_scale is None or not a dict.")

            if nms_output is not None: yolo_nms_outputs_list.append(nms_output)
            # Dictionaries should now always be dicts, even if potentially empty internally
            yolo_output_dicts_for_loss.append(output_dict_loss_scale)
            yolo_target_dicts_for_loss.append(target_dict_loss_scale)

        final_outputs_for_nms = None
        if yolo_nms_outputs_list:
            valid_nms_outputs = [out for out in yolo_nms_outputs_list if out is not None and out.numel() > 0]
            if valid_nms_outputs:
                final_outputs_for_nms = torch.cat(valid_nms_outputs, 1)

        results_dict = {"yolo_outputs": final_outputs_for_nms}
        if targets is not None:
            results_dict["yolo_output_dicts"] = yolo_output_dicts_for_loss
            results_dict["yolo_target_dicts"] = yolo_target_dicts_for_loss

        if self.debug_mode and targets is None and results_dict["yolo_outputs"] is not None:
            final_output_tensor = results_dict["yolo_outputs"]
            if final_output_tensor.numel() > 0:
                final_confs = final_output_tensor[..., 4]
                if final_confs.numel() > 0:
                    print(
                        f"[DEBUG YoloDecoder forward VAL] Final 'yolo_outputs' conf stats: min={final_confs.min().item():.4f}, max={final_confs.max().item():.4f}, mean={final_confs.mean().item():.4f}")
                    conf_thresholds_to_check = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
                    for thres in conf_thresholds_to_check:
                        count_above_thres = (final_confs > thres).sum().item()
                        print(f"[DEBUG YoloDecoder forward VAL]   Preds with conf > {thres:.2f}: {count_above_thres}")

        if self.debug_mode: print(f"--- [DEBUG YoloDecoder forward END] ---\n")
        return results_dict
