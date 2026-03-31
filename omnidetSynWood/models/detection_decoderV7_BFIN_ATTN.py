import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union, Dict
from typing import List, Tuple, Dict, Optional, Any
from collections import OrderedDict

# --- Utility/Stub Functions ---
# Ensure these are correctly imported or implemented in your actual project
try:
    # Assuming these are part of your training/utility scripts
    from train_utils.detection_utils import build_targets, bbox_wh_iou, bbox_iou  # type: ignore

    print("Using build_targets, bbox_wh_iou, and bbox_iou from train_utils.detection_utils.")
except ImportError:
    print("Warning: Could not import detection_utils. Using placeholder stubs.")


    def build_targets(*args: Any, **kwargs: Any) -> Any:
        print("Warning: build_targets called but not implemented/imported.")
        # Expected to return: obj_mask, noobj_mask, tx, ty, tw, th, tcls,
        # raw_pred_x_obj, raw_pred_y_obj, raw_pred_w_obj, raw_pred_h_obj,
        # raw_pred_obj_conf_obj, raw_pred_cls_obj, raw_pred_obj_conf_noobj,
        # iou_scores, class_mask
        # This stub needs to be replaced by your actual implementation.
        raw_preds_arg = kwargs.get("raw_predictions")
        if raw_preds_arg is None:
            num_raw_preds = 1
        else:
            num_raw_preds = raw_preds_arg.shape[0]

        dummy_tensor = torch.empty(0)
        return (dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
                dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor, dummy_tensor,
                dummy_tensor, dummy_tensor)


    def bbox_wh_iou(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bbox_wh_iou not imported or defined.")


    def bbox_iou(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bbox_iou not imported or defined.")


# --- Common YOLO Modules ---

class Conv(nn.Module):
    # Standard convolution with BatchNorm and SiLU activation, now with fusion capability
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1, act: bool = True,
                 deploy: bool = False):
        super(Conv, self).__init__()
        self.deploy = deploy
        if p is None:
            p = k // 2 if k > 1 else 0

        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g,
                              bias=True if self.deploy else False)  # Bias True if deployed and fused
        if not self.deploy:
            self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

    def _fuse_kernel(self) -> None:
        """Fuses BatchNorm layer into the Conv2d layer for deployment."""
        if self.deploy or not hasattr(self, 'bn'):
            return

        bn_gamma = self.bn.weight
        bn_beta = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps
        conv_w = self.conv.weight

        std = (bn_var + bn_eps).sqrt()
        t = bn_gamma / std

        fused_conv_w = conv_w * t.reshape(-1, 1, 1, 1)
        fused_conv_b = bn_beta - bn_mean * t

        # Replace conv with a new one that has bias
        fused_conv_layer = nn.Conv2d(self.conv.in_channels,
                                     self.conv.out_channels,
                                     self.conv.kernel_size,
                                     self.conv.stride,
                                     self.conv.padding,
                                     groups=self.conv.groups,
                                     bias=True).to(conv_w.device)
        fused_conv_layer.weight.data = fused_conv_w
        fused_conv_layer.bias.data = fused_conv_b

        self.conv = fused_conv_layer
        del self.bn
        self.deploy = True
        # print(f"Fused Conv layer: {self}")


class SPPCSPC(nn.Module):
    # Spatial Pyramid Pooling – Cross Stage Partial module
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5,
                 k: Tuple[int, int, int] = (5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(c_ * (1 + len(k)), c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat([y1, y2], dim=1))


class RepConv(nn.Module):
    # RepConv is a Conv module with a training-time Reparameterization
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1, act: bool = True,
                 deploy: bool = False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        assert k == 3, "RepConv typically uses 3x3 kernel"
        assert p == k // 2, "Padding should be half of kernel size for RepConv 3x3"

        if deploy:
            self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
        else:
            self.rconv_bn = nn.BatchNorm2d(c2)
            self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)

            self.branch_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
            self.branch_bn = nn.BatchNorm2d(c2)

            self.branch_linear_bn = nn.BatchNorm2d(c2)
            self.branch_linear = nn.Conv2d(c1, c2, 1, s, groups=g, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rconv(x))
        else:
            main_path = self.rconv_bn(self.rconv(x))
            branch_path = self.branch_bn(self.branch_conv(x))
            linear_path = self.branch_linear_bn(self.branch_linear(x))
            return self.act(main_path + branch_path + linear_path)

    def _get_equivalent_kernel_bias(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        bn_gamma, bn_beta, bn_mean, bn_var, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
        std = (bn_var + bn_eps).sqrt()
        t = bn_gamma / std
        equivalent_kernel = conv.weight.data * t.reshape(-1, 1, 1, 1)
        equivalent_bias = bn_beta - bn_mean * t
        return equivalent_kernel, equivalent_bias

    def _fuse_kernel(self) -> None:
        if self.deploy or not hasattr(self, 'rconv_bn'):
            return

        kernel_rconv, bias_rconv = self._get_equivalent_kernel_bias(self.rconv, self.rconv_bn)
        kernel_branch, bias_branch = self._get_equivalent_kernel_bias(self.branch_conv, self.branch_bn)
        kernel_linear, bias_linear = self._get_equivalent_kernel_bias(self.branch_linear, self.branch_linear_bn)

        kernel_linear_padded = F.pad(kernel_linear, [1, 1, 1, 1])

        fused_kernel = kernel_rconv + kernel_branch + kernel_linear_padded
        fused_bias = bias_rconv + bias_branch + bias_linear

        fused_conv_layer = nn.Conv2d(self.in_channels, self.out_channels,
                                     kernel_size=self.rconv.kernel_size, stride=self.rconv.stride,
                                     padding=self.rconv.padding,
                                     groups=self.groups, bias=True).to(fused_kernel.device)
        fused_conv_layer.weight.data = fused_kernel
        fused_conv_layer.bias.data = fused_bias

        self.rconv = fused_conv_layer
        del self.rconv_bn, self.branch_conv, self.branch_bn, self.branch_linear, self.branch_linear_bn
        self.deploy = True
        # print(f"Fused RepConv layer: {self}")


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class BiFPNBlock(nn.Module):
    def __init__(self, in_channels_high_res: int, in_channels_low_res: int, out_channels: int, use_cbam: bool = False):
        super().__init__()
        self.conv_high_res = Conv(in_channels_high_res, out_channels, k=1, s=1)
        self.conv_low_res = Conv(in_channels_low_res, out_channels, k=1, s=1)
        self.weight_high_res = nn.Parameter(torch.ones(1))
        self.weight_low_res = nn.Parameter(torch.ones(1))
        self.relu = nn.ReLU()
        self.conv_after_fusion = Conv(out_channels, out_channels, k=3, s=1)
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, feature_high_res: torch.Tensor, feature_low_res: torch.Tensor) -> torch.Tensor:
        upsampled_low_res = F.interpolate(feature_low_res, size=feature_high_res.shape[2:], mode='nearest')
        transformed_high_res = self.conv_high_res(feature_high_res)
        transformed_low_res = self.conv_low_res(upsampled_low_res)

        weight_high_res_norm = self.relu(self.weight_high_res)
        weight_low_res_norm = self.relu(self.weight_low_res)
        epsilon = 1e-4
        fusion_weight_sum = weight_high_res_norm + weight_low_res_norm + epsilon
        fused_feature = (transformed_high_res * weight_high_res_norm +
                         transformed_low_res * weight_low_res_norm) / fusion_weight_sum
        fused_feature = self.conv_after_fusion(fused_feature)
        if self.use_cbam:
            fused_feature = self.cbam(fused_feature)
        return fused_feature


class YOLOLayer(nn.Module):
    def __init__(self, anchors: List[List[float]], args: Any):
        super().__init__()
        self.args = args
        self.debug_mode = getattr(args, 'debug_decoder', False)  # Debug flag
        self.anchors = anchors
        self.num_anchors = len(self.anchors)
        self.num_classes = self.args.num_classes_detection
        self.ignore_thres = float(self.args.ignore_thres)
        self.metrics: Dict[str, Any] = dict()
        self.img_dim = [self.args.input_width, self.args.input_height]
        self.grid_size: List[int] = [0, 0]
        self.stride: List[float] = [0.0, 0.0]
        self.in_channels_head: Optional[int] = None
        self.conv1: Optional[RepConv] = None
        self.conv2: Optional[RepConv] = None
        self.prediction: Optional[nn.Conv2d] = None
        self.register_buffer('grid_x', torch.empty(0))
        self.register_buffer('grid_y', torch.empty(0))
        self.register_buffer('scaled_anchors', torch.empty(0))
        self.register_buffer('anchor_w', torch.empty(0))
        self.register_buffer('anchor_h', torch.empty(0))
        self.output: Optional[torch.Tensor] = None  # Initialize self.output

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
            self.stride[1] = self.img_dim[1] / g_h
            self.stride[0] = self.img_dim[0] / g_w

        grid_x_list = [torch.arange(g_w, device=self.args.device) for _ in range(g_h)]
        self.grid_x = torch.stack(grid_x_list).view([1, 1, g_h, g_w])
        grid_y_list = [torch.arange(g_h, device=self.args.device) for _ in range(g_w)]
        self.grid_y = torch.stack(grid_y_list).t().view([1, 1, g_h, g_w])

        stride_w = self.stride[0] if self.stride[0] > 0 else 1.0
        stride_h = self.stride[1] if self.stride[1] > 0 else 1.0

        _scaled_anchors = torch.tensor([(a_w / stride_w, a_h / stride_h)
                                        for a_w, a_h in self.anchors], device=self.args.device)
        self.scaled_anchors = _scaled_anchors
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def _post_process_output_target(self, targets: Optional[torch.Tensor], raw_predictions: torch.Tensor,
                                    grid_shape: List[int], decoded_pred_conf: torch.Tensor) -> Tuple[
        Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.debug_mode:
            print(
                f"[DEBUG YOLOLayer _post_process_output_target] Targets shape: {targets.shape if targets is not None else 'None'}")
            print(f"[DEBUG YOLOLayer _post_process_output_target] Raw predictions shape: {raw_predictions.shape}")
            if decoded_pred_conf.numel() > 0:
                print(
                    f"[DEBUG YOLOLayer _post_process_output_target] Decoded pred_conf sum: {decoded_pred_conf.sum().item()}")
            else:
                print(f"[DEBUG YOLOLayer _post_process_output_target] Decoded pred_conf is empty.")

        if targets is None:
            return self.output, None, None
        else:
            # Call build_targets
            (obj_mask, noobj_mask, tx, ty, tw, th, tcls,
             raw_pred_x_obj, raw_pred_y_obj, raw_pred_w_obj, raw_pred_h_obj,
             raw_pred_obj_conf_obj, raw_pred_cls_obj,
             raw_pred_obj_conf_noobj,
             iou_scores, class_mask) = build_targets(
                raw_predictions=raw_predictions,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
                args=self.args
            )
            if self.debug_mode:
                print(
                    f"[DEBUG YOLOLayer build_targets output] obj_mask sum: {obj_mask.sum().item()}, noobj_mask sum: {noobj_mask.sum().item()}")
                if obj_mask.sum().item() > 0:  # Ensure there are positive samples before accessing masked tensors
                    if tx[obj_mask].numel() > 0: print(
                        f"[DEBUG YOLOLayer build_targets output] tx sum (pos): {tx[obj_mask].sum().item()}")
                    if tcls[obj_mask].numel() > 0: print(
                        f"[DEBUG YOLOLayer build_targets output] tcls sum (pos): {tcls[obj_mask].sum().item()}")
                    if raw_pred_x_obj.numel() > 0: print(
                        f"[DEBUG YOLOLayer build_targets output] raw_pred_x_obj sum: {raw_pred_x_obj.sum().item()}")
                    if raw_pred_obj_conf_obj.numel() > 0: print(
                        f"[DEBUG YOLOLayer build_targets output] raw_pred_obj_conf_obj sum: {raw_pred_obj_conf_obj.sum().item()}")
                if noobj_mask.sum().item() > 0:
                    if raw_pred_obj_conf_noobj.numel() > 0:
                        print(
                            f"[DEBUG YOLOLayer build_targets output] raw_pred_obj_conf_noobj sum: {raw_pred_obj_conf_noobj.sum().item()}")
                    else:
                        print(
                            f"[DEBUG YOLOLayer build_targets output] raw_pred_obj_conf_noobj is empty despite noobj_mask having positive sum.")

            target_dict = dict(
                x=tx[obj_mask], y=ty[obj_mask], w=tw[obj_mask], h=th[obj_mask],
                obj_conf=torch.ones_like(raw_pred_obj_conf_obj) if raw_pred_obj_conf_obj.numel() > 0 else torch.empty(0,
                                                                                                                      device=raw_pred_obj_conf_obj.device),
                no_obj_conf=torch.zeros_like(
                    raw_pred_obj_conf_noobj) if raw_pred_obj_conf_noobj.numel() > 0 else torch.empty(0,
                                                                                                     device=raw_pred_obj_conf_noobj.device),
                cls=tcls[obj_mask],
                iou_scores=iou_scores[obj_mask],
                class_mask=class_mask[obj_mask],
                obj_mask=obj_mask, noobj_mask=noobj_mask,
                grid_shapes=[grid_shape]
            )
            output_dict = dict(
                x=raw_pred_x_obj, y=raw_pred_y_obj, w=raw_pred_w_obj, h=raw_pred_h_obj,
                obj_conf=raw_pred_obj_conf_obj,
                no_obj_conf=raw_pred_obj_conf_noobj,
                cls=raw_pred_cls_obj,
                pred_conf=decoded_pred_conf,
                grid_shapes=[grid_shape]
            )
            return self.output, output_dict, target_dict

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                img_dim: Optional[Union[List[int], torch.Tensor]] = None) -> Tuple[
        Optional[torch.Tensor], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.in_channels_head is None or self.conv1 is None or self.conv2 is None or self.prediction is None:
            raise RuntimeError("YOLOLayer's layers not initialized. Call set_in_channels() first.")

        if self.debug_mode:
            print(f"\n[DEBUG YOLOLayer forward] Input x shape: {x.shape}, Targets provided: {targets is not None}")

        prediction_raw = self.prediction(self.conv2(self.conv1(x)))
        if self.debug_mode:
            print(f"[DEBUG YOLOLayer forward] prediction_raw shape: {prediction_raw.shape}")
            print(
                f"[DEBUG YOLOLayer forward] prediction_raw stats: min={prediction_raw.min().item():.4f}, max={prediction_raw.max().item():.4f}, mean={prediction_raw.mean().item():.4f}")

        if img_dim is not None:
            if torch.is_tensor(img_dim):
                img_dim_list = img_dim.cpu().tolist()
            else:
                img_dim_list = img_dim  # type: ignore
            self.img_dim = img_dim_list

        num_samples = prediction_raw.size(0)
        current_grid_shape = [prediction_raw.size(2), prediction_raw.size(3)]

        if self.grid_size != current_grid_shape or self.scaled_anchors.numel() == 0:
            self.compute_grid_offsets(current_grid_shape)

        num_of_output_params = 5 + self.num_classes
        prediction_reshaped = (prediction_raw.view(num_samples, self.num_anchors, num_of_output_params,
                                                   current_grid_shape[0], current_grid_shape[1]).permute(0, 1, 3, 4,
                                                                                                         2).contiguous())

        decoded_x = torch.sigmoid(prediction_reshaped[..., 0])
        decoded_y = torch.sigmoid(prediction_reshaped[..., 1])
        decoded_pred_conf = torch.sigmoid(prediction_reshaped[..., 4])
        decoded_pred_cls = torch.sigmoid(prediction_reshaped[..., 5:])
        raw_w = prediction_reshaped[..., 2]
        raw_h = prediction_reshaped[..., 3]

        if self.debug_mode:
            if decoded_pred_conf.numel() > 0:
                print(
                    f"[DEBUG YOLOLayer forward] decoded_pred_conf stats: min={decoded_pred_conf.min().item():.4f}, max={decoded_pred_conf.max().item():.4f}, mean={decoded_pred_conf.mean().item():.4f}")
                if decoded_pred_conf.max().item() > 0.01:
                    print(
                        f"[DEBUG YOLOLayer forward] Top 5 decoded_pred_conf: {torch.topk(decoded_pred_conf.flatten(), min(5, decoded_pred_conf.numel())).values.tolist()}")
            else:
                print(f"[DEBUG YOLOLayer forward] decoded_pred_conf is empty.")

        pred_boxes = torch.empty(prediction_reshaped[..., :4].shape, device=self.args.device)
        pred_boxes[..., 0] = (decoded_x * 2. - 0.5 + self.grid_x)
        pred_boxes[..., 1] = (decoded_y * 2. - 0.5 + self.grid_y)
        pred_boxes[..., 2] = (torch.sigmoid(raw_w) * 2).pow(2) * self.anchor_w
        pred_boxes[..., 3] = (torch.sigmoid(raw_h) * 2).pow(2) * self.anchor_h

        output_boxes_scaled = torch.empty_like(pred_boxes, device=self.args.device)
        output_boxes_scaled[..., 0] = pred_boxes[..., 0] * self.stride[0]
        output_boxes_scaled[..., 1] = pred_boxes[..., 1] * self.stride[1]
        output_boxes_scaled[..., 2] = pred_boxes[..., 2] * self.stride[0]
        output_boxes_scaled[..., 3] = pred_boxes[..., 3] * self.stride[1]

        self.output = torch.cat((output_boxes_scaled.view(num_samples, -1, 4),
                                 decoded_pred_conf.view(num_samples, -1, 1),
                                 decoded_pred_cls.view(num_samples, -1, self.num_classes)), -1).to(self.args.device)

        if self.debug_mode:
            print(f"[DEBUG YOLOLayer forward] self.output shape: {self.output.shape}")
            if self.output.numel() > 0:
                output_conf_scores = self.output[..., 4]
                if output_conf_scores.numel() > 0:
                    print(
                        f"[DEBUG YOLOLayer forward] self.output conf stats: min={output_conf_scores.min().item():.4f}, max={output_conf_scores.max().item():.4f}, mean={output_conf_scores.mean().item():.4f}")
                    if output_conf_scores.max().item() > 0.01:
                        print(
                            f"[DEBUG YOLOLayer forward] Top 5 self.output confs: {torch.topk(output_conf_scores.flatten(), min(5, output_conf_scores.numel())).values.tolist()}")
                else:
                    print(f"[DEBUG YOLOLayer forward] self.output conf_scores slice is empty.")
            else:
                print(f"[DEBUG YOLOLayer forward] self.output is empty.")

        return self._post_process_output_target(targets, prediction_raw, current_grid_shape, decoded_pred_conf)


class YoloDecoder(nn.Module):
    def __init__(self, _out_filters: List[int], args: Any):
        super(YoloDecoder, self).__init__()
        self.args = args
        self.debug_mode = getattr(args, 'debug_decoder', False)  # Debug flag
        self.no = 5 + args.num_classes_detection
        self.nc = args.num_classes_detection
        self.all_in_channels = _out_filters
        self.target_strides = [8, 16, 32]

        if len(self.all_in_channels) < 3:
            raise ValueError(f"Expected _out_filters for at least 3 maps, got {len(self.all_in_channels)}.")

        p3_backbone_channels = self.all_in_channels[-3]
        p4_backbone_channels = self.all_in_channels[-2]
        p5_backbone_channels = self.all_in_channels[-1]

        self.num_scales = len(self.target_strides)
        anchors_list = [list(args.anchors3), list(args.anchors2), list(args.anchors1)]
        self.num_anchors_per_scale = len(anchors_list[0])
        self.register_buffer('anchors', torch.tensor(anchors_list, dtype=torch.float32).view(self.num_scales, -1, 2).to(
            self.args.device))
        self.strides = [8, 16, 32]
        img_w, img_h = self.args.input_width, self.args.input_height
        self.grid_sizes = [[img_h // s, img_w // s] for s in self.strides]

        sppcspc_ch = getattr(args, 'neck_sppcspc_channels', 512)
        p5_to_p4_ch = sppcspc_ch // 2
        bifpn_p4_ch = getattr(args, 'neck_bifpn_p4_channels', 512)
        p4_to_p3_ch = bifpn_p4_ch // 2
        bifpn_p3_ch = getattr(args, 'neck_bifpn_p3_channels', 256)
        self.use_neck_cbam = getattr(args, 'use_neck_cbam', False)

        if self.debug_mode:
            print(
                f"[DEBUG YoloDecoder __init__] Backbone channels P3:{p3_backbone_channels}, P4:{p4_backbone_channels}, P5:{p5_backbone_channels}")
            print(
                f"[DEBUG YoloDecoder __init__] Neck configured channels: SPPCSPC_out:{sppcspc_ch}, P5_to_P4:{p5_to_p4_ch}, BiFPN_P4_out:{bifpn_p4_ch}, P4_to_P3:{p4_to_p3_ch}, BiFPN_P3_out:{bifpn_p3_ch}")
            print(
                f"[DEBUG YoloDecoder __init__] Using CBAM in neck: {self.use_neck_cbam}, CBAM in BiFPN: {getattr(args, 'use_bifpn_cbam', False)}")

        self.sppcspc = SPPCSPC(p5_backbone_channels, sppcspc_ch)
        if self.use_neck_cbam: self.cbam_p5 = CBAM(sppcspc_ch)

        self.conv_p5_to_p4_branch = Conv(sppcspc_ch, p5_to_p4_ch, k=1)
        self.upsample_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bifpn_block_p4 = BiFPNBlock(p4_backbone_channels, p5_to_p4_ch, bifpn_p4_ch,
                                         use_cbam=self.use_neck_cbam and getattr(args, 'use_bifpn_cbam', False))
        if self.use_neck_cbam and not getattr(args, 'use_bifpn_cbam', False): self.cbam_p4 = CBAM(bifpn_p4_ch)

        self.conv_p4_to_p3_branch = Conv(bifpn_p4_ch, p4_to_p3_ch, k=1)
        self.upsample_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bifpn_block_p3 = BiFPNBlock(p3_backbone_channels, p4_to_p3_ch, bifpn_p3_ch,
                                         use_cbam=self.use_neck_cbam and getattr(args, 'use_bifpn_cbam', False))
        if self.use_neck_cbam and not getattr(args, 'use_bifpn_cbam', False): self.cbam_p3 = CBAM(bifpn_p3_ch)

        self.head_p3 = YOLOLayer(anchors_list[0], self.args)
        self.head_p3.set_in_channels(bifpn_p3_ch)
        self.head_p3.compute_grid_offsets(self.grid_sizes[0])

        self.head_p4 = YOLOLayer(anchors_list[1], self.args)
        self.head_p4.set_in_channels(bifpn_p4_ch)
        self.head_p4.compute_grid_offsets(self.grid_sizes[1])

        self.head_p5 = YOLOLayer(anchors_list[2], self.args)
        self.head_p5.set_in_channels(sppcspc_ch)
        self.head_p5.compute_grid_offsets(self.grid_sizes[2])

        self.final_layers = nn.ModuleList([self.head_p3, self.head_p4, self.head_p5])
        self._init_biases()

    def _init_biases(self) -> None:
        for head in self.final_layers:
            if head.prediction is not None and head.prediction.bias is not None:
                b = head.prediction.bias.data.view(head.num_anchors, -1)
                prior = 0.01
                bias_value = -math.log((1 - prior) / prior)
                b[:, 4] = bias_value
                if self.debug_mode:
                    print(
                        f"[DEBUG YoloDecoder _init_biases] Initialized obj bias for head with anchors {head.anchors} to {bias_value:.3f}")

    def switch_to_deploy(self) -> None:
        print("Switching YoloDecoder to deploy mode...")
        for m_name, module in self.named_modules():
            if isinstance(module, RepConv):
                module._fuse_kernel()
            elif isinstance(module, Conv):
                module._fuse_kernel()
        print("YoloDecoder switched to deploy mode. RepConv and Conv layers fused.")

    def forward(self, backbone_features: List[torch.Tensor],
                img_dim: Optional[Union[List[int], torch.Tensor]] = None,
                targets: Optional[torch.Tensor] = None) -> Union[Dict[str, Any], torch.Tensor, None]:

        current_epoch_for_debug = getattr(self.args, 'current_epoch', 'N/A')
        current_step_for_debug = getattr(self.args, 'current_step', 'N/A')  # Corrected selection

        if self.debug_mode:
            print(
                f"\n--- [DEBUG YoloDecoder forward] --- Epoch: {current_epoch_for_debug}, Step: {current_step_for_debug} ---")
            print(f"[DEBUG YoloDecoder forward] Received {len(backbone_features)} backbone_features.")
            for i, f in enumerate(backbone_features):
                print(
                    f"[DEBUG YoloDecoder forward] Backbone feature {i} shape: {f.shape}, device: {f.device}, dtype: {f.dtype}, sum: {f.sum().item():.4f}")
            if targets is not None:
                print(
                    f"[DEBUG YoloDecoder forward] Targets shape: {targets.shape}, device: {targets.device}, num_targets: {targets.size(0)}")
                if targets.numel() > 0: print(
                    f"[DEBUG YoloDecoder forward] Example target (first): {targets[0].tolist()}")
            else:
                print(f"[DEBUG YoloDecoder forward] Targets: None (Validation/Inference Mode)")

        expected_p3_spatial = tuple(self.grid_sizes[0])
        expected_p4_spatial = tuple(self.grid_sizes[1])
        expected_p5_spatial = tuple(self.grid_sizes[2])

        p3_in, p4_in, p5_in = None, None, None
        for feature_map in backbone_features:
            spatial_size = (feature_map.shape[2], feature_map.shape[3])
            if spatial_size == expected_p3_spatial:
                p3_in = feature_map.to(self.args.device)
            elif spatial_size == expected_p4_spatial:
                p4_in = feature_map.to(self.args.device)
            elif spatial_size == expected_p5_spatial:
                p5_in = feature_map.to(self.args.device)

        if p3_in is None or p4_in is None or p5_in is None:
            found_shapes = [tuple(f.shape[2:]) for f in backbone_features]
            err_msg = (f"Could not find all P3/P4/P5 features. "
                       f"Expected P3={expected_p3_spatial}, P4={expected_p4_spatial}, P5={expected_p5_spatial}. "
                       f"Found shapes: {found_shapes}. ")
            if p3_in is None: err_msg += " P3 (stride 8 features) MISSING."
            if p4_in is None: err_msg += " P4 (stride 16 features) MISSING."
            if p5_in is None: err_msg += " P5 (stride 32 features) MISSING."
            raise ValueError(err_msg)

        if self.debug_mode:
            print(
                f"[DEBUG YoloDecoder forward] Identified P3_in: {p3_in.shape}, P4_in: {p4_in.shape}, P5_in: {p5_in.shape}")

        sppcspc_out = self.sppcspc(p5_in)
        if self.use_neck_cbam and hasattr(self, 'cbam_p5'): sppcspc_out = self.cbam_p5(sppcspc_out)
        if self.debug_mode: print(
            f"[DEBUG YoloDecoder forward] sppcspc_out shape: {sppcspc_out.shape}, sum: {sppcspc_out.sum().item():.4f}")

        p5_branch_for_p4 = self.conv_p5_to_p4_branch(sppcspc_out)
        x1_in_upsampled = self.upsample_p5_to_p4(p5_branch_for_p4)
        bifpn_p4_out = self.bifpn_block_p4(feature_high_res=p4_in, feature_low_res=x1_in_upsampled)
        if self.use_neck_cbam and hasattr(self, 'cbam_p4') and not getattr(self.args, 'use_bifpn_cbam', False):
            bifpn_p4_out = self.cbam_p4(bifpn_p4_out)
        if self.debug_mode: print(
            f"[DEBUG YoloDecoder forward] bifpn_p4_out shape: {bifpn_p4_out.shape}, sum: {bifpn_p4_out.sum().item():.4f}")

        p4_branch_for_p3 = self.conv_p4_to_p3_branch(bifpn_p4_out)
        x2_in_upsampled = self.upsample_p4_to_p3(p4_branch_for_p3)
        bifpn_p3_out = self.bifpn_block_p3(feature_high_res=p3_in, feature_low_res=x2_in_upsampled)
        if self.use_neck_cbam and hasattr(self, 'cbam_p3') and not getattr(self.args, 'use_bifpn_cbam', False):
            bifpn_p3_out = self.cbam_p3(bifpn_p3_out)
        if self.debug_mode: print(
            f"[DEBUG YoloDecoder forward] bifpn_p3_out shape: {bifpn_p3_out.shape}, sum: {bifpn_p3_out.sum().item():.4f}")

        neck_outputs_for_heads = [bifpn_p3_out, bifpn_p4_out, sppcspc_out]

        yolo_outputs_cat: List[Optional[torch.Tensor]] = []
        yolo_output_dict_list: List[Optional[Dict[str, Any]]] = []
        yolo_target_dict_list: List[Optional[Dict[str, Any]]] = []

        current_img_dim_val = img_dim
        if current_img_dim_val is None and hasattr(self.args, 'input_width') and hasattr(self.args, 'input_height'):
            current_img_dim_val = [self.args.input_width, self.args.input_height]
        elif current_img_dim_val is None:
            raise ValueError("img_dim is None and cannot be inferred from args.input_width/height")

        for i in range(self.num_scales):
            if self.debug_mode: print(
                f"[DEBUG YoloDecoder forward] Processing head {i} (P{i + 3}) with input shape {neck_outputs_for_heads[i].shape}")
            head_output, output_dict, target_dict = self.final_layers[i](neck_outputs_for_heads[i], targets,
                                                                         current_img_dim_val)
            yolo_outputs_cat.append(head_output)
            if output_dict: yolo_output_dict_list.append(output_dict)
            if target_dict: yolo_target_dict_list.append(target_dict)

        outputs_final: Dict[str, Any] = {}
        valid_outputs_for_cat = [out for out in yolo_outputs_cat if out is not None and out.numel() > 0]

        if valid_outputs_for_cat:
            outputs_final["yolo_outputs"] = torch.cat(valid_outputs_for_cat, 1)
            if self.debug_mode:
                final_output_tensor = outputs_final["yolo_outputs"]
                print(f"[DEBUG YoloDecoder forward] Final 'yolo_outputs' shape: {final_output_tensor.shape}")
                if final_output_tensor.numel() > 0:
                    final_confs = final_output_tensor[..., 4]
                    if final_confs.numel() > 0:  # Ensure final_confs is not empty
                        print(
                            f"[DEBUG YoloDecoder forward] Final 'yolo_outputs' conf stats: min={final_confs.min().item():.4f}, max={final_confs.max().item():.4f}, mean={final_confs.mean().item():.4f}")
                        if final_confs.max().item() > 0.01:
                            print(
                                f"[DEBUG YoloDecoder forward] Top 5 final_confs: {torch.topk(final_confs.flatten(), min(5, final_confs.numel())).values.tolist()}")
                            if targets is None:
                                print(
                                    f"[DEBUG YoloDecoder forward] Confidence Distribution in 'yolo_outputs' (for NMS):")
                                conf_thresholds_to_check = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                                for thres in conf_thresholds_to_check:
                                    count_above_thres = (final_confs > thres).sum().item()
                                    print(
                                        f"[DEBUG YoloDecoder forward]   Preds with conf > {thres:.2f}: {count_above_thres}")
                    else:
                        print(f"[DEBUG YoloDecoder forward] Final 'yolo_outputs' confidence slice is empty.")
                else:
                    print(
                        f"[DEBUG YoloDecoder forward] Final 'yolo_outputs' is empty after filtering Nones/empty tensors.")
        else:
            outputs_final["yolo_outputs"] = None
            if self.debug_mode:
                print(f"[DEBUG YoloDecoder forward] All head outputs were None or empty. 'yolo_outputs' is None.")

        if torch.onnx.is_in_onnx_export():
            return outputs_final.get("yolo_outputs", None)

        if targets is not None:
            outputs_final["yolo_output_dicts"] = yolo_output_dict_list
            outputs_final["yolo_target_dicts"] = yolo_target_dict_list
            if self.debug_mode and yolo_output_dict_list:
                for i, d in enumerate(yolo_output_dict_list):
                    if d and 'x' in d and d['x'] is not None and d['x'].numel() > 0:
                        print(
                            f"[DEBUG YoloDecoder forward] yolo_output_dict scale {i} has {d['x'].numel()} positive samples for loss.")
                    elif d:
                        print(
                            f"[DEBUG YoloDecoder forward] yolo_output_dict scale {i} is present but 'x' key is missing, None, or empty.")
                    else:
                        print(f"[DEBUG YoloDecoder forward] yolo_output_dict scale {i} is None.")

        if self.debug_mode: print(f"--- [DEBUG YoloDecoder forward END] ---\n")
        return outputs_final
