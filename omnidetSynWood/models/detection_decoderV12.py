import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Union, Dict, Optional, Any
from collections import OrderedDict

# --- Utility/Stub Functions ---
try:
    from train_utils.detection_utils import build_targets, bbox_wh_iou, bbox_iou, xywh2xyxy, \
        get_tensor_value  # Added get_tensor_value

    print("Successfully imported from train_utils.detection_utils.")
except ImportError:
    print("Warning: Could not import detection_utils. Using placeholder stubs for YoloDecoder analysis.")


    def build_targets(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_targets not imported.")


    def bbox_wh_iou(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bbox_wh_iou not imported.")


    def bbox_iou(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("bbox_iou not imported.")


    def xywh2xyxy(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("xywh2xyxy not imported.")


    def get_tensor_value(tensor: torch.Tensor) -> float:
        return tensor.item() if tensor is not None else 0.0


# --- Common YOLO Modules (Non-Attentional Versions used if use_attention=False) ---
class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: Union[int, Tuple[int, int]] = 1, s: int = 1,
                 p: Optional[Union[int, Tuple[int, int]]] = None, g: int = 1, act: bool = True, deploy: bool = False):
        super(Conv, self).__init__()
        self.deploy = deploy
        if p is None:
            if isinstance(k, int):
                p = k // 2
            elif isinstance(k, tuple):
                p = (k[0] // 2, k[1] // 2)
            else:
                p = 0
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True if self.deploy else False)
        if not self.deploy: self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

    def _fuse_kernel(self) -> None:
        if self.deploy or not hasattr(self, 'bn'): return
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


class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]] = ((3, 3), (3, 3)), e: float = 0.5):
        super().__init__();
        c_ = int(c2 * e)
        kernel_cv1, kernel_cv2 = k[0], k[1]
        groups_cv1 = g if not ((isinstance(kernel_cv1, int) and kernel_cv1 == 1) or (
                    isinstance(kernel_cv1, tuple) and kernel_cv1 == (1, 1))) else 1
        self.cv1 = Conv(c1, c_, k=kernel_cv1, s=1, act=nn.ReLU(inplace=True), g=groups_cv1)
        self.cv2 = Conv(c_, c2, k=kernel_cv2, s=1, g=g, act=nn.ReLU(inplace=True))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.cv2(self.cv1(x)) if self.add else self.cv2(
        self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__();
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=nn.ReLU(inplace=True))
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=nn.ReLU(inplace=True))
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=0.5) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1));
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__();
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU(inplace=True))
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=nn.ReLU(inplace=True))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x);
        y1 = self.m(x);
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__();
        self.avg_pool = nn.AdaptiveAvgPool2d(1);
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x));
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__();
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False);
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True);
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7):
        super().__init__();
        self.channel_attention = ChannelAttention(channels, reduction_ratio);
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ca = self.channel_attention(x);
        return x_ca * self.spatial_attention(x_ca)


class AttentionConv(Conv):
    def __init__(self, c1: int, c2: int, k: Union[int, Tuple[int, int]] = 1, s: int = 1,
                 p: Optional[Union[int, Tuple[int, int]]] = None, g: int = 1, act: bool = True, deploy: bool = False,
                 use_attention: bool = True, attention_reduction: int = 16):
        super().__init__(c1, c2, k, s, p, g, act, deploy)
        self.use_attention = use_attention
        if self.use_attention and c2 >= 16:
            self.attention_module = CBAM(c2, reduction_ratio=max(1, c2 // attention_reduction))
        else:
            self.attention_module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x); return self.attention_module(x)


class AttentionBottleneck(Bottleneck):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: Tuple[Union[int, Tuple[int, int]], Union[int, Tuple[int, int]]] = ((3, 3), (3, 3)), e: float = 0.5,
                 use_attention: bool = True):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.use_attention = use_attention
        if self.use_attention:
            c_ = int(c2 * e);
            kernel_cv1, kernel_cv2 = k[0], k[1]
            groups_cv1 = g if (isinstance(kernel_cv1, int) and kernel_cv1 != 1) or (
                        isinstance(kernel_cv1, tuple) and kernel_cv1 != (1, 1)) else 1
            self.cv1 = AttentionConv(c1, c_, k=kernel_cv1, s=1, g=groups_cv1, use_attention=True,
                                     act=nn.ReLU(inplace=True))
            self.cv2 = AttentionConv(c_, c2, k=kernel_cv2, s=1, g=g, use_attention=True, act=nn.ReLU(inplace=True))
            self.ms_attention = nn.Identity()
        else:
            self.ms_attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x));
        out = out + x if self.add else out
        return self.ms_attention(out)


class AttentionC2f(C2f):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5,
                 use_attention: bool = True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.use_attention = use_attention;
        self.c = int(c2 * e)
        self.cv1 = AttentionConv(c1, 2 * self.c, 1, 1, use_attention=self.use_attention, act=nn.ReLU(inplace=True))
        self.cv2 = AttentionConv((2 + n) * self.c, c2, 1, use_attention=self.use_attention, act=nn.ReLU(inplace=True))
        self.m = nn.ModuleList(AttentionBottleneck(self.c, self.c, shortcut=shortcut, g=g, k=((3, 3), (3, 3)), e=0.5,
                                                   use_attention=self.use_attention) for _ in range(n))
        if self.use_attention and c2 >= 32:
            self.fusion_attention = CBAM(c2)
        else:
            self.fusion_attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1));
        y.extend(m(y[-1]) for m in self.m)
        return self.fusion_attention(self.cv2(torch.cat(y, 1)))


class AttentionSPPF(SPPF):
    def __init__(self, c1: int, c2: int, k: int = 5, use_attention: bool = True):
        super().__init__(c1, c2, k)
        self.use_attention = use_attention;
        c_ = c1 // 2
        self.cv1 = AttentionConv(c1, c_, 1, 1, use_attention=self.use_attention, act=nn.ReLU(inplace=True))
        self.cv2 = AttentionConv(c_ * 4, c2, 1, 1, use_attention=self.use_attention, act=nn.ReLU(inplace=True))
        if self.use_attention and c_ * 4 >= 32:
            self.pool_attention = CBAM(c_ * 4)
        else:
            self.pool_attention = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x);
        y1 = self.m(x);
        y2 = self.m(y1);
        y3 = self.m(y2)
        concatenated = torch.cat((x, y1, y2, y3), 1)
        attended = self.pool_attention(concatenated)
        return self.cv2(attended)


class CrossScaleAttention(nn.Module):
    def __init__(self, channels_list: List[int], common_dim_ratio: float = 0.25):
        super().__init__();
        self.channels_list = channels_list
        if not channels_list: raise ValueError("channels_list cannot be empty")
        min_ch = min(channels_list) if channels_list else 128
        common_dim = max(16, int(min_ch * common_dim_ratio));
        common_dim = ((common_dim + 7) // 8) * 8
        self.projections = nn.ModuleList([Conv(ch, common_dim, k=1, act=False) for ch in channels_list])
        num_heads = max(1, common_dim // 64);
        self.cross_attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, batch_first=False,
                                                     dropout=0.1)
        self.out_projections = nn.ModuleList([Conv(common_dim, ch, k=1, act=False) for ch in channels_list])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(common_dim) for _ in channels_list])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != len(self.channels_list) or len(features) <= 1: return features
        projected_and_flattened = [];
        original_shapes = []
        for i, feat in enumerate(features):
            b, _, h, w = feat.shape;
            original_shapes.append((b, h, w))
            proj_feat = self.projections[i](feat)
            projected_and_flattened.append(proj_feat.flatten(2).permute(2, 0, 1))
        attended_outputs_reshaped = []
        for i in range(len(projected_and_flattened)):
            query = projected_and_flattened[i]
            context_list = [projected_and_flattened[j] for j in range(len(projected_and_flattened)) if j != i]
            context = torch.cat(context_list, dim=0) if context_list else query
            attended_feat_seq, _ = self.cross_attention(query, context, context)
            attended_feat_seq = self.layer_norms[i](query + attended_feat_seq)
            b, h, w = original_shapes[i];
            common_dim_val = attended_feat_seq.size(-1)
            attended_feat_hw_first = attended_feat_seq.permute(1, 2, 0).reshape(b, common_dim_val, h, w)
            attended_outputs_reshaped.append(self.out_projections[i](attended_feat_hw_first))
        return attended_outputs_reshaped


# --- YOLOLayer (Head - Adapted for new neck, outputs raw logits for conf/cls for BCEWithLogitsLoss) ---
class YOLOLayer(nn.Module):
    """Detection Head for YOLOv12-style decoder.
       Outputs raw logits for confidence and class for BCEWithLogitsLoss.
       Maintains original box decoding (tx,ty,exp(tw),exp(th)) for original build_targets & MSE box loss.
    """

    def __init__(self, anchors, args, in_channels_from_neck: int):
        super().__init__()
        self.args = args;
        self.anchors = anchors;
        self.num_anchors = len(anchors)
        self.num_classes = args.num_classes_detection
        self.ignore_thres = float(getattr(args, 'ignore_thres', 0.5))
        self.img_dim = [args.input_width, args.input_height];
        self.grid_size = [0, 0];
        self.stride = [0.0, 0.0]
        self.in_channels_head = in_channels_from_neck

        # Pre-convolution before the final prediction layer
        # Controlled by 'use_attention' and 'use_head_pre_conv_attention' flags from args
        use_overall_attention = getattr(args, 'use_attention', False)
        use_head_pre_conv_attention = getattr(args, 'use_head_pre_conv_attention', False) and use_overall_attention

        if use_head_pre_conv_attention and self.in_channels_head >= 16:
            self.pre_conv = AttentionConv(self.in_channels_head, self.in_channels_head, k=3, s=1,
                                          use_attention=True,
                                          attention_reduction=getattr(args, 'attention_reduction', 16),
                                          act=nn.ReLU(inplace=True))
        else:
            # Standard Conv if not using attention in pre_conv, or Identity if no pre_conv desired
            # For now, let's include a standard Conv if pre_conv attention is off.
            self.pre_conv = Conv(self.in_channels_head, self.in_channels_head, k=3, s=1, act=nn.ReLU(inplace=True))

        self.prediction_conv = nn.Conv2d(
            in_channels=self.in_channels_head,  # Output of pre_conv is still in_channels_head
            out_channels=self.num_anchors * (5 + self.num_classes),
            kernel_size=1
        ).to(self.args.device)

        # Bias initialization for the final prediction layer
        if hasattr(self.prediction_conv, 'bias') and self.prediction_conv.bias is not None:
            b = self.prediction_conv.bias.data.view(self.num_anchors, -1)
            # Initialize objectness bias towards background (predict low confidence initially)
            # Using a common heuristic from YOLOv5 for objectness prior
            obj_prior = 0.01
            b[:, 4] += math.log(obj_prior / (1 - obj_prior))  # For BCEWithLogitsLoss, this biases towards obj_prior
            # Initialize class bias (optional, can help if class imbalance is severe)
            if self.num_classes > 0:  # Avoid division by zero if nc=0 (though nc should be >0)
                cls_prior = 0.01  # Small prior for classes
                b[:, 5:] += math.log(cls_prior / (1 - cls_prior))  # For BCEWithLogitsLoss
            self.prediction_conv.bias.data = b.view(-1)

        self.register_buffer('grid_x', torch.empty(0));
        self.register_buffer('grid_y', torch.empty(0))
        self.register_buffer('scaled_anchors', torch.empty(0));
        self.register_buffer('anchor_w', torch.empty(0));
        self.register_buffer('anchor_h', torch.empty(0))

        # For loss_dict (raw values needed by original ObjectDetectionLoss for x,y,w,h)
        self.x_sigmoid_for_loss: Optional[torch.Tensor] = None
        self.y_sigmoid_for_loss: Optional[torch.Tensor] = None
        self.w_raw_for_loss: Optional[torch.Tensor] = None
        self.h_raw_for_loss: Optional[torch.Tensor] = None

        # For loss_dict (raw logits needed for BCEWithLogitsLoss for conf, cls)
        self.obj_conf_logits_for_loss: Optional[torch.Tensor] = None
        self.cls_logits_for_loss: Optional[torch.Tensor] = None

        # For build_targets (sigmoid probabilities for pred_cls) & NMS (sigmoid for conf, cls)
        self.pred_conf_probs_for_nms_buildtargets: Optional[torch.Tensor] = None
        self.pred_cls_probs_for_nms_buildtargets: Optional[torch.Tensor] = None

        self.pred_boxes_for_buildtargets: Optional[torch.Tensor] = None  # Decoded boxes (cx,cy,gw,gh on grid)
        self.output_for_nms: Optional[torch.Tensor] = None  # Final NMS output (x1y1x2y2_img, obj_prob, cls_probs)

    def compute_grid_offsets(self, grid_size_hw):
        self.grid_size = grid_size_hw;
        g_h, g_w = grid_size_hw[0], grid_size_hw[1]
        if self.img_dim[0] == 0 or self.img_dim[1] == 0: self.img_dim = [self.args.input_width, self.args.input_height]
        self.stride = [self.img_dim[0] / g_w, self.img_dim[1] / g_h]
        self.grid_x = torch.arange(g_w, device=self.args.device).float().view(1, 1, 1, g_w).expand(1, self.num_anchors,
                                                                                                   g_h, g_w)
        self.grid_y = torch.arange(g_h, device=self.args.device).float().view(1, 1, g_h, 1).expand(1, self.num_anchors,
                                                                                                   g_h, g_w)
        self.scaled_anchors = torch.tensor([(aw / self.stride[0], ah / self.stride[1]) for aw, ah in self.anchors],
                                           device=self.args.device)
        self.anchor_w = self.scaled_anchors[:, 0:1].view(1, self.num_anchors, 1, 1)
        self.anchor_h = self.scaled_anchors[:, 1:2].view(1, self.num_anchors, 1, 1)

    def _post_proccess_output_target(self, targets):
        device = self.args.device
        output_dict = {  # For log_metrics and original loss
            "pred_conf": self.pred_conf_probs_for_nms_buildtargets  # Full grid probabilities
        }
        target_dict = {}
        if targets is None: return self.output_for_nms, output_dict, target_dict

        iou_s, cls_m, obj_m_f, noobj_m_f, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=self.pred_boxes_for_buildtargets,  # Original style decoded boxes
            pred_cls=self.pred_cls_probs_for_nms_buildtargets,  # Sigmoid class probabilities
            target=targets,
            anchors=self.scaled_anchors, ignore_thres=self.ignore_thres, args=self.args)

        obj_m_b, noobj_m_b = obj_m_f.bool(), noobj_m_f.bool()

        target_dict.update({  # For original loss and log_metrics
            "x": tx[obj_m_b], "y": ty[obj_m_b], "w": tw[obj_m_b], "h": th[obj_m_b],
            "obj_conf": tconf[obj_m_b], "no_obj_conf": tconf[noobj_m_b], "cls": tcls[obj_m_b],
            "iou_scores": iou_s, "tconf": tconf, "class_mask": cls_m, "obj_mask": obj_m_b})

        output_dict.update({  # For original loss and log_metrics
            "x": self.x_sigmoid_for_loss[obj_m_b], "y": self.y_sigmoid_for_loss[obj_m_b],
            "w": self.w_raw_for_loss[obj_m_b], "h": self.h_raw_for_loss[obj_m_b],
            "obj_conf": self.obj_conf_logits_for_loss[obj_m_b],  # Pass raw logits for obj
            "no_obj_conf": self.obj_conf_logits_for_loss[noobj_m_b],  # Pass raw logits for no_obj
            "cls": self.cls_logits_for_loss[obj_m_b]  # Pass raw logits for cls
            # "pred_conf" is already full grid probabilities
        })
        return self.output_for_nms, output_dict, target_dict

    def forward(self, x_in, targets=None, img_dim=None):
        if img_dim is not None:
            if torch.is_tensor(img_dim):
                self.img_dim = img_dim.cpu().tolist()
            else:
                self.img_dim = img_dim
        elif self.img_dim[0] == 0 or self.img_dim[1] == 0:
            self.img_dim = [self.args.input_width, self.args.input_height]

        features_after_pre_conv = self.pre_conv(x_in)
        prediction_logits_map = self.prediction_conv(features_after_pre_conv)

        num_samples, grid_size_hw = prediction_logits_map.size(0), [prediction_logits_map.size(2),
                                                                    prediction_logits_map.size(3)]
        pred_logits_reshaped = prediction_logits_map.view(num_samples, self.num_anchors, self.num_classes + 5,
                                                          grid_size_hw[0], grid_size_hw[1]).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        # For Loss (original style for x,y,w,h; logits for conf/cls)
        self.x_sigmoid_for_loss = torch.sigmoid(pred_logits_reshaped[..., 0])
        self.y_sigmoid_for_loss = torch.sigmoid(pred_logits_reshaped[..., 1])
        self.w_raw_for_loss = pred_logits_reshaped[..., 2]
        self.h_raw_for_loss = pred_logits_reshaped[..., 3]
        self.obj_conf_logits_for_loss = pred_logits_reshaped[..., 4]
        self.cls_logits_for_loss = pred_logits_reshaped[..., 5:]

        # For NMS and build_targets (needs probabilities for conf/cls)
        self.pred_conf_probs_for_nms_buildtargets = torch.sigmoid(self.obj_conf_logits_for_loss)
        self.pred_cls_probs_for_nms_buildtargets = torch.sigmoid(self.cls_logits_for_loss)

        if self.grid_size != grid_size_hw or self.anchor_w.numel() == 0: self.compute_grid_offsets(grid_size_hw)

        # Box decoding for build_targets (original OmniDet style)
        pred_cx_g = self.x_sigmoid_for_loss + self.grid_x  # Using sigmoid(tx)
        pred_cy_g = self.y_sigmoid_for_loss + self.grid_y  # Using sigmoid(ty)
        pred_w_g = torch.exp(self.w_raw_for_loss) * self.anchor_w
        pred_h_g = torch.exp(self.h_raw_for_loss) * self.anchor_h
        self.pred_boxes_for_buildtargets = torch.stack((pred_cx_g, pred_cy_g, pred_w_g, pred_h_g), dim=-1)

        # Box decoding for NMS output (x1y1x2y2 image scale)
        out_cx_img = pred_cx_g * self.stride[0];
        out_cy_img = pred_cy_g * self.stride[1]
        out_w_img = pred_w_g * self.stride[0];
        out_h_img = pred_h_g * self.stride[1]
        self.output_for_nms = torch.cat(((out_cx_img - out_w_img / 2).view(num_samples, -1, 1),
                                         (out_cy_img - out_h_img / 2).view(num_samples, -1, 1),
                                         (out_cx_img + out_w_img / 2).view(num_samples, -1, 1),
                                         (out_cy_img + out_h_img / 2).view(num_samples, -1, 1),
                                         self.pred_conf_probs_for_nms_buildtargets.view(num_samples, -1, 1),
                                         self.pred_cls_probs_for_nms_buildtargets.view(num_samples, -1,
                                                                                       self.num_classes)), -1)
        return self._post_proccess_output_target(targets)


# --- YOLOv12-Enhanced Decoder ---
class YoloDecoder(nn.Module):
    def __init__(self, _out_filters: List[int], args: Any):
        super().__init__()
        self.args = args;
        self.debug_mode = getattr(args, 'debug_decoder', False)
        self.use_attention = getattr(args, 'use_attention', False)
        self.use_cross_scale_attention = getattr(args, 'use_cross_scale_attention', False) and self.use_attention

        _out_filters_int = [int(ch) for ch in _out_filters]
        if len(_out_filters_int) < 5: raise ValueError(
            f"YoloV12Decoder expects _out_filters for at least 5 stages. Got {len(_out_filters_int)}")
        p3_ch_bb, p4_ch_bb, p5_ch_bb = _out_filters_int[2], _out_filters_int[3], _out_filters_int[4]

        self.nc = args.num_classes_detection;
        self.target_strides = [8, 16, 32]
        img_w, img_h = args.input_width, args.input_height
        self.grid_sizes = [[img_h // s, img_w // s] for s in self.target_strides]
        self.anchors_list_p3_p4_p5 = [list(args.anchors3), list(args.anchors2), list(args.anchors1)]

        def get_ch_config(key, default_val):
            val = getattr(args, key, default_val)
            try:
                return int(val)
            except (TypeError, ValueError):
                if self.debug_mode: print(
                    f"Warning: Config key '{key}' yielded non-integer '{val}' (type {type(val)}). Using default: {int(default_val)}")
                return int(default_val)

        self.head_in_ch_p3 = get_ch_config('neck_p3_head_input_channels', p3_ch_bb + 128)
        self.head_in_ch_p4 = get_ch_config('neck_p4_head_input_channels', p4_ch_bb + 256)
        self.head_in_ch_p5 = get_ch_config('neck_p5_head_input_channels', p5_ch_bb)

        sppf_c2 = get_ch_config('neck_sppf_c2', self.head_in_ch_p5 // 2 if self.head_in_ch_p5 > 0 else 256)
        fpn_p5_out_for_pan = get_ch_config('neck_fpn_p5_out_for_pan', sppf_c2)
        conv_p5_to_p4_ch = get_ch_config('neck_conv_p5_to_p4_ch', fpn_p5_out_for_pan // 2)
        fpn_p4_out_for_pan = get_ch_config('neck_fpn_p4_out_for_pan',
                                           self.head_in_ch_p4 // 2 if self.head_in_ch_p4 > 0 else 256)
        conv_p4_to_p3_ch = get_ch_config('neck_conv_p4_to_p3_ch', fpn_p4_out_for_pan // 2)
        pan_p3_ds_ch = get_ch_config('neck_pan_p3_ds_ch', self.head_in_ch_p3)
        pan_p4_ds_ch = get_ch_config('neck_pan_p4_ds_ch', self.head_in_ch_p4)

        if self.debug_mode:  # Print statements for channel verification
            print(
                f"[DEBUG YoloV12Decoder __init__] Using Attention: {self.use_attention}, CrossScaleAttention: {self.use_cross_scale_attention}")
            # ... (rest of debug prints) ...
            #self.debug_anchor_scales_internal()

        ConvBlk = AttentionConv if self.use_attention else Conv
        C2fBlk = AttentionC2f if self.use_attention else C2f
        SPPFBlk = AttentionSPPF if self.use_attention else SPPF
        att_kwargs = {'use_attention': True, 'attention_reduction': getattr(args, 'attention_reduction', 16)}

        self.sppf = SPPFBlk(p5_ch_bb, sppf_c2, k=5, **(att_kwargs if isinstance(SPPFBlk, AttentionSPPF) else {}))
        self.conv_fpn_p5 = ConvBlk(sppf_c2, fpn_p5_out_for_pan, 1, 1,
                                   **(att_kwargs if isinstance(ConvBlk, AttentionConv) else {}))
        self.conv_p5_to_p4_branch = ConvBlk(fpn_p5_out_for_pan, conv_p5_to_p4_ch, 1, 1,
                                            **(att_kwargs if isinstance(ConvBlk, AttentionConv) else {}))
        self.upsample_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p4_fpn = C2fBlk(p4_ch_bb + conv_p5_to_p4_ch, fpn_p4_out_for_pan, n=3, shortcut=False,
                                 **(att_kwargs if isinstance(C2fBlk, AttentionC2f) else {}))
        self.conv_p4_to_p3_branch = ConvBlk(fpn_p4_out_for_pan, conv_p4_to_p3_ch, 1, 1,
                                            **(att_kwargs if isinstance(ConvBlk, AttentionConv) else {}))
        self.upsample_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_p3_fpn = C2fBlk(p3_ch_bb + conv_p4_to_p3_ch, self.head_in_ch_p3, n=3, shortcut=False,
                                 **(att_kwargs if isinstance(C2fBlk, AttentionC2f) else {}))

        self.downsample_p3_to_p4 = ConvBlk(self.head_in_ch_p3, pan_p3_ds_ch, 3, 2,
                                           **(att_kwargs if isinstance(ConvBlk, AttentionConv) else {}))
        self.c2f_p4_pan = C2fBlk(pan_p3_ds_ch + fpn_p4_out_for_pan, self.head_in_ch_p4, n=3, shortcut=False,
                                 **(att_kwargs if isinstance(C2fBlk, AttentionC2f) else {}))
        self.downsample_p4_to_p5 = ConvBlk(self.head_in_ch_p4, pan_p4_ds_ch, 3, 2,
                                           **(att_kwargs if isinstance(ConvBlk, AttentionConv) else {}))
        self.c2f_p5_pan = C2fBlk(pan_p4_ds_ch + fpn_p5_out_for_pan, self.head_in_ch_p5, n=3, shortcut=False,
                                 **(att_kwargs if isinstance(C2fBlk, AttentionC2f) else {}))

        if self.use_cross_scale_attention:
            self.cross_scale_attention = CrossScaleAttention(
                [self.head_in_ch_p3, self.head_in_ch_p4, self.head_in_ch_p5])
        else:
            self.cross_scale_attention = None

        self.final_layer0 = YOLOLayer(self.anchors_list_p3_p4_p5[2], args, in_channels_from_neck=self.head_in_ch_p5)
        self.final_layer1 = YOLOLayer(self.anchors_list_p3_p4_p5[1], args, in_channels_from_neck=self.head_in_ch_p4)
        self.final_layer2 = YOLOLayer(self.anchors_list_p3_p4_p5[0], args, in_channels_from_neck=self.head_in_ch_p3)

    def debug_anchor_scales_internal(self, anchors_list_p3_p4_p5_ordered):
        print("\n=== ANCHOR ANALYSIS (Pixel values vs. Grid units) ===")
        for i, (anchors_on_scale, stride_val) in enumerate(zip(anchors_list_p3_p4_p5_ordered, self.target_strides)):
            print(f"Head for P{i + 3} (stride {stride_val}):")
            for j, (w, h) in enumerate(anchors_on_scale):
                scaled_w, scaled_h = float(w) / stride_val, float(h) / stride_val
                print(f"  Anchor {j}: {float(w):.1f}x{float(h):.1f} -> on grid: {scaled_w:.2f}x{scaled_h:.2f}")
            print()

    def forward(self, backbone_features: List[torch.Tensor], img_dim: List[int],
                targets: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        device = self.args.device;
        backbone_features = [feat.to(device) for feat in backbone_features]
        p3_bb, p4_bb, p5_bb = backbone_features[-3], backbone_features[-2], backbone_features[-1]

        x_p5_sppf = self.sppf(p5_bb)
        p5_fpn_conv_out = self.conv_fpn_p5(x_p5_sppf)
        p5_branch_for_p4 = self.conv_p5_to_p4_branch(p5_fpn_conv_out)
        p4_up = self.upsample_p5_to_p4(p5_branch_for_p4)
        p4_fpn_out = self.c2f_p4_fpn(torch.cat([p4_up, p4_bb], 1))
        p4_branch_for_p3 = self.conv_p4_to_p3_branch(p4_fpn_out)
        p3_up = self.upsample_p4_to_p3(p4_branch_for_p3)
        p3_fpn_head_feed = self.c2f_p3_fpn(torch.cat([p3_up, p3_bb], 1))

        p3_ds = self.downsample_p3_to_p4(p3_fpn_head_feed)
        p4_pan_head_feed = self.c2f_p4_pan(torch.cat([p3_ds, p4_fpn_out], 1))
        p4_ds = self.downsample_p4_to_p5(p4_pan_head_feed)
        p5_pan_head_feed = self.c2f_p5_pan(torch.cat([p4_ds, p5_fpn_conv_out], 1))

        neck_outputs = [p3_fpn_head_feed, p4_pan_head_feed, p5_pan_head_feed]
        if self.cross_scale_attention is not None: neck_outputs = self.cross_scale_attention(neck_outputs)
        p3_final_in, p4_final_in, p5_final_in = neck_outputs[0], neck_outputs[1], neck_outputs[2]

        outputs = dict();
        yolo_outputs_list = [];
        yolo_output_dict_list = [];
        yolo_target_dict_list = []

        out_l2, out_dict2, tgt_dict2 = self.final_layer2(p3_final_in, targets, img_dim)
        out_l1, out_dict1, tgt_dict1 = self.final_layer1(p4_final_in, targets, img_dim)
        out_l0, out_dict0, tgt_dict0 = self.final_layer0(p5_final_in, targets, img_dim)

        yolo_outputs_list = [out_l0, out_l1, out_l2]
        yolo_output_dict_list = [out_dict0, out_dict1, out_dict2]
        yolo_target_dict_list = [tgt_dict0, tgt_dict1, tgt_dict2]

        if not torch.onnx.is_in_onnx_export():
            valid_nms = [o for o in yolo_outputs_list if o is not None and o.numel() > 0]
            batch_size = backbone_features[0].size(0) if backbone_features and len(backbone_features) > 0 and \
                                                         backbone_features[0] is not None else 0
            outputs["yolo_outputs"] = torch.cat(valid_nms, 1).detach().cpu() if valid_nms else torch.empty(batch_size,
                                                                                                           0,
                                                                                                           5 + self.nc,
                                                                                                           device='cpu')
            outputs["yolo_output_dicts"] = [d for d in yolo_output_dict_list if d] if targets is not None else None
            outputs["yolo_target_dicts"] = [d for d in yolo_target_dict_list if d] if targets is not None else None
            return outputs
        else:
            valid_nms = [o for o in yolo_outputs_list if o is not None and o.numel() > 0]
            batch_size = backbone_features[0].size(0) if backbone_features and len(backbone_features) > 0 and \
                                                         backbone_features[0] is not None else 0
            return torch.cat(valid_nms, 1).detach().cpu() if valid_nms else torch.empty(batch_size, 0, 5 + self.nc,
                                                                                        device='cpu')

