"""
Detection decoder model for OmniDet.
Adapted from PyTorch-YOLOv3 and YOLOv7 implementations.
"""


import torch
import torch.nn as nn
import math # Needed for SPPCSPC and potentially other modules
import torch.nn.functional as F # Needed for F.conv2d
import numpy as np # Needed for numpy operations
from typing import List, Dict # Import for type hints
from collections import OrderedDict # Needed for OrderedDict in _make_cbl


# Import necessary utility functions from your detection_utils
# Ensure these are correctly imported and available
try:
   from train_utils.detection_utils import build_targets, bbox_wh_iou, bbox_iou # Keeping imports as in the original code
   print("Using build_targets, bbox_wh_iou, and bbox_iou from train_utils.detection_utils.")
except ImportError:
   print("Warning: Could not import detection_utils. Assuming necessary functions are defined elsewhere.")
   def build_targets(*args, **kwargs):
       raise NotImplementedError("build_targets not imported or defined.")
   def bbox_wh_iou(*args, **kwargs):
       raise NotImplementedError("bbox_wh_iou not imported or defined.")
   def bbox_iou(*args, **kwargs):
       raise NotImplementedError("bbox_iou not imported or defined.")




# --- Common YOLOv7 Modules ---
# Includes standard Conv, SPPCSPC, E_ELAN, and RepConv (with fusion logic).


class Conv(nn.Module):
   # Standard convolution with BatchNorm and SiLU activation
   def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
       super(Conv, self).__init__()
       # Calculate padding if not provided
       if p is None:
            p = k // 2 if k > 1 else 0
       self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
       self.bn   = nn.BatchNorm2d(c2)
       # Using SiLU activation (Standard in YOLOv7)
       self.act  = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


   def forward(self, x):
       return self.act(self.bn(self.conv(x)))


   def fuseforward(self, x):
       # Forward pass used when fusing BN for inference
       # Note: This fuseforward still applies the activation after fusion.
       return self.act(self.conv(x))




class SPPCSPC(nn.Module):
   # Spatial Pyramid Pooling – Cross Stage Partial module
   # Processes the deepest feature map for multi-scale pooling and feature aggregation.
   def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5,9,13)):
       super(SPPCSPC, self).__init__()
       # Adjusted hidden channels calculation to match typical YOLOv7 structure (c2 * expansion)
       c_ = int(c2 * e)  # hidden channels for internal convolutions
       self.cv1 = Conv(c1, c_, 1, 1) # Initial 1x1 conv
       self.cv2 = Conv(c1, c_, 1, 1) # Shortcut 1x1 conv
       self.cv3 = Conv(c_, c_, 3, 1) # 3x3 conv after cv1
       self.cv4 = Conv(c_, c_, 1, 1) # 1x1 conv after cv3
       # Max pooling layers with different kernel sizes
       self.m   = nn.ModuleList([nn.MaxPool2d(x, 1, padding=x//2) for x in k])
       # Conv after concatenating pooled features
       self.cv5 = Conv(4*c_, c_, 1, 1)
       self.cv6 = Conv(c_, c_, 3, 1) # 3x3 conv after cv5
       # Final conv after concatenating with shortcut
       self.cv7 = Conv(2*c_, c2, 1, 1)


   def forward(self, x):
       x1 = self.cv4(self.cv3(self.cv1(x)))
       y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
       y2 = self.cv2(x) # Shortcut connection
       return self.cv7(torch.cat([y1, y2], 1)) # Concatenate and final conv




class E_ELAN(nn.Module):
   # Extended efficient layer aggregation network module
   # Replaces the standard Darknet-style embedding blocks in the neck.
   # Simplified constructor to take only input and output channels.
   def __init__(self, in_ch, out_ch, e=0.5):
       super(E_ELAN, self).__init__()
       hidden = int(in_ch * e)  # hidden channels for the main branch
       # Adjusted split ratio to match typical YOLOv7 E-ELAN structure
       split = hidden // 4 # channels for the split branches (output of initial split convs)


       self.cv1 = Conv(in_ch, hidden, 1, 1) # Main branch initial conv (in_ch -> hidden)


       # Parallel branches, each starting with a conv from hidden to split
       self.cv2 = Conv(hidden, split, 1, 1) # Split branch 1 conv (hidden -> split)
       self.cv3 = Conv(split, split, 3, 1) # Split branch 1 conv (split -> split)


       self.cv4 = Conv(hidden, split, 1, 1) # Split branch 2 conv (hidden -> split)
       self.cv5 = Conv(split, split, 3, 1) # Split branch 2 conv (split -> split)


       self.cv6 = Conv(hidden, split, 1, 1) # Split branch 3 conv (hidden -> split)
       self.cv7 = Conv(split, split, 3, 1) # Split branch 3 conv (split -> split)


       self.cv8 = Conv(hidden, split, 1, 1) # Split branch 4 conv (hidden -> split)
       self.cv9 = Conv(split, split, 3, 1) # Split branch 4 conv (split -> split)


       # Concatenate the outputs of the four split branches and the initial conv
       # Channels: hidden (from x1) + split*4 (from x3, x5, x7, x9) = hidden + (hidden/4)*4 = 2*hidden
       self.cv10 = Conv(hidden + split * 4, out_ch, 1, 1) # Final conv to output channels (2*hidden -> out_ch)




   def forward(self, x):
       x1 = self.cv1(x) # in_ch -> hidden


       # Parallel branches
       x2 = self.cv2(x1) # hidden -> split
       x3 = self.cv3(x2) # split -> split


       x4 = self.cv4(x1) # hidden -> split
       x5 = self.cv5(x4) # split -> split


       x6 = self.cv6(x1) # hidden -> split
       x7 = self.cv7(x6) # split -> split


       x8 = self.cv8(x1) # hidden -> split
       x9 = self.cv9(x8) # split -> split


       # Concatenate along the channel dimension
       return self.cv10(torch.cat((x1, x3, x5, x7, x9), dim=1)) # hidden + 4*split -> out_ch




class RepConv(nn.Module):
   # RepConv is a Conv module with a training-time Reparameterization
   # This includes the logic to fuse the branches for inference.
   # The 'deploy' attribute is used to switch between training and inference modes.


   def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True, deploy=False):
       super(RepConv, self).__init__()
       self.deploy = deploy
       self.groups = g
       self.in_channels = c1
       self.out_channels = c2
       # Using SiLU activation
       self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())


       assert k == 3 # RepConv typically uses 3x3 kernel
       assert p == k // 2 # Padding should be half of kernel size


       if deploy:
           # In deploy mode, only the fused convolution exists
           self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
       else:
           # Training-time branches
           self.rconv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
           self.rconv_bn = nn.BatchNorm2d(c2)


           self.branch_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
           self.branch_bn = nn.BatchNorm2d(c2)


           # Linear branch (1x1 conv)
           self.branch_linear = nn.Conv2d(c1, c2, 1, s, groups=g, bias=False) # Corrected: Removed p=1
           self.branch_linear_bn = nn.BatchNorm2d(c2)


   def forward(self, x):
       if self.deploy:
           # Inference-time forward pass (fused kernel)
           return self.act(self.rconv(x))
       else:
           # Training-time forward pass with multiple branches
           return self.act(self.rconv_bn(self.rconv(x)) + self.branch_bn(self.branch_conv(x)) + self.branch_linear_bn(self.branch_linear(x)))


   # Helper function to get equivalent kernel and bias from a Conv + BN pair
   def _get_equivalent_kernel_bias(self, conv, bn):
       """
       Calculates the equivalent kernel and bias for a Conv + BN layer.
       Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
       """
       # Get BN stats
       bn_gamma, bn_beta, bn_mean, bn_var, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps


       # Calculate equivalent kernel and bias
       std = (bn_var + bn_eps).sqrt()
       t = bn_gamma / std # Scaling factor for weights and bias


       # Reshape t for broadcasting with weights
       t = t.reshape(-1, 1, 1, 1)


       # Equivalent kernel
       equivalent_kernel = conv.weight.data * t


       # Equivalent bias
       # Original conv bias is 0 since bias=False
       equivalent_bias = bn_beta - bn_mean * bn_gamma / std


       return equivalent_kernel, equivalent_bias


   # Helper function to fuse a Conv + BN pair into a single Conv
   def _fuse_conv_bn(self, conv, bn):
       """Fuses a Conv + BN layer into a single Conv layer."""
       fused_kernel, fused_bias = self._get_equivalent_kernel_bias(conv, bn)
       # Create a new Conv layer with the fused kernel and bias
       fused_conv = nn.Conv2d(self.in_channels, self.out_channels,
                              self.rconv.kernel_size, self.rconv.stride, self.rconv.padding,
                              groups=self.groups, bias=True).to(conv.weight.device)
       fused_conv.weight.data = fused_kernel
       fused_conv.bias.data = fused_bias
       return fused_conv


   # Method to fuse the training branches into a single kernel and bias
   def _fuse_kernel(self):
       """Fuses the training-time branches into a single convolutional kernel and bias."""
       if hasattr(self, 'rconv_bn'): # Check if in training mode
           # Get equivalent kernel and bias for each branch
           kernel_rconv, bias_rconv = self._get_equivalent_kernel_bias(self.rconv, self.rconv_bn)
           kernel_branch, bias_branch = self._get_equivalent_kernel_bias(self.branch_conv, self.branch_bn)


           # Handle the linear branch (1x1 conv). Need to pad the 1x1 kernel to 3x3.
           kernel_linear, bias_linear = self._get_equivalent_kernel_bias(self.branch_linear, self.branch_linear_bn)
           # Pad the 1x1 kernel to 3x3 by adding zeros
           kernel_linear_padded = F.pad(kernel_linear, [1, 1, 1, 1]) # Pad (left, right, top, bottom)


           # Sum the equivalent kernels and biases
           fused_kernel = kernel_rconv + kernel_branch + kernel_linear_padded
           fused_bias = bias_rconv + bias_branch + bias_linear


           # Replace the training branches with a single fused convolution
           # Ensure the new conv has the correct kernel size, stride, and padding (3x3, s, p=1)
           fused_conv = nn.Conv2d(self.in_channels, self.out_channels,
                                  kernel_size=self.rconv.kernel_size, stride=self.rconv.stride, padding=self.rconv.padding,
                                  groups=self.groups, bias=True).to(fused_kernel.device)
           fused_conv.weight.data = fused_kernel
           fused_conv.bias.data = fused_bias


           # Assign the fused conv to self.rconv
           self.rconv = fused_conv


           # Remove the training branches
           del self.rconv_bn
           del self.branch_conv
           del self.branch_bn
           del self.branch_linear
           del self.branch_linear_bn


           # Set deploy mode to True
           self.deploy = True




class YOLOLayer(nn.Module):
   """Detection layer (Head) processing features for a specific scale."""


   def __init__(self, anchors, args):
       super().__init__()
       self.args = args
       self.anchors = anchors
       self.num_anchors = len(self.anchors)
       self.num_classes = self.args.num_classes_detection
       # Use ignore_thres from args and ensure it's a float
       self.ignore_thres = float(self.args.ignore_thres) # Explicitly convert to float
       self.metrics = dict()
       self.img_dim = [self.args.input_width, self.args.input_height]
       self.grid_size = [0, 0]  # grid size (Height, Width) - Will be set by compute_grid_offsets
       self.stride = [0, 0]  # stride (Width, Height) - Will be set by compute_grid_offsets


       # Placeholder for input channels to this head. This will be set by the Decoder.
       self.in_channels_head = None # This will be an integer


       # --- YOLOv7 Head Structure (Simplified) ---
       # Standard YOLOv7 head has a few Conv/RepConv layers before the final prediction conv.
       # This structure is defined and initialized in the set_in_channels method
       # once the input channel count is known from the Decoder.


       # Layers are initialized in set_in_channels
       self.conv1 = None
       self.conv2 = None
       self.prediction = None




   # Method to set input channels and initialize head layers dynamically
   def set_in_channels(self, in_channels):
       """Sets the input channel count for the head and initializes its layers."""
       self.in_channels_head = in_channels
       # Initialize the head layers with the correct input channels
       # Simplified head sequence: RepConv -> RepConv -> Final Prediction Conv
       # Output channels of the RepConv layers are kept same as input for simplicity.
       # The final prediction conv outputs num_anchors * (5 + num_classes) channels.


       # Get device from args
       device = self.args.device


       self.conv1 = RepConv(self.in_channels_head, self.in_channels_head, k=3).to(device)
       self.conv2 = RepConv(self.in_channels_head, self.in_channels_head, k=3).to(device)
       self.prediction = nn.Conv2d(self.in_channels_head, self.num_anchors * (5 + self.num_classes), 1).to(device)




   def compute_grid_offsets(self, grid_size):
       """Computes grid coordinates and scaled anchors for decoding."""
       self.grid_size = grid_size
       # grid shape is [Height, Width]
       g = self.grid_size
       # Ensure img_dim is set before calculating stride
       if self.img_dim[0] == 0 or self.img_dim[1] == 0:
            # This should not happen if img_dim is correctly passed or set
            print("Warning: img_dim not set in YOLOLayer.compute_grid_offsets. Cannot compute stride.")
            # Fallback to a dummy stride, but this indicates an issue in the calling code.
            self.stride = [1, 1]
       else:
            self.stride[1] = self.img_dim[1] / self.grid_size[0]  # Height stride
            self.stride[0] = self.img_dim[0] / self.grid_size[1]  # Width stride




       # Calculate offsets for each grid using loops for onnx generation.
       grid_x = []
       for i in range(g[0]):
           grid_x.append(torch.arange(g[1]))
       self.grid_x = torch.stack(grid_x).view([1, 1, g[0], g[1]]).to(self.args.device)


       grid_y = []
       for i in range(g[1]):
           grid_y.append(torch.arange(g[0]))
       self.grid_y = torch.stack(grid_y).t().view([1, 1, g[0], g[1]]).to(self.args.device)


       # Scale anchors by stride for decoding
       # Ensure stride is not zero before division
       stride_w = self.stride[0] if self.stride[0] > 0 else 1
       stride_h = self.stride[1] if self.stride[1] > 0 else 1
       self.scaled_anchors = torch.Tensor([(a_w / stride_w, a_h / stride_h)
                                           for a_w, a_h in self.anchors]).to(self.args.device)


       self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
       self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))




   def _post_proccess_output_target(self, targets, raw_class_preds, raw_obj_preds, grid_shape):
       """
       Processes raw predictions and targets for loss calculation.
       Now also takes raw_class_preds (logits) and raw_obj_preds (logits) as input.
       """
       if targets is None:
           # If no targets are provided (e.g., during inference), return only the processed output
           return self.output, None, None
       else:
           # Call the actual build_targets and related functions from detection_utils
           # Note: The original build_targets in the previous immersive expected raw predictions.
           # The current YOLOLayer passes decoded/sigmoid predictions (self.pred_boxes, self.pred_conf, self.pred_cls).
           # This is a potential inconsistency that might need addressing in a later step
           # if your actual build_targets expects raw inputs.
           # For now, we assume the imported build_targets is compatible with these inputs.
           # Ensure ignore_thres is a float when passed to build_targets
           iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
               pred_boxes=self.pred_boxes, # Passing decoded boxes
               pred_cls=self.pred_cls,     # Passing sigmoid class probs
               target=targets,
               anchors=self.scaled_anchors,
               ignore_thres=self.ignore_thres, # This is already a float now
               args=self.args)


           # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
           target_dict = dict(x=tx[obj_mask], y=ty[obj_mask],
                              w=tw[obj_mask], h=th[obj_mask],
                              obj_conf=tconf[obj_mask], # Target objectness for positive samples (usually 1.0)
                              no_obj_conf=tconf[noobj_mask], # Target objectness for negative samples (usually 0.0)
                              cls=tcls[obj_mask], # Target class (indices or one-hot, based on build_targets)
                              iou_scores=iou_scores, tconf=tconf,
                              class_mask=class_mask, obj_mask=obj_mask, noobj_mask=noobj_mask, # Include masks
                              grid_shapes=[grid_shape]) # Add grid_shape to target_dict




           # --- Add raw predictions and masks to output_dict ---
           # Use the masks to select the relevant raw predictions
           num_samples = raw_class_preds.size(0)
           grid_size = [raw_class_preds.size(2), raw_class_preds.size(3)]
           # Flatten raw tensors and masks for easier indexing
           raw_class_preds_flat = raw_class_preds.view(num_samples, self.num_anchors, grid_size[0], grid_size[1], self.num_classes).permute(0, 1, 3, 4, 2).contiguous().view(-1, self.num_classes)
           raw_obj_preds_flat = raw_obj_preds.view(-1) # Flatten raw objectness logits
           obj_mask_flat = obj_mask.view(-1) # Flatten object mask
           noobj_mask_flat = noobj_mask.view(-1) # Flatten no-object mask




           output_dict = dict(x=self.x[obj_mask], y=self.y[obj_mask], # Decoded box coords for objects
                              w=self.w[obj_mask], h=self.h[obj_mask], # Decoded box dims for objects
                              obj_conf=self.pred_conf[obj_mask], # Sigmoid objectness for positive samples
                              no_obj_conf=self.pred_conf[noobj_mask], # Sigmoid objectness for negative samples
                              cls=self.pred_cls[obj_mask], # Sigmoid class probs for positive samples
                              raw_cls=raw_class_preds_flat[obj_mask_flat], # Raw class logits for positive samples
                              raw_obj_pos=raw_obj_preds_flat[obj_mask_flat], # Raw objectness logits for positive samples
                              raw_obj_neg=raw_obj_preds_flat[noobj_mask_flat], # Raw objectness logits for negative samples
                              pred_conf=self.pred_conf, # All sigmoid objectness scores
                              obj_mask=obj_mask, noobj_mask=noobj_mask, # Include masks (original shape)
                              grid_shapes=[grid_shape]) # Add grid_shape to output_dict


           return self.output, output_dict, target_dict


   def forward(self, x, targets=None, img_dim=None):
       # x: input feature map from the neck for this head (shape: B, C_in, H, W)


       # Ensure head layers are initialized with correct input channels
       if self.in_channels_head is None:
            # This indicates an issue in the Decoder's initialization flow
            raise RuntimeError("YOLOLayer's input channels not set. Call set_in_channels() after initialization.")


       # Pass input through the head layers to get raw predictions
       # Output shape: B, num_anchors * (5 + num_classes), H, W
       x = self.conv1(x)
       x = self.conv2(x)
       prediction = self.prediction(x) # Final prediction convolution




       # targets: full batch target tensor (num_targets, 6)
       # img_dim: image dimensions [W, H]


       if torch.is_tensor(img_dim):
           img_dim = img_dim.cpu().numpy()


       self.img_dim = img_dim
       num_samples = prediction.size(0) # Use prediction size after head layers
       grid_size = [prediction.size(2), prediction.size(3)]  # Height, Width


       # Reshape raw prediction tensor to (B, num_anchors, grid_h, grid_w, 5 + num_classes)
       num_of_output = 5 # 4 box coords + 1 objectness
       prediction_reshaped = (prediction.view(num_samples, self.num_anchors, self.num_classes + num_of_output,
                            grid_size[0], grid_size[1]).permute(0, 1, 3, 4, 2).contiguous())


       # --- Store raw class and objectness predictions before activation ---
       raw_obj_preds = prediction_reshaped[..., 4] # Shape: (B, num_anchors, grid_h, grid_w) - Raw objectness logits
       raw_class_preds = prediction_reshaped[..., 5:] # Shape: (B, num_anchors, grid_h, grid_w, num_classes) - Raw class logits




       # --- Apply YOLOv7 Activations and Decoding ---
       # Decode raw predictions into box coordinates, objectness, and class probabilities.
       # Use torch.sigmoid for x, y, and objectness confidence.
       # Use torch.sigmoid for class probabilities for multi-label support.
       # Use torch.exp for width and height scaling.


       # Apply sigmoid to x, y, and objectness confidence
       self.x = torch.sigmoid(prediction_reshaped[..., 0]) # Decoded x (relative to cell, range 0-1)
       self.y = torch.sigmoid(prediction_reshaped[..., 1]) # Decoded y (relative to cell, range 0-1)
       self.pred_conf = torch.sigmoid(raw_obj_preds) # Objectness confidence (Sigmoid) - Calculated from raw logits


       # Apply sigmoid to class probabilities for multi-label support
       self.pred_cls = torch.sigmoid(raw_class_preds) # Class probabilities (Sigmoid over class dimension)




       # Raw width and height predictions (used in decoding formula)
       self.w = prediction_reshaped[..., 2]
       self.h = prediction_reshaped[..., 3]




       # If grid size does not match current, recompute grid offsets and scaled anchors
       # Simplified grid size check comparing lists of integers
       if self.grid_size != grid_size:
           self.compute_grid_offsets(grid_size)


       # Add grid offsets and scale with anchors to get box coordinates in grid scale
       # bx = sigmoid(tx) + grid_x
       # by = sigmoid(ty) + grid_y
       # bw = exp(tw) * anchor_w
       # bh = exp(th) * anchor_h


       self.pred_boxes = torch.empty(prediction_reshaped[..., :4].shape).to(self.args.device) # Initialize tensor


       # Corrected decoding formula for x, y, w, h based on standard YOLOv7
       # x, y: (sigmoid(tx) * 2 – 0.5 + grid_x) * stride
       # w, h: (sigmoid(tw) * 2) ** 2 * anchor_w/h (apply to raw w, h before exp/sigmoid)


       # Note: The YOLOv7 official implementation applies (sigmoid(tx) * 2 - 0.5 + grid)
       # to the raw tx, ty *before* sigmoid.
       # The implementation here applies sigmoid first, then * 2 - 0.5.
       # Let's stick to the standard YOLOv7 formula for now, applying it to the raw predictions.


       # Re-calculating pred_boxes using standard YOLOv7 decoding on raw predictions
       # Need access to the raw prediction tensor 'prediction_reshaped' here.


       # Decoded x, y (relative to cell, range -0.5 to 1.5 after *2-0.5) + grid offset
       self.pred_boxes[..., 0] = (torch.sigmoid(prediction_reshaped[..., 0]) * 2. - 0.5 + self.grid_x)
       self.pred_boxes[..., 1] = (torch.sigmoid(prediction_reshaped[..., 1]) * 2. - 0.5 + self.grid_y)


       # Decoded w, h (scaled by anchors)
       # Apply (sigmoid(tw) * 2)^2 to raw w, h and then multiply by anchor_w/h
       # This requires sigmoid on the raw w, h predictions.
       # Let's use the standard YOLOv7 formula which applies (sigmoid(raw_wh) * 2)**2
       # Note: The original code used exp(raw_wh) * anchor_wh.
       # The review suggested (wh * 2) ** 2 * self.anchor_grid applied to some 'wh'.
       # The standard YOLOv7 applies (sigmoid(raw_wh) * 2)**2 to the raw predictions.


       # Corrected decoding for w, h based on standard YOLOv7:
       self.pred_boxes[..., 2] = (torch.sigmoid(prediction_reshaped[..., 2]) * 2).pow(2) * self.anchor_w
       self.pred_boxes[..., 3] = (torch.sigmoid(prediction_reshaped[..., 3]) * 2).pow(2) * self.anchor_h




       # Combine all predictions and scale to image dimensions
       # Output format: [x_center, y_center, width, height, objectness, class_probs]
       # Corrected stride scaling: scale x/w by width stride, y/h by height stride
       output_boxes_scaled = torch.empty_like(self.pred_boxes)
       output_boxes_scaled[..., 0] = self.pred_boxes[..., 0] * self.stride[0] # Scale x by width stride
       output_boxes_scaled[..., 1] = self.pred_boxes[..., 1] * self.stride[1] # Scale y by height stride
       output_boxes_scaled[..., 2] = self.pred_boxes[..., 2] * self.stride[0] # Scale w by width stride
       output_boxes_scaled[..., 3] = self.pred_boxes[..., 3] * self.stride[1] # Scale h by height stride




       # Concatenate scaled boxes, confidence, and class probabilities
       # Reshape to (batch_size, num_total_predictions, 4 + 1 + num_classes)
       self.output = torch.cat((output_boxes_scaled.view(num_samples, -1, 4), # Scaled box coords
                                self.pred_conf.view(num_samples, -1, 1), # Sigmoid confidence
                                self.pred_cls.view(num_samples, -1, self.num_classes)), -1) # Sigmoid class probs


       # Get current grid shape for _post_proccess_output_target
       current_grid_shape = [prediction.size(2), prediction.size(3)] # Height, Width


       # Call the post-processing/target building method, passing raw predictions and grid_shape
       return self._post_proccess_output_target(targets, raw_class_preds, raw_obj_preds, current_grid_shape)




######################################################################################################################
# #############################              Define YOLO Decoder             #########################################
######################################################################################################################




class YoloDecoder(nn.Module):
   # __init__ signature matches the user's existing code call
   def __init__(self, _out_filters, args):
       """
       Initializes the YOLO Decoder by extracting parameters from args and building the neck and heads.
       Dynamically determines input channels for neck based on _out_filters.


       Args:
           _out_filters (list): List of input channel dimensions from the backbone.
                                Assumed to be ordered by increasing stride.
                                This decoder is designed to use the feature maps corresponding
                                to strides 8, 16, and 32 (P3, P4, P5).
                                The list is expected to have at least enough elements
                                to identify P3, P4, and P5 based on standard strides.
           args (object): An object (like argparse.Namespace) containing configuration parameters.
                          Expected attributes: num_classes_detection, input_width, input_height,
                          anchors1, anchors2, anchors3, ignore_thres, device.
                          anchors1, anchors2, anchors3 are assumed to correspond to P5, P4, P3 scales respectively
                          based on the original code's final layer assignments.
       """
       super(YoloDecoder, self).__init__()
       self.args = args
       # Total output features per anchor: 4 box coords + 1 objectness + num_classes
       self.no = 5 + args.num_classes_detection # Store as self.no for _init_biases
       self.nc = args.num_classes_detection # Store num_classes as self.nc


       # Store all input channel dimensions from the backbone
       self.all_in_channels = _out_filters


       # Define the target strides for the detection heads (P3, P4, P5)
       self.target_strides = [8, 16, 32] # Order: P3, P4, P5


       # Determine the indices of the feature maps corresponding to the target strides
       # This assumes the backbone outputs are ordered by increasing stride.
       # We need to find the indices in _out_filters that correspond to strides 8, 16, 32.
       # A simple way is to assume the feature map at stride S has a spatial size of Image_Dim / S.
       # We can calculate the expected spatial size for each target stride and find the closest match
       # in the spatial dimensions of the backbone outputs.
       # However, a more robust way is if the backbone explicitly provides stride information
       # or if we rely on standard backbone architectures where we know which layer corresponds to which stride.


       # Assuming standard backbone output order by increasing stride and knowing the typical strides:
       # ResNet/Darknet typically have strides 4, 8, 16, 32, 64...
       # P3 (stride 8) is usually the second detection layer.
       # P4 (stride 16) is usually the third detection layer.
       # P5 (stride 32) is usually the fourth detection layer.


       # Let's assume _out_filters corresponds to stages with strides that are powers of 2,
       # starting from a certain minimum stride.
       # We need to find the indices for strides 8, 16, and 32.


       # A simplified approach: Assume the last 3 feature maps are P3, P4, P5 in order of increasing stride.
       # This is consistent with the previous version and common for FPN-like structures.
       if len(self.all_in_channels) < 3:
            raise ValueError(f"Expected _out_filters to contain channel dimensions for at least 3 feature maps (P3, P4, P5), but got {len(self.all_in_channels)}.")


       # Extract the channel dimensions for P3, P4, P5 from the end of the list
       p3_channels = self.all_in_channels[-3]
       p4_channels = self.all_in_channels[-2]
       p5_channels = self.all_in_channels[-1]


       # Store the indices of the P3, P4, P5 feature maps in the backbone_features list
       # This assumes backbone_features is ordered by increasing stride and matches _out_filters.
       # If _out_filters is [64, 256, 512, 1024, 2048], indices are 1, 2, 3.
       # If _out_filters is [256, 512, 1024], indices are 0, 1, 2.
       # We need a way to map the target strides [8, 16, 32] to indices in the backbone_features list.


       # Let's assume the backbone_features list passed to forward is ordered by increasing stride,
       # and we need to find the indices corresponding to strides 8, 16, and 32.
       # We can't reliably get stride from channel count alone.
       # A more robust solution would be to pass the strides along with _out_filters,
       # or assume standard backbone output structure.


       # Sticking to the assumption that backbone_features is ordered by increasing stride,
       # and we need the features corresponding to strides 8, 16, and 32.
       # We need to find the indices of these features in the backbone_features list.
       # This mapping needs to be known or determined.


       # Let's assume the indices for P3 (stride 8), P4 (stride 16), and P5 (stride 32)
       # are provided or can be inferred from the backbone structure.
       # For ResNet-50 with 5 stages, P3 is typically stage 2 (index 1), P4 stage 3 (index 2), P5 stage 4 (index 3).
       # For a backbone with only 3 output stages (P3, P4, P5), indices would be 0, 1, 2.


       # To make it flexible, let's require the user to provide the indices of the P3, P4, P5 features
       # in the backbone_features list, or assume a standard structure.
       # Let's assume a standard structure where the last 3 features are P3, P4, P5 in order of increasing stride.
       # This means the indices in the backbone_features list will be len(backbone_features)-3, len(backbone_features)-2, len(backbone_features)-1.


       # The channel dimensions for neck initialization are taken from the last 3 elements of _out_filters.
       # The indices for accessing features in the forward pass will be based on the length of the list passed to forward.


       self.num_scales = len(self.target_strides) # Number of detection heads


       # Extract anchors from args. Assuming args has anchors1, anchors2, anchors3 attributes
       # Note: These are the YOLOv3 anchors from your YAML.
       # Correcting the order to match P3, P4, P5 scales for consistency
       # Assuming anchors1 -> P5 (largest stride), anchors2 -> P4 (medium stride), anchors3 -> P3 (smallest stride)
       # based on the original code's final layer assignments.
       anchors_list = [list(args.anchors3), list(args.anchors2), list(args.anchors1)] # Order P3, P4, P5


       # Assuming same number of anchors per scale, based on the provided YAML structure
       self.num_anchors_per_scale = len(anchors_list[0])


       # Register anchors as buffers
       self.register_buffer('anchors', torch.tensor(anchors_list).float().view(self.num_scales, -1, 2)) # Shape: (num_scales, num_anchors_per_scale, 2)


       # Define strides for each head (P3, P4, P5) - assuming standard YOLO strides
       # These correspond to the downsampling factor from the input image size
       self.strides = [8, 16, 32] # Order: P3, P4, P5 - Corrected order


       # Calculate grid sizes based on input image dimensions and strides
       img_w, img_h = self.args.input_width, self.args.input_height
       self.grid_sizes = [
           [img_h // self.strides[0], img_w // self.strides[0]], # P3 grid size (H, W)
           [img_h // self.strides[1], img_w // self.strides[1]], # P4 grid size (H, W)
           [img_h // self.strides[2], img_w // self.strides[2]], # P5 grid size (H, W)
       ]




       # --- Construct YOLOv7 Neck (Incorporating SPPCSPC and E-ELAN) ---
       # Dynamically initialize neck modules based on the extracted P3, P4, P5 channels.


       # P5 path (Largest stride, smallest spatial size): SPPCSPC -> Conv -> Output for P5 head and branch for P4 path
       # Input channels: p5_channels
       # Output channels of SPPCSPC: Set to match the expected input of the next E-ELAN path (P4 path).
       sppcspc_branch_out_channels = 256 # Channels for the branch feeding into the P4 path


       # SPPCSPC module applied to the P5 feature map
       self.sppcspc = SPPCSPC(p5_channels, sppcspc_branch_out_channels)


       # A convolutional layer after SPPCSPC to produce the input for the P5 detection head.
       p5_head_out_channels = 512 # Desired output channels for P5 head input - tune if needed
       self.conv_p5_head_in = Conv(sppcspc_branch_out_channels, p5_head_out_channels, k=1)
       # Use the actual output channels for the head initialization
       p5_head_in_channels = self.conv_p5_head_in.conv.out_channels




       # Add Upsample layers for the P4 and P3 paths
       self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
       self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')




       # P4 path (Medium stride, medium spatial size): Upsample P5 branch -> Concatenate with P4_in -> E-ELAN -> Output for P4 head and branch for P3 path
       # Input channels to E-ELAN P4: sppcspc_branch_out_channels + p4_channels
       p4_head_out_channels = 512 # Desired output channels for P4 head input - tune if needed
       e_elan_p4_in = sppcspc_branch_out_channels + p4_channels
       # E-ELAN module for the P4 path.
       self.e_elan_p4 = E_ELAN(e_elan_p4_in, p4_head_out_channels, e=0.5)
       # Use the actual output channels for the head initialization
       p4_head_in_channels = self.e_elan_p4.cv10.conv.out_channels




       # A convolutional layer after E-ELAN P4 to produce the branch for the next level (P3 path).
       # Input channels: Output channels of E-ELAN P4 (p4_head_in_channels)
       e_elan_p4_branch_out_channels = 128 # Channels for the branch feeding into the P3 path
       self.conv_e_elan_p4_branch = Conv(p4_head_in_channels, e_elan_p4_branch_out_channels, k=1)




       # P3 path (Smallest stride, largest spatial size): Upsample P4 branch -> Concatenate with P3_in -> E-ELAN -> Output for P3 head
       # Input channels to E-ELAN P3: e_elan_p4_branch_out_channels + p3_channels
       p3_head_out_channels = 256 # Desired output channels for P3 head input - tune if needed
       e_elan_p3_in = e_elan_p4_branch_out_channels + p3_channels
       # E-ELAN module for the P3 path.
       self.e_elan_p3 = E_ELAN(e_elan_p3_in, p3_head_out_channels, e=0.5)
       # Use the actual output channels for the head initialization
       p3_head_in_channels = self.e_elan_p3.cv10.conv.out_channels




       # --- Define Final YOLOLayers (Detection Heads) ---
       # These layers process the output of the neck and produce the final predictions.
       # Each YOLOLayer is initialized with the dynamically determined input channel count.


       # P3 Detection Head (Smallest stride) - Receives output from e_elan_p3
       self.final_layer2 = YOLOLayer(list(self.args.anchors3), self.args)
       self.final_layer2.set_in_channels(p3_head_in_channels) # Set input channels for the P3 head
       self.final_layer2.compute_grid_offsets(self.grid_sizes[0]) # P3 uses grid_sizes[0]




       # P4 Detection Head (Medium stride) - Receives output from e_elan_p4
       self.final_layer1 = YOLOLayer(list(self.args.anchors2), self.args)
       self.final_layer1.set_in_channels(p4_head_in_channels) # Set input channels for the P4 head
       self.final_layer1.compute_grid_offsets(self.grid_sizes[1]) # P4 uses grid_sizes[1]




       # P5 Detection Head (Largest stride) - Receives output from conv_p5_head_in
       self.final_layer0 = YOLOLayer(list(self.args.anchors1), self.args)
       self.final_layer0.set_in_channels(p5_head_in_channels) # Set input channels for the P5 head
       self.final_layer0.compute_grid_offsets(self.grid_sizes[2]) # P5 uses grid_sizes[2]




       # Store final layers in a list for easier iteration in forward pass and target building
       # The order here should match the order of neck outputs feeding into them (P3, P4, P5)
       self.final_layers = nn.ModuleList([self.final_layer2, self.final_layer1, self.final_layer0]) # Order P3, P4, P5 heads


       # --- Bias Initialization for Detection Heads ---
       self._init_biases()




   def _init_biases(self):
       """Initializes the bias of the final prediction layers in each head."""
       prior = 0.01 # A common prior for objectness
       bias_value = -math.log((1 - prior) / prior) # log(0.01 / 0.99) approx -4.595


       yolo_layers = self.final_layers


       for i, head in enumerate(yolo_layers):
           conv = head.prediction
           b = conv.bias.data.view(head.num_anchors, -1)
           b[:, 4] = bias_value




   def switch_to_deploy(self):
       """
       Switches the model to deploy mode by fusing RepConv layers.
       Should be called after training and before inference.
       """
       for module in self.modules():
           if isinstance(module, RepConv) and not module.deploy:
               print(f"Fusing RepConv layer: {module}")
               module._fuse_kernel()
       print("RepConv layers fused for deployment.")




   def forward(self, backbone_features, img_dim, targets=None):
       """
       Forward pass of the YOLO Decoder.


       Args:
           backbone_features (list): List of feature maps from the backbone.
                                     Assumed to be ordered by increasing stride.
                                     We use the features corresponding to strides 8, 16, and 32 (P3, P4, P5).
           img_dim (list or torch.Tensor): Image dimensions [Width, Height].
           targets (torch.Tensor, optional): Ground truth targets. Defaults to None.


       Returns:
           dict: Dictionary containing detection outputs and target/output dictionaries for loss.
       """
       # backbone_features is a list of feature maps from the backbone, ordered by increasing stride.
       # We need to find the features corresponding to strides 8, 16, and 32.
       # This requires knowing the strides of the feature maps in backbone_features.
       # A common convention is that the feature maps are ordered by increasing stride.


       # Let's assume backbone_features is ordered by increasing stride and contains
       # features for at least strides 8, 16, and 32.
       # We need to find the indices of these features.


       # A simplified approach: Assume the feature maps corresponding to strides 8, 16, 32
       # are present in the backbone_features list, and we can find them by matching
       # their spatial dimensions to the expected grid sizes.


       # Expected spatial sizes for P3, P4, P5 based on input image size and strides
       expected_p3_spatial = (self.grid_sizes[0][0], self.grid_sizes[0][1]) # H, W
       expected_p4_spatial = (self.grid_sizes[1][0], self.grid_sizes[1][1]) # H, W
       expected_p5_spatial = (self.grid_sizes[2][0], self.grid_sizes[2][1]) # H, W


       # Find the indices of the feature maps in backbone_features that match the expected spatial sizes
       p3_index, p4_index, p5_index = -1, -1, -1


       for i, feature_map in enumerate(backbone_features):
           spatial_size = (feature_map.shape[2], feature_map.shape[3]) # H, W
           if spatial_size == expected_p3_spatial:
               p3_index = i
           elif spatial_size == expected_p4_spatial:
               p4_index = i
           elif spatial_size == expected_p5_spatial:
               p5_index = i


       # Check if all required feature maps were found
       if -1 in [p3_index, p4_index, p5_index]:
           raise ValueError(f"Could not find feature maps for all target strides (8, 16, 32) in backbone_features. "
                            f"Expected spatial sizes: P3={expected_p3_spatial}, P4={expected_p4_spatial}, P5={expected_p5_spatial}. "
                            f"Found spatial sizes: {[f.shape[2:] for f in backbone_features]}.")


       # Access the P3, P4, and P5 feature maps using the found indices
       p3_in = backbone_features[p3_index]
       p4_in = backbone_features[p4_index]
       p5_in = backbone_features[p5_index]




       outputs = dict()
       yolo_outputs = list() # List to collect flattened predictions from each head
       yolo_output_dict_list = list() # List to collect output dictionaries from each head
       yolo_target_dict_list = list() # List to collect target dictionaries from each head
       grid_shapes_list = list() # List to store grid shapes for each head




       # --- Forward pass through the YOLOv7 Neck (SPPCSPC and E-ELAN) ---
       # Processes the backbone features to create feature maps for the detection heads.


       # P5 path (Largest stride): SPPCSPC -> Conv -> Output for P5 head and branch for P4 path
       sppcspc_out = self.sppcspc(p5_in) # Process P5 feature map with SPPCSPC
       out0_head_in = self.conv_p5_head_in(sppcspc_out) # Final conv for P5 head input
       out0_branch = sppcspc_out # Branch output from SPPCSPC for the P4 path




       # P4 path (Medium stride): Upsample P5 branch -> Concatenate with P4_in -> E-ELAN -> Output for P4 head and branch for P3 path
       x1_in = self.embedding1_upsample(out0_branch) # Upsample the branch from the P5 path
       x1_in = torch.cat([x1_in, p4_in], 1) # Concatenate with the P4 feature map from backbone
       e_elan_p4_out = self.e_elan_p4(x1_in) # Pass through E-ELAN for P4 path
       out1_head_in = e_elan_p4_out # Output of E-ELAN P4 for the P4 detection head
       out1_branch = self.conv_e_elan_p4_branch(e_elan_p4_out) # Branch output from E-ELAN P4 for the P3 path




       # P3 path (Smallest stride): Upsample P4 branch -> Concatenate with P3_in -> E-ELAN -> Output for P3 head
       x2_in = self.embedding2_upsample(out1_branch) # Upsample the branch from the P4 path
       x2_in = torch.cat([x2_in, p3_in], 1) # Concatenate with the P3 feature map from backbone
       out2_head_in = self.e_elan_p3(x2_in) # Pass through E-ELAN for P3 path
       # out2_head_in is the output for the P3 detection head




       # Collect outputs from the neck for each head (P3, P4, P5)
       # The order should match the final layers (P3, P4, P5)
       neck_outputs_for_heads = [out2_head_in, out1_head_in, out0_head_in] # Order P3, P4, P5




       # --- Apply Final YOLOLayers (Detection Heads) and Collect Outputs ---
       # Iterate through the detection heads (YOLOLayers) and process the neck outputs.
       # The order of final layers corresponds to P3, P4, P5 scales.
       final_layers = self.final_layers




       for i in range(self.num_scales):
           # Call the YOLOLayer for the current scale's neck output
           # The YOLOLayer's forward returns (flattened_output, output_dict_batch, target_dict_batch)
           flattened_output, output_dict_batch, target_dict_batch = final_layers[i](neck_outputs_for_heads[i], targets, img_dim)


           # Collect the outputs from each head
           yolo_outputs.append(flattened_output)
           yolo_output_dict_list.append(output_dict_batch)
           yolo_target_dict_list.append(target_dict_batch)


           # Get grid shape for this head from the neck output *before* the YOLOLayer
           # Use the shape of the input to the YOLOLayer
           grid_h, grid_w = neck_outputs_for_heads[i].shape[2:]
           grid_shapes_list.append((grid_h, grid_w))




       # Concatenate flattened predictions across all scales for the main output
       # This output is typically used for NMS and final detection results.
       # Only concatenate if there are outputs (e.g., not during ONNX export with targets=None)
       if len(yolo_outputs) > 0 and yolo_outputs[0] is not None:
            yolo_outputs_concatenated = torch.cat(yolo_outputs, 1).detach().cpu()
       else:
            yolo_outputs_concatenated = None # Handle case with no outputs




       # --- Add Grid Shapes to Output and Target Dictionaries ---
       # Add the list of grid shapes for all heads to each head's output dictionary.
       # This is needed by the loss function or metrics calculation.
       for i in range(len(yolo_output_dict_list)):
           if yolo_output_dict_list[i] is not None:
               yolo_output_dict_list[i]['all_grid_shapes'] = grid_shapes_list # Add to output dict
       for i in range(len(yolo_target_dict_list)):
           if yolo_target_dict_list[i] is not None:
               yolo_target_dict_list[i]['all_grid_shapes'] = grid_shapes_list # Add to target dict




       # If exporting to ONNX, return only the concatenated predictions
       if torch.onnx.is_in_onnx_export():
           return yolo_outputs_concatenated
       else:
           # Return a dictionary containing the concatenated outputs and the per-head dictionaries
           outputs["yolo_outputs"] = yolo_outputs_concatenated
           outputs["yolo_output_dicts"] = yolo_output_dict_list if targets is not None else None
           outputs["yolo_target_dicts"] = yolo_target_dict_list if targets is not None else None
           return outputs

