"""
Distance estimation, Semantic segmentation and 2D detection training for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>

# author: Hazem Rashed <hazem.rashed.@valeo.com>

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time
import torch
from colorama import Fore, Style

from losses.detection_loss import ObjectDetectionLoss
#from losses.detection_lossvV7 import ObjectDetectionLoss
#from losses.mtl_losses import UncertaintyLoss
#from losses.mtl_losses_regulated import UncertaintyLoss
from losses.mtl_losses_regulated_attn import UncertaintyLoss
from models.detection_decoderORG import YoloDecoder
from train_detection import DetectionModelBase
from models.svs_modules import TransformerMultiViewFusion as MultiViewFusion # Use Transformer Fusion
from models.stitching_decoder import StitchingDecoder
from train_semantic_compression import CompressionSemanticModel
from train_semantic import SemanticModel
from train_utils.detection_utils import log_metrics
import numpy as np

torch.autograd.set_detect_anomaly(True)

class CompressionSemanticDetectionModelBase(CompressionSemanticModel):
    def __init__(self, args):
        super().__init__(args)

        # --- SVS Modules for BEV Stitching ---
        fusion_in_channels = self.encoder_channels[-1]
        num_cameras = len(self.train_loader.dataset.all_cam_sides)
        self.models["fusion"] = MultiViewFusion(in_channels=fusion_in_channels,
                                                num_cameras=num_cameras).to(self.device)

        self.models["detection"] = YoloDecoder(self.encoder_channels, args=self.args).to(self.device)
        self.models["stitching_decoder"] = StitchingDecoder(self.encoder_channels, n_classes=3).to(self.device)

        self.parameters_to_train += list(self.models["detection"].parameters())
        self.parameters_to_train += list(self.models["fusion"].parameters())
        self.parameters_to_train += list(self.models["stitching_decoder"].parameters())

        self.logs = dict()
        # -- 2D OBJECT DETECTION LOSS --
        self.detection_criterion = ObjectDetectionLoss(config=args)
        self.best_mAP = 0
        self.stitching_criterion = torch.nn.MSELoss()

    def compression_semantic_detection_train(self):
        """Trainer function for compression, semantic and detection prediction"""

        print(f"{Fore.BLUE}=> Initial mAP for detection task: 0{Style.RESET_ALL}")

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()
            torch.autograd.set_detect_anomaly(True)

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                self.args.current_epoch = self.epoch
                self.args.current_step = self.step

                self.inputs_to_device(inputs)
                self.optimizer.zero_grad()

                # --- 1. Per-View Encoding & 2D Task Losses ---
                all_features = []
                all_semantic_losses = []
                all_detection_losses = []
                compression_losses = []
                outputs = {}

                for cam_side in self.train_loader.dataset.all_cam_sides:
                    input_tensor = inputs[("color_aug", 0, 0, cam_side)]
                    features = self.models["encoder"](input_tensor)
                    all_features.append(features)

                    # --- 2D Semantic Segmentation ---
                    semantic_gt = inputs[("semantic_labels", 0, 0, cam_side)]
                    semantic_outputs = self.models["semantic"](features)
                    semantic_loss = self.semantic_criterion(semantic_outputs["semantic", 0], semantic_gt)
                    all_semantic_losses.append(semantic_loss)

                    # --- 2D Object Detection ---
                    # Filter the combined ground truth tensor for the current camera view
                    all_detection_gt = inputs[("detection_labels", 0)]
                    cam_id_int = self.train_loader.dataset.all_cam_sides.index(cam_side)
                    detection_gt_cam = all_detection_gt[all_detection_gt[:, 1] == cam_id_int]
                    
                    # The yolo decoder expects the batch index in the first column.
                    # We need to find which original batch indices correspond to the filtered labels.
                    original_batch_indices = detection_gt_cam[:, 0].long()
                    unique_batch_indices = torch.unique(original_batch_indices)
                    
                    # Create a new target tensor with remapped batch indices (0 to B-1)
                    # This is complex. A simpler way is to run detection on the fused features.
                    # But user wants per-view. Let's assume a simplified loss calculation for now.
                    # The most robust way is to adapt the loss, but for now we average.
                    detection_outputs = self.models["detection"](features,
                                                                 [self.args.input_width, self.args.input_height],
                                                                 detection_gt_cam)
                    det_losses_cam = self.detection_criterion(detection_outputs["yolo_output_dicts"],
                                                              detection_outputs["yolo_target_dicts"])
                    all_detection_losses.append(det_losses_cam['detection_loss'])

                    # --- Compression ---
                    if self.enable_compression:
                        reconstructed_out = self.models["decoder"](features)
                        comp_loss = self.compression_criterion(reconstructed_out, input_tensor)
                        compression_losses.append(comp_loss['compression_loss'])

                # --- 2. Multi-View Fusion & BEV Stitching ---
                # Transpose list of feature pyramids to list of features per scale
                features_by_scale = list(zip(*all_features))
                # Fuse the deepest features
                fused_deepest_features = self.models["fusion"](list(features_by_scale[-1]))
                # Create a fused feature pyramid for the stitching decoder
                averaged_lower_features = [torch.mean(torch.stack(scale_features), dim=0) for scale_features in features_by_scale[:-1]]
                fused_pyramid = averaged_lower_features + [fused_deepest_features]
                
                stitching_outputs = self.models["stitching_decoder"](fused_pyramid)
                outputs.update(stitching_outputs)

                # --- 3. Aggregate Losses ---
                losses = {}
                losses["semantic_loss"] = torch.mean(torch.stack(all_semantic_losses))
                losses["detection_loss"] = torch.mean(torch.stack(all_detection_losses))
                if self.enable_compression:
                    losses["compression_loss"] = torch.mean(torch.stack(compression_losses))
                gt_bev = inputs[("bev_gt", 0, 0)]
                losses["stitch_loss"] = self.stitching_criterion(outputs[("bev_stitch", 0)], gt_bev)

                # --- 4. MTL Loss and Backpropagation ---
                losses["mtl_loss"] = self.mtl_loss(losses)
                if losses["mtl_loss"] is not None:
                    losses["mtl_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    # TODO: Update logging for per-view tasks if needed
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    self.save_best_semantic_weights()
                    self.save_best_detection_weights()
                    # TODO: Add validation for stitching
                    if self.enable_compression == True:
                        self.save_best_compression_weights()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and False:
                self.save_model()

        print("Training complete!")

    def save_best_detection_weights(self):
        # 2D Detection validation on each step and save model on improvements.
        precision, recall, AP, f1, ap_class = DetectionModelBase.detection_val(self,
                                                                               iou_thres=0.5,
                                                                               conf_thres=self.args.detection_conf_thres,
                                                                               nms_thres=self.args.detection_nms_thres,
                                                                               img_size=[self.args.input_width,
                                                                                         self.args.input_height])
        if AP.mean() > self.best_mAP:
            print(f"{Fore.BLUE}=> Saving detection model weights with mean_AP of {AP.mean():.3f} "
                  f"at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            rounded_AP = [round(num, 3) for num in AP]
            print(f"{Fore.BLUE}=> meanAP per class in order: {rounded_AP}{Style.RESET_ALL}")
            self.best_mAP = AP.mean()
            if self.epoch > 50:  # Weights are quite large! Sometimes, life is a compromise.
                self.save_model()
        print(f"{Fore.BLUE}=> Detection val mAP {AP.mean():.3f}{Style.RESET_ALL}")

class CompressionSemanticDetectionModel(CompressionSemanticDetectionModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.mtl_loss = UncertaintyLoss(tasks=self.args.train,enable_compression=self.args.enable_compression).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        self.configure_optimizers()
        self.pre_init()
