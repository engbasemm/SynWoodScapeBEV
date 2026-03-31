"""
Multi-task training for Compression, Semantic Segmentation, 2D Detection,
and BEV Stitching for OmniDet.

# author: Eslam Mohamed AbdelRahman <eslam.mohamed-abdelrahman@valeo.com>
# author: Hazem Rashed <hazem.rashed.@valeo.com>
# author: Varun Ravi Kumar <rvarun7777@gmail.com>
# BEV integration by: Basem Barakat & Gemini

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from colorama import Fore, Style
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from losses.detection_loss import ObjectDetectionLoss
from losses.mtl_losses_regulated_attn import UncertaintyLoss
from models.bev_models import MultiScaleViewFusionModule, BEVNeck, BEVDecoder, CombinedStitchingLoss
from train_semantic_compression import CompressionSemanticModel
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from train_semantic import SemanticModel
from train_utils.detection_utils import non_max_suppression, get_batch_statistics, ap_per_class, xywh2xyxy
from pytorch_msssim import ssim

torch.autograd.set_detect_anomaly(True)

class CompressionSemanticDetectionBevModelBase(CompressionSemanticModel):
    def __init__(self, args):
        super().__init__(args)

        # --- OVERRIDE DATALOADER ---
        print("=> Overriding dataloader for BEV training...")
        train_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                               path_file=args.train_file,
                                               is_train=True,
                                               config=args)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=train_dataset.collate_fn)

        val_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                             path_file=args.val_file,
                                             is_train=False,
                                             config=args)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False, # Should be False for validation
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True,
                                     collate_fn=val_dataset.collate_fn)
        print(f"=> Dataloader overridden. Training examples: {len(train_dataset)}, Validation examples: {len(val_dataset)}")

        # --- SVS Modules for BEV Stitching ---
        num_cameras = len(self.train_loader.dataset.all_cam_sides)
        # Use the last two feature scales from the encoder
        encoder_channels = self.encoder_channels[-2:]
        fusion_out_channels = 256 # A common dimension for fusion

        self.models["view_transformer"] = MultiScaleViewFusionModule(
            in_channels_list=encoder_channels,
            out_channels=fusion_out_channels,
            num_views=num_cameras
        ).to(self.device)

        # The decoder input will be the concatenated size from the fusion module
        decoder_in_channels = fusion_out_channels * len(encoder_channels)
        self.models["bev_neck"] = BEVNeck(in_channels=decoder_in_channels).to(self.device)
        self.models["stitching_decoder"] = BEVDecoder(in_channels=decoder_in_channels, out_channels=3).to(self.device)

        # Update parameters to train
        self.parameters_to_train += list(self.models["view_transformer"].parameters())
        self.parameters_to_train += list(self.models["bev_neck"].parameters())
        self.parameters_to_train += list(self.models["stitching_decoder"].parameters())

        self.stitching_criterion = CombinedStitchingLoss(alpha=0.85)
        self.best_stitch_loss = float('inf')

        # --- Update MTL Loss ---
        self.mtl_loss = UncertaintyLoss(tasks=self.args.train, enable_compression=self.args.enable_compression).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())

    def compression_semantic_detection_loss_predictions(self, inputs, features):
        losses = dict()
        outputs = dict()
        num_cameras = len(self.train_loader.dataset.all_cam_sides)
        all_cam_images = torch.cat([inputs[("color_aug", 0, 0, cam_side)] for cam_side in self.train_loader.dataset.all_cam_sides], dim=0)

        # --- Prepare features for downstream tasks (Detection/Semantic) ---
        features_for_downstream = list(features)
        if self.enable_joint_contect_mic:
            deepest_features = features[-1]
            fused_features = self.models["feature_fusion"](deepest_features, num_cameras)
            features_for_downstream[-1] = fused_features

        semantic_outputs = self.models["semantic"](features_for_downstream)
        if self.args.has_semantic_annotations:
            outputs.update(semantic_outputs)

        outputs, detection_losses = self.predict_detection(inputs, outputs, features=features_for_downstream)
        losses.update(detection_losses)

        # Compression
        if self.enable_compression:
            features_for_compression = features[-2]
            projected_features = self.models["compression_projection"](features_for_compression)
            compression_output_dict = self.models["decoder"](projected_features, num_camera=num_cameras)
            outputs[("compression", 0)] = compression_output_dict["x_hat"]
            losses["compression_loss"] = self.compression_criterion(compression_output_dict, all_cam_images)["compression_loss"]

        # Semantic Loss
        if self.args.has_semantic_annotations:
            all_semantic_labels = inputs.get(("semantic_labels_all_views", 0, 0))
            if all_semantic_labels is not None and all_semantic_labels.numel() > 0:
                if all_semantic_labels.dim() == 4 and all_semantic_labels.shape[1] == 1:
                    all_semantic_labels = all_semantic_labels.squeeze(1)
                losses["semantic_loss"] = self.semantic_criterion(outputs["semantic", 0], all_semantic_labels)

        return outputs, losses

    def compression_semantic_detection_bev_train(self):
        """Trainer function for compression, semantic, detection and BEV stitching prediction"""
        for self.epoch in range(self.args.epochs):
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()
            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()

                self.inputs_to_device(inputs)
                self.optimizer.zero_grad()

                # --- 1. Per-View 2D Tasks & Feature Extraction ---
                all_cam_images = torch.cat([inputs[("color_aug", 0, 0, cam_side)] for cam_side in self.train_loader.dataset.all_cam_sides], dim=0)
                features = self.models["encoder"](all_cam_images)

                outputs, losses = self.compression_semantic_detection_loss_predictions(inputs, features)

                # --- 2. BEV Pipeline ---
                num_cameras = len(self.train_loader.dataset.all_cam_sides)
                b, _, _, _ = inputs[("color_aug", 0, 0, 'FV')].shape

                # Reshape features from [B*V, C, H, W] to [V, B, C, H, W] then list of scales
                multi_view_features_list = [
                    [f.view(num_cameras, b, *f.shape[1:])[:,i] for f in features]
                    for i in range(b)
                ]
                # Flatten the list
                multi_view_features_list = [item for sublist in multi_view_features_list for item in sublist]


                num_scales = len(multi_view_features_list[0])
                multi_scale_features = [
                    torch.stack([multi_view_features_list[v][s] for v in range(num_cameras)], dim=1)
                    for s in range(num_scales)
                ]
                aggregated_bev_features = self.models["view_transformer"](multi_scale_features)
                refined_bev_features = self.models["bev_neck"](aggregated_bev_features)
                predicted_bev = self.models["stitching_decoder"](refined_bev_features)
                outputs[("bev_stitch", 0)] = predicted_bev

                # --- 3. Aggregate Losses & Backpropagate ---
                gt_bev = inputs[("bev_gt", 0, 0)]
                gt_bev_resized = F.interpolate(gt_bev, size=predicted_bev.shape[2:], mode='bilinear', align_corners=False)
                losses["bev_loss"] = self.stitching_criterion(predicted_bev, gt_bev_resized)

                losses["mtl_loss"] = self.mtl_loss(losses)

                if losses["mtl_loss"] is not None:
                    losses["mtl_loss"].mean().backward()
                    self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data, data_loading_time, gpu_time)
                    self.stitching_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    self.save_best_semantic_weights()
                    self.save_best_stitching_weights()
                    if self.enable_compression:
                        self.save_best_compression_weights()

                self.step += 1
                before_op_time = time.time()
            self.lr_scheduler.step()

    @torch.no_grad()
    def stitching_val(self):
        """Validate the BEV stitching model"""
        self.set_eval()
        total_stitch_loss, num_batches = 0, 0
        vis_inputs, vis_outputs = None, None

        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            multi_view_features_list = []
            for cam_side in self.val_loader.dataset.all_cam_sides:
                input_tensor = inputs[("color", 0, 0, cam_side)]
                all_features_per_cam = self.models["encoder"](input_tensor)
                multi_view_features_list.append(all_features_per_cam[-2:])

            num_scales = len(multi_view_features_list[0])
            multi_scale_features = [
                torch.stack([multi_view_features_list[v][s] for v in range(len(self.val_loader.dataset.all_cam_sides))], dim=1)
                for s in range(num_scales)
            ]
            aggregated_bev_features = self.models["view_transformer"](multi_scale_features)
            refined_bev_features = self.models["bev_neck"](aggregated_bev_features)
            predicted_bev = self.models["stitching_decoder"](refined_bev_features)

            gt_bev = inputs[("bev_gt", 0, 0)]
            gt_bev_resized = F.interpolate(gt_bev, size=predicted_bev.shape[2:], mode='bilinear', align_corners=False)
            stitch_loss = self.stitching_criterion(predicted_bev, gt_bev_resized)
            total_stitch_loss += stitch_loss.item()
            num_batches += 1

            if vis_inputs is None:
                vis_inputs = {("bev_gt", 0, 0): gt_bev}
                vis_outputs = {("bev_stitch", 0): predicted_bev}

        avg_loss = total_stitch_loss / num_batches if num_batches > 0 else 0
        losses = {"bev_loss": torch.tensor(avg_loss)}
        self.stitching_statistics("val", vis_inputs, vis_outputs, losses)
        self.set_train()
        return avg_loss

    def stitching_statistics(self, mode, inputs, outputs, losses):
        """Log BEV stitching stats to tensorboard"""
        writer = self.writers[mode]
        if "bev_loss" in losses:
            writer.add_scalar("bev_loss", losses["bev_loss"].mean(), self.step)

        if inputs and outputs and ("bev_gt", 0, 0) in inputs and ("bev_stitch", 0) in outputs:
            pred_bev = outputs[("bev_stitch", 0)]
            gt_bev = inputs[("bev_gt", 0, 0)]
            gt_bev_grid = make_grid(gt_bev[:4].cpu(), nrow=4, normalize=True)
            pred_bev_grid = make_grid(pred_bev[:4].cpu(), nrow=4, normalize=True)
            writer.add_image(f"bev/{mode}/ground_truth", gt_bev_grid, self.step)
            writer.add_image(f"bev/{mode}/prediction", pred_bev_grid, self.step)

    def save_best_stitching_weights(self):
        """Validate BEV stitching and save model on improvements"""
        stitch_loss = self.stitching_val()
        print(f"{Fore.GREEN}Stitching val loss: {stitch_loss:.4f}{Style.RESET_ALL}")
        if stitch_loss < self.best_stitch_loss:
            self.best_stitch_loss = stitch_loss
            print(f"{Fore.GREEN}=> Saving model with new best stitching loss: {self.best_stitch_loss:.4f}{Style.RESET_ALL}")
            self.save_model()

class CompressionSemanticDetectionBevModel(CompressionSemanticDetectionBevModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.configure_optimizers()
        self.pre_init()
