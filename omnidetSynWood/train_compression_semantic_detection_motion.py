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
from losses.mtl_losses import UncertaintyLoss
from models.detection_decoderORG import YoloDecoder
from train_detection import DetectionModelBase
from train_semantic_compression import CompressionSemanticModel
from train_compression_semantic_detection import CompressionSemanticDetectionModelBase
from train_motion import MotionModel, MotionInit
from train_semantic import SemanticModel
from models.motion_decoder import MotionDecoder
from train_utils.detection_utils import log_metrics


class CompressionSemanticDetectionMotionModel(CompressionSemanticDetectionModelBase , MotionInit):
    def __init__(self, args):
        super().__init__(args)

        self.models["motion"] = MotionDecoder(self.encoder_channels,
                                              n_classes=2,
                                              siamese_net=self.args.siamese_net).to(self.device)

        self.parameters_to_train += list(self.models["motion"].parameters())

        if args.use_multiple_gpu:
            self.models["motion"] = torch.nn.DataParallel(self.models["motion"])

    def compression_semantic_detection_motion_train(self):
        """Trainer function for compression, semantic and detection prediction"""
        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                self.inputs_to_device(inputs)

                # -- DISTANCE, SEMANTIC SEGMENTATION AND OBJECT DETECTION MODEL PREDICTIONS AND LOSS CALCULATIONS --
                outputs, losses = self.compression_semantic_detection_motion_loss_predictions(inputs)

                # -- MTL LOSS --
                losses["mtl_loss"] = self.mtl_loss(losses)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["mtl_loss"].mean().backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    if self.enable_compression == True:
                        self.compression_statistics("train", inputs, outputs[("compression",0)], losses)
                    SemanticModel.semantic_statistics(self, "train", inputs, outputs, losses)
                    DetectionModelBase.detection_statistics(self, "train")
                    MotionModel.motion_statistics(self, "train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_semantic_weights()
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.save_best_detection_weights()
                    if self.enable_compression == True :
                        # -- SAVE COMPRESSION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                        self.save_best_compression_weights()
                    # -- SAVE MOTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_motion_weights()

                    DetectionModelBase.detection_statistics(self, "val")

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0 and False:
                self.save_model()

        print("Training complete!")

    def compression_semantic_detection_motion_loss_predictions(self, inputs):
        features, outputs, losses = self.compression_semantic_detection_loss_predictions(inputs)

        # -- MOTION SEGMENTATION --
        motion_predictions = self.predict_motion_seg(inputs, features=features,  mode='train')
        outputs.update(motion_predictions)

        # -- MOTION SEGMENTATION LOSS --
        losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])
        if self.args.use_multiple_gpu:
            losses["motion_loss"] = losses["motion_loss"].unsqueeze(0)

        return outputs, losses

    def predict_motion_seg(self, inputs, features=None, mode='val'):
        outputs = dict()
        if self.args.siamese_net:
            previous_frames = self.models["encoder"](inputs["color_aug", -1, 0])
            current_frames = features if mode != 'val' else self.models["encoder"](inputs["color_aug", 0, 0])
            features = [torch.cat([i, j], dim=1) for i, j in zip(previous_frames, current_frames)]
            outputs.update(self.models["motion"](features))
        else:
            features = self.models["encoder"](torch.cat([inputs["color_aug", -1, 0], inputs["color_aug", 0, 0]], 1))
            outputs.update(self.models["motion"](features))
        return outputs

    @torch.no_grad()
    def motion_val(self):
        """Validate the motion model"""
        self.set_eval()

        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)
            outputs = self.predict_motion_seg(inputs, features=None, mode='val')
            losses["motion_loss"] = self.motion_criterion(outputs["motion", 0], inputs["motion_labels", 0, 0])
            if self.args.use_multiple_gpu:
                losses["motion_loss"] = losses["motion_loss"].unsqueeze(0)
            _, predictions = torch.max(outputs["motion", 0].data, 1)
            self.motion_metric.add(predictions, inputs["motion_labels", 0, 0])

        outputs["class_iou"], outputs["mean_iou"] = self.motion_metric.value()

        # Compute stats for the tensorboard
        MotionModel.motion_statistics(self, "val", inputs, outputs, losses)
        self.motion_metric.reset()
        del inputs, losses
        self.set_train()

        return outputs

    def save_best_motion_weights(self):
        # Motion Seg. validation on each step and save model on improvements.
        motion_val_metrics = self.motion_val()
        print(
            f"{Fore.MAGENTA}epoch {self.epoch:>3} | Motion IoU: {motion_val_metrics['mean_iou']:.3f}{Style.RESET_ALL}")
        if motion_val_metrics["mean_iou"] >= self.best_motion_iou:
            print(f"{Fore.MAGENTA}=> Saving motion model weights with mean_iou of {motion_val_metrics['mean_iou']:.3f} "
                  f"at step {self.step} on {self.epoch} epoch.{Style.RESET_ALL}")
            self.best_motion_iou = motion_val_metrics["mean_iou"]
            if self.epoch > 50:  # Weights are quite large! Sometimes, life is a compromise.
                self.save_model()

    def predict_detection(self, inputs, outputs, features=None):
        losses = dict()

        if self.args.pose_model_type == "shared":
            # Use semantic features in MTL instead of encoder features
            detection_output = self.models["detection"](features[0],
                                                        [self.args.input_width, self.args.input_height],
                                                        inputs[("detection_labels", 0)])
        else:
            # Use semantic features in MTL instead of encoder features
            detection_output = self.models["detection"](features,
                                                        [self.args.input_width, self.args.input_height],
                                                        inputs[("detection_labels", 0)])


        outputs[("detection", 0)] = detection_output["yolo_outputs"]

        # -- DETECTION LOSSES --
        detection_losses = self.detection_criterion(detection_output["yolo_output_dicts"],
                                                    detection_output["yolo_target_dicts"])

        losses.update(dict(detection_loss=detection_losses['detection_loss']))

        # -- DETECTION LOGS --
        self.logs.update(log_metrics(detection_output["yolo_output_dicts"],
                                     detection_output["yolo_target_dicts"], detection_losses))
        return outputs, losses

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

