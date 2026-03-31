"""
Distance estimation and Semantic segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import torch
from colorama import Fore, Style

from losses.mtl_losses import UncertaintyLoss
from models.semantic_decoder import SemanticDecoder
from models.compression_decoder import UnetDecoder
from train_distance import DistanceModelBase
from train_depth import DepthModelBase
from train_compression import CompressionModel, CompressionInit

class DistanceCompressionModelBase(DepthModelBase, CompressionInit):
#class DistanceCompressionModelBase(DistanceModelBase, CompressionInit):
    def __init__(self, args):
        super().__init__(args)

        self.models["compression"] = UnetDecoder(2).to(self.device)
        self.parameters_to_train += list(self.models["compression"].parameters())

        if args.use_multiple_gpu:
            self.models["compression"] = torch.nn.DataParallel(self.models["compression"])

    def distance_compression_train(self):
        """Trainer function for distance and compression prediction"""

        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            # MTL loss initialization
            for batch_idx, inputs in enumerate(self.train_loader):
                data_loading_time += (time.time() - before_op_time)
                before_op_time = time.time()
                self.inputs_to_device(inputs)

                # -- DISTANCE AND SEMANTIC SEGMENTATION MODEL PREDICTIONS AND LOSS CALCULATIONS --
                _, outputs, losses = self.distance_compression_loss_predictions(inputs)

                # -- MTL LOSS --
                losses["mtl_loss"] = self.mtl_loss(losses)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["mtl_loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mtl_loss"].mean().cpu().data,
                                  data_loading_time, gpu_time)
                    self.distance_statistics("train", inputs, outputs, losses)
                    CompressionModel.compression_statistics(self, "train", inputs, outputs["compression"], losses)
                    data_loading_time = 0
                    gpu_time = 0

                if (self.step % self.args.val_frequency == 0) and (self.enable_compression == True):
                    # -- SAVE SEMANTIC MODEL WITH BEST WEIGHTS BASED ON VALIDATION IoU --
                    self.save_best_compression_weights()

                self.step += 1
                before_op_time = time.time()

            self.lr_scheduler.step()

            if (self.epoch + 1) % self.args.save_frequency == 0:
                self.save_model()

        print("Training complete!")

    def distance_compression_loss_predictions(self, inputs):
        losses = dict()
        # -- SEMANTIC SEGMENTATION --
        outputs, features = self.predict_compression_seg(inputs)
        # -- SEMANTIC SEGMENTATION LOSS --
        losses["compression_loss"] = self.criterion(outputs["compression"],
                                                    inputs["color_aug", 0, 0])
        if self.args.use_multiple_gpu:
            losses["compression_loss"] = losses["compression_loss"].unsqueeze(0)

        # -- DISTANCE ESTIMATION --
        distance_outputs, features = self.predict_distances(inputs, features=features)
        outputs.update(distance_outputs)
        # -- POSE ESTIMATION --
        outputs.update(self.predict_poses(inputs, features))
        # -- PHOTOMETRIC LOSSES --
        distance_losses, distance_outputs = self.photometric_losses(inputs, outputs)
        losses.update(distance_losses)
        outputs.update(distance_outputs)

        return features, outputs, losses

    def predict_compression_seg(self, inputs):
        outputs = dict()
        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs["compression"] = self.models["compression"](features)


        return outputs, features

    @torch.no_grad()
    def compression_val(self):
        """Validate the compression model"""
        self.set_eval()
        losses = dict()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)

            outputs, _ = self.predict_compression_seg(inputs)
            losses["compression_loss"] = self.criterion(outputs["compression"],
                                                        inputs["color_aug", 0, 0])

            if self.args.use_multiple_gpu:
                losses["compression_loss"] = losses["compression_loss"].unsqueeze(0)

        # Compute stats for the tensorboard
        CompressionModel.compression_statistics(self, "val", inputs, outputs["compression"], losses)

        del inputs, outputs
        self.set_train()
        return losses["compression_loss"]

    # create default evaluator for doctests
    def save_best_compression_weights(self):
        # 2D Compression validation on each step and save model on improvements.
        compression_loss = self.compression_val( )
        if (compression_loss < self.best_compression_loss):
            self.best_compression_loss = compression_loss
            #torch.save(self.models["encoder"].state_dict(), './models/AEs/results_models_resnet_ae/Encoder.pth')
            #torch.save(self.models["decoder"].state_dict(), './models/AEs/results_models_resnet_ae/Decoder.pth')
            print("Saving better model with loss:" + str(compression_loss))


class DistanceCompressionModel(DistanceCompressionModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.mtl_loss = UncertaintyLoss(tasks=self.args.train).to(self.device)
        self.parameters_to_train += list(self.mtl_loss.parameters())
        if args.use_multiple_gpu:
            self.mtl_loss = torch.nn.DataParallel(self.mtl_loss)
        self.configure_optimizers()
        self.pre_init()
