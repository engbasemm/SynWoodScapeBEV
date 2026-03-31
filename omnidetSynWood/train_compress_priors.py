"""
Semantic segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.woodscape_loader import WoodScapeRawDataset
from omnidet.models.compressAI.priors import FactorizedPrior
from utils import TrainUtils, semantic_color_encoding, IoU
from losses.ratedistortion_loss import RateDistortionLoss
from torchvision import utils
from utils import TrainUtils, semantic_color_encoding, IoU
from losses.compression_loss import CompressionLoss
from torchvision import transforms, utils
import torch.nn.functional as F
from utils_ae import *

class CompressionPriorInit(TrainUtils):
    def __init__(self, args):
        super().__init__(args)

        compression_class_weights = dict(
            woodscape_enet=([3.25, 2.33, 20.42, 30.59, 38.4, 45.73, 10.76, 34.16, 44.3, 49.19]),
            woodscape_mfb=(0.04, 0.03, 0.43, 0.99, 2.02, 4.97, 0.17, 1.01, 3.32, 20.35))

        print(f"=> Setting Class weights based on: {args.compression_class_weighting} \n"
              f"=> {compression_class_weights[args.compression_class_weighting]}")

        #compression_class_weights = torch.tensor(compression_class_weights[args.compression_class_weighting]).to(args.device)

        # Setup Metrics
        self.metric = IoU(args.compression_num_classes, args.dataset, ignore_index=None)


        self.compression_criterion = RateDistortionLoss()

        self.best_compression_iou = 0.0
        self.alpha = 0.5  # to blend compression predictions with color image
        self.color_encoding = semantic_color_encoding(args)



class CompressionPriorModel(CompressionPriorInit):
    def __init__(self, args):
        super().__init__(args)

        # --- Init model ---
        self.models["compressionPrior"] = FactorizedPrior(128,192).to(self.device)
        #self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
        #self.models["compression"] = SemanticDecoder(self.models["encoder"].num_ch_enc,
        #                                          n_classes=args.compression_num_classes).to(self.device)

        #self.parameters_to_train += list(self.models["encoder"].parameters())
        #self.parameters_to_train += list(self.models["compression"].parameters())
        self.parameters_to_train += list(self.models["compressionPrior"].parameters())

        if args.use_multiple_gpu:
            #self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            #self.models["compression"] = torch.nn.DataParallel(self.models["compression"])
            self.models["compressionPrior"] = torch.nn.DataParallel(self.models["compressionPrior"])

        print(f"=> Training on the {self.args.dataset.upper()} dataset \n"
              f"=> Training model named: {self.args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {self.args.output_directory} \n"
              f"=> Training is using the cuda device id: {self.args.cuda_visible_devices} \n"
              f"=> Loading {self.args.dataset} training and validation dataset")

        # --- Load Data ---
        train_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                            path_file=args.train_file,
                                            is_train=True,
                                            config=args)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       drop_last=False)

        val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True,
                                     drop_last=True)

        print(f"=> Total number of training examples: {len(train_dataset)} \n"
              f"=> Total number of validation examples: {len(val_dataset)}")

        self.num_total_steps = len(train_dataset) // args.batch_size * args.epochs
        self.configure_optimizers()

        if args.pretrained_weights:
            self.load_model()

        #self.save_args()

        if 'cuda' in self.device:
            torch.cuda.synchronize()


    def compression_train(self):
        l_sum = 0
        for self.epoch in range(self.args.epochs):
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()
            i = 0
            out = dict()
            for batch_idx, inputs in enumerate(self.train_loader):
                current_time = time.time()
                data_loading_time += (current_time - before_op_time)
                before_op_time = current_time
                # -- PUSH INPUTS DICT TO DEVICE --
                self.inputs_to_device(inputs)
                x = inputs["color", 0, 0].cuda()
                out = self.models["compressionPrior"].forward(x)


                #utils.save_image(out["x_hat"].data, 'results/test.png', normalize=True)
                #losses = dict()
                losses = self.compression_criterion(out, x)
                l_sum += losses["compression_loss"]
                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()
                losses["compression_loss"].backward()
                self.optimizer.step()

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["mse_loss"].cpu().data, data_loading_time, gpu_time)
                    self.compression_statistics("train", inputs, out, losses)
                    data_loading_time = 0
                    gpu_time = 0

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            val_metrics = self.compression_val()
            #print(self.epoch, "IoU:", val_metrics["mean_iou"])
            #if val_metrics["mean_iou"] >= self.best_compression_iou:
            #    print(f"=> Saving model weights with mean_iou of {val_metrics['mean_iou']:.3f} "
            #          f"at step {self.step} on {self.epoch} epoch.")
            #    self.best_compression_iou = val_metrics["mean_iou"]
             #   self.save_model()
            self.save_model()
            #self.lr_scheduler.step(val_metrics["mean_iou"])

        print("Training complete!")



    @torch.no_grad()
    def compression_val(self):
        """Validate the compression model"""
        self.set_eval()
        losses = dict()
        i = 0
        #self.models["compressionPrior"].update()
        for inputs in self.val_loader:
            self.inputs_to_device(inputs)

            out = self.models["compressionPrior"].forward(inputs["color", 0, 0])
            losses = self.compression_criterion(out, inputs["color", 0, 0])

            #utils.save_image(out["x_hat"].data, 'results/' + str(i) + '.png', normalize=True)
            #i += 1

            #_, predictions = torch.max(out["likelihoods"].values(), 1)
            #self.metric.add(predictions, inputs["color", 0, 0])
        #outputs["class_iou"], outputs["mean_iou"] = self.metric.value()

        # Compute stats for the tensorboard
        self.compression_statistics("val", inputs, out, losses)

        del inputs, losses
        self.set_train()

        return out


    def compression_statistics(self, mode, inputs, outputs, losses) -> None:

        writer = self.writers[mode]

        orig_grid = make_grid(inputs["color", 0, 0].data, nrow=self.args.batch_size)

        writer.add_image('Original images', orig_grid, self.step)
        #writer.add_graph(visualizemodel ,inputs["color", 0, 0].data)
        #writer.add_graph(self.models["encoder"], inputs["color", 0, 0].data)
        #writer.add_graph(self.models["decoder"], compressed_data.data)
        grid = make_grid(outputs["x_hat"].data, nrow=self.args.batch_size)

        if mode == 'train':
            writer.add_image('reconstructed trained images', grid, self.step)
        else :
            writer.add_image('reconstructed validated images', grid, self.step)

        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)

