"""
Trainer for unified BEV + Compression model.

This trainer wires the ResNet encoder to the BEV_MTL_Compression wrapper,
which uses a shared encoder to feed both a BEV fusion pipeline and a
per-camera compression model.

Author: patched for Basem Barakat
Reviewed_by: Gemini
"""

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

# dataset and encoder
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from models.resnet import ResnetEncoder

# unified model and loss
try:
    from models.bev_models import BEV_MTL_Compression, StitchingLoss
except Exception as e:
    raise ImportError("Could not import from models.bev_models. "
                      "Make sure you have placed the provided module at models/bev_models.py. "
                      f"Original import error: {e}")


class BEVCompressionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.args.output_directory, exist_ok=True)

        # Dataloader setup
        print("=> Loading dataset")
        grid_config = {
            'xbound': [-75.0, 75.0, 0.5], 'ybound': [-120.0, 90.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 60.0, 1.0],
        }
        train_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=args.train_file, is_train=True, config=args, grid_config=grid_config)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                       collate_fn=train_dataset.collate_fn)

        val_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir, path_file=args.val_file, is_train=False, config=args, grid_config=grid_config)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                     collate_fn=val_dataset.collate_fn)

        self.num_views = len(train_dataset.all_cam_sides)
        print(f"=> Dataloader ready. Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

        # --- Models ---
        print("=> Initializing models")
        self.models = {}
        self.models["encoder"] = ResnetEncoder(args.network_layers, pretrained=True).to(self.device)

        encoder_channels = self.models["encoder"].num_ch_enc
        print(f"=> Using encoder feature channels: {encoder_channels}")

        self.models["mtl"] = BEV_MTL_Compression(
            num_cameras=self.num_views,
            per_camera_in_channels_list=encoder_channels,
            bev_proj_channels= 64,#args.bev_proj_channels,
            compression_M= 192,#args.compression_M,
            backbone_fuse_mid= 256 #args.bev_channels
        ).to(self.device)

        # --- Losses ---
        self.bev_criterion = StitchingLoss(alpha=args.ssim_weight)
        self.recon_criterion = StitchingLoss(alpha=args.ssim_weight)
        self.alpha = 1.0# args.loss_alpha_bev
        self.beta = 1.0#args.loss_alpha_recon

        # --- Optimizer & Scheduler ---
        parameters = list(self.models["encoder"].parameters()) + list(self.models["mtl"].parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # --- Logging & Bookkeeping ---
        self.step = 0
        self.best_val_loss = float("inf")
        self.log_path = os.path.join(args.output_directory, args.model_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writers = {"train": SummaryWriter(os.path.join(self.log_path, "train")),
                        "val": SummaryWriter(os.path.join(self.log_path, "val"))}
        self.start_epoch = 0

    def set_train(self):
        for m in self.models.values(): m.train()

    def set_eval(self):
        for m in self.models.values(): m.eval()

    def _get_features_per_camera(self, inputs, color_key):
        """
        Returns:
            list (len=V) of lists (len=num_scales) of tensors [B, C, H, W]
        """
        features_per_camera = [[] for _ in range(self.num_views)]
        for i, cam_side in enumerate(self.train_loader.dataset.all_cam_sides):
            input_tensor = inputs[color_key + (cam_side,)]
            # encoder returns a list of features for different scales
            features_per_camera[i] = self.models["encoder"](input_tensor)
        return features_per_camera

    def _process_batch(self, inputs, is_train=True):
        for k, v in inputs.items():
            if torch.is_tensor(v): inputs[k] = v.to(self.device)

        color_key = ("color_aug", 0, 0) if is_train else ("color", 0, 0)
        features_per_camera = self._get_features_per_camera(inputs, color_key)

        outputs = self.models["mtl"](features_per_camera)
        bev_pred = outputs["bev"]
        recon_pred_list = outputs["compression"]["x_hat"] # List of [B, 3, H, W]

        gt_bev = inputs[("bev_gt", 0, 0)]
        gt_recon_list = [inputs[color_key + (cam_side,)] for cam_side in self.train_loader.dataset.all_cam_sides]

        # --- Loss Calculation ---
        bev_pred_resized = F.interpolate(bev_pred, size=gt_bev.shape[-2:], mode='bilinear', align_corners=False)
        bev_loss = self.bev_criterion(bev_pred_resized, gt_bev)

        recon_loss = 0
        for i in range(self.num_views):
            #recon_pred_resized = F.interpolate(recon_pred_list[i], size=gt_recon_list[i].shape[-2:], mode='bilinear', align_corners=False)
            recon_loss += self.recon_criterion(recon_pred_list[i], gt_recon_list[i])
        recon_loss /= self.num_views # Average loss across views

        total_loss = self.alpha * bev_loss + self.beta * recon_loss

        return outputs, {"total": total_loss, "bev": bev_loss, "recon": recon_loss}, gt_bev, gt_recon_list

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.set_train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
            for batch_idx, inputs in enumerate(pbar):
                self.optimizer.zero_grad()
                outputs, losses, gt_bev, gt_recon = self._process_batch(inputs, is_train=True)
                losses["total"].backward()
                self.optimizer.step()

                pbar.set_postfix({"loss": losses["total"].item()})
                if self.step % self.args.log_frequency == 0:
                    self.writers["train"].add_scalar("loss/total", losses["total"].item(), self.step)
                    self.writers["train"].add_scalar("loss/bev", losses["bev"].item(), self.step)
                    self.writers["train"].add_scalar("loss/recon", losses["recon"].item(), self.step)

                    # Log images
                    bev_grid = make_grid(F.interpolate(outputs["bev"], size=gt_bev.shape[-2:]).cpu(), normalize=True)
                    self.writers["train"].add_image("bev/pred", bev_grid, self.step)
                    self.writers["train"].add_image("bev/gt", make_grid(gt_bev.cpu(), normalize=True), self.step)

                    recon_pred_grid = make_grid(torch.cat(outputs["compression"]["x_hat"], 0).cpu(), normalize=True, nrow=self.num_views)
                    gt_recon_grid = make_grid(torch.cat(gt_recon, 0).cpu(), normalize=True, nrow=self.num_views)
                    self.writers["train"].add_image("recon/pred_views", recon_pred_grid, self.step)
                    self.writers["train"].add_image("recon/gt_views", gt_recon_grid, self.step)

                if self.step > 0 and self.step % self.args.val_frequency == 0:
                    self.validate(epoch)
                    self.set_train()
                self.step += 1
            self.scheduler.step()

    @torch.no_grad()
    def validate(self, epoch):
        self.set_eval()
        total_loss, total_bev, total_recon, n = 0.0, 0.0, 0.0, 0
        pbar = tqdm(self.val_loader, desc="Validating")
        for i, inputs in enumerate(pbar):
            outputs, losses, gt_bev, gt_recon = self._process_batch(inputs, is_train=False)
            total_loss += losses["total"].item()
            total_bev += losses["bev"].item()
            total_recon += losses["recon"].item()
            n += 1

            if i == 0: # Log visuals for the first validation batch
                bev_grid = make_grid(F.interpolate(outputs["bev"], size=gt_bev.shape[-2:]).cpu(), normalize=True)
                self.writers["val"].add_image("bev/pred", bev_grid, self.step)
                self.writers["val"].add_image("bev/gt", make_grid(gt_bev.cpu(), normalize=True), self.step)

                recon_pred_grid = make_grid(torch.cat(outputs["compression"]["x_hat"], 0).cpu(), normalize=True, nrow=self.num_views)
                gt_recon_grid = make_grid(torch.cat(gt_recon, 0).cpu(), normalize=True, nrow=self.num_views)
                self.writers["val"].add_image("recon/pred_views", recon_pred_grid, self.step)
                self.writers["val"].add_image("recon/gt_views", gt_recon_grid, self.step)

        avg_loss, avg_bev, avg_recon = total_loss / n, total_bev / n, total_recon / n
        print(f"{Fore.GREEN}Validation: total={avg_loss:.4f}, bev={avg_bev:.4f}, recon={avg_recon:.4f}{Style.RESET_ALL}")
        self.writers["val"].add_scalar("loss/total", avg_loss, self.step)
        self.writers["val"].add_scalar("loss/bev", avg_bev, self.step)
        self.writers["val"].add_scalar("loss/recon", avg_recon, self.step)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"{Fore.GREEN}=> New best model: {avg_loss:.4f}, saving...{Style.RESET_ALL}")
            self.save_model(epoch)

    def save_model(self, epoch):
        save_dict = {
            "step": self.step, "epoch": epoch,
            "model_state_dict": {k: v.state_dict() for k, v in self.models.items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss
        }
        path = os.path.join(self.log_path, "best_model.pth")
        torch.save(save_dict, path)
        print(f"Saved model to {path}")

def main():
    parser = argparse.ArgumentParser(description="Unified BEV + Compression Trainer")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_directory", type=str, default="./bev_output")
    parser.add_argument("--model_name", type=str, default="unified_bev_comp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--network_layers", type=int, default=50, choices=[18, 50])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_frequency", type=int, default=100)
    parser.add_argument("--val_frequency", type=int, default=500)
    parser.add_argument("--loss_alpha_bev", type=float, default=1.0)
    parser.add_argument("--loss_alpha_recon", type=float, default=1.0)
    parser.add_argument("--bev_channels", type=int, default=256)
    parser.add_argument("--attn_channels", type=int, default=128)
    parser.add_argument("--ssim_weight", type=float, default=0.85)
    parser.add_argument("--bev_proj_channels", type=int, default=64)
    parser.add_argument("--compression_M", type=int, default=192)
    args = parser.parse_args()

    trainer = BEVCompressionTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
