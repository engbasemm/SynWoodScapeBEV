import os
import math
import cv2
import yaml
import torch
from PIL import Image
import torch
from torchvision import transforms, utils
import torchvision.transforms.functional as G
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchmetrics import PeakSignalNoiseRatio
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import nms
from train_utils.detection_utils import * # get_contour should be imported from here
from main import collect_args
from models.detection_decoderORG import YoloDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import Tupperware, IoU, AverageMeter
from utils import semantic_color_encoding
from train_utils.detection_utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class, bbox_iou, \
    compute_ap
from models.compressAI.priors import (
    ScaleHyperpriorDecoderAtten, FactorizedPriorDecoder, ScaleHyperpriorOrg,
    ScaleHyperpriorDecoder, ScaleHyperpriorNoEntropy, ScaleHyperprior,
    FactorizedPriorDecoderAtten
)
from models.compressAI.waseda import (
    Cheng2020Anchor, Cheng2020Attention, Cheng2020AttentionDecoder
)
import torchvision.transforms as T
import re
import numpy as np

from torch.utils.data import DataLoader
from data_loader.woodscape_loader import WoodScapeRawDataset

FRAME_RATE = 1


def parse_compression_results(file_path):
    """
    Parses the compression results from a text file and extracts
    image name, bits, PSNR, and SSIM for each image.
    """
    results = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: The HEVC/VVC log file '{file_path}' was not found. Skipping parsing for this quality.")
        return {}
    except Exception as e:
        print(f"Warning: An error occurred while reading the HEVC/VVC log file '{file_path}': {e}. Skipping parsing.")
        return {}

    image_blocks = re.split(r'(HM software: Encoder Version \[[\d.]+\](?: \(including RExt\))?\[.*?\]\[.*?\]\[.*?\])',
                            content)

    processed_blocks = []
    for i in range(1, len(image_blocks), 2):
        if i + 1 < len(image_blocks):
            processed_blocks.append(image_blocks[i] + image_blocks[i + 1])
        else:
            processed_blocks.append(image_blocks[i])

    for block in processed_blocks:
        if not block.strip():
            continue

        image_name = None
        bits = None
        psnr = None
        ssim = None

        image_name_match = re.search(r'Input\s+File\s+:\s+(\S+\.yuv)', block)
        if image_name_match:
            image_name = os.path.basename(image_name_match.group(1))

        bits_match = re.search(r'POC\s+\d+\s+TId:\s+\d+\s+\(.*\)\s+(\d+)\s+bits', block)
        if bits_match:
            bits = int(bits_match.group(1))

        psnr_match = re.search(r'SUMMARY.*?\n\s*\d+\s+a\s+[\d\.]+\s+([\d.]+)', block, re.DOTALL)
        if psnr_match:
            psnr = float(psnr_match.group(1))

        ssim_match = re.search(r'SUMMARY.*?\n\s*\d+\s+a\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+([\d.]+)',
                               block, re.DOTALL)
        if ssim_match:
            ssim = float(ssim_match.group(1))

        if image_name and bits is not None and psnr is not None and ssim is not None:
            results[image_name] = {
                'Bits': bits,
                'PSNR': psnr,
                'SSIM': ssim
            }
    return results


def pre_image_op(args, index, frame_index, cam_side):
    total_car1_images = 6054
    cropped_coords = dict(Car1=dict(FV=(114, 110, 1176, 610),
                                    MVL=(343, 5, 1088, 411),
                                    MVR=(185, 5, 915, 425),
                                    RV=(186, 203, 1105, 630)),
                          Car2=dict(FV=(160, 272, 1030, 677),
                                    MVL=(327, 7, 1096, 410),
                                    MVR=(175, 4, 935, 404),
                                    RV=(285, 187, 1000, 572)))
    if args.crop:
        if int(frame_index[1:]) < total_car1_images:
            cropped_coords = cropped_coords["Car1"][cam_side]
        else:
            cropped_coords = cropped_coords["Car2"][cam_side]
    else:
        cropped_coords = None

    return get_image(args, index, cropped_coords, frame_index, cam_side)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim == 3:
        a = a.unsqueeze(0)
    if b.ndim == 3:
        b = b.unsqueeze(0)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    mse = F.mse_loss(a, b).item()

    if mse == 0:
        return float('inf')

    return -10 * math.log10(mse)


def get_image(args, index, cropped_coords, frame_index, cam_side):
    recording_folder = args.rgb_images if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)

    image = Image.open(path).convert('RGB')

    image_q40, image_q42, image_q45 = None, None, None
    if args.comprae_vvc_hevc_jpeg2000 == True:
        path_q40 = os.path.join(args.dataset_dir, "rgb_images_Q40", file)
        path_q42 = os.path.join(args.dataset_dir, "rgb_images_Q42", file)
        path_q45 = os.path.join(args.dataset_dir, "rgb_images_Q45", file)
        image_q40 = Image.open(path_q40).convert('RGB')
        image_q42 = Image.open(path_q42).convert('RGB')
        image_q45 = Image.open(path_q45).convert('RGB')

    if args.comprae_vvc_hevc_jpeg2000 == True:
        return image, image_q40, image_q42, image_q45
    else:
        return image


def color_encoding_woodscape_detection():
    detection_classes = dict(vehicles=(43, 125, 255), rider=(255, 0, 0), person=(216, 45, 128),
                             traffic_sign=(255, 175, 58), traffic_light=(43, 255, 255))
    detection_color_encoding = np.zeros((5, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(detection_classes.items()):
        detection_color_encoding[i] = v
    return detection_color_encoding


def calculate_averages(data_list, keys):
    """Calculates the average for specified keys in a list of dictionaries,
    handling None/NaN values by ignoring them in the average."""
    averages = {key: [] for key in keys}
    for item in data_list:
        for key in keys:
            value = item.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                averages[key].append(value)

    result = {}
    for key, values in averages.items():
        if values:
            result[key] = sum(values) / len(values)
        else:
            result[key] = np.nan
    return result


def get_contour(bbox_coordinates, alpha=0.5):
    """
    Creates a shapely box polygon from bounding box coordinates and an optional scaling factor.
    Assumes bbox_coordinates are [cx, cy, w, h].
    """
    import shapely.geometry as sg
    from shapely.affinity import scale

    x1 = bbox_coordinates[0] - alpha * bbox_coordinates[2] / 2
    y1 = bbox_coordinates[1] - alpha * bbox_coordinates[3] / 2
    x2 = bbox_coordinates[0] + alpha * bbox_coordinates[2] / 2
    y2 = bbox_coordinates[1] + alpha * bbox_coordinates[3] / 2

    bbox = sg.box(x1, y1, x2, y2)
    return bbox


def custom_collate_fn(batch):
    collated_batch = {}
    detection_labels_list = []

    for i, sample in enumerate(batch):
        if "detection_labels" in sample and sample["detection_labels"] is not None:
            labels = sample["detection_labels"]
            if labels.numel() > 0:
                batch_idx_column = torch.full((labels.shape[0], 1), float(i), dtype=labels.dtype, device=labels.device)
                detection_labels_list.append(torch.cat((batch_idx_column, labels), dim=1))

        for key, value in sample.items():
            if key == "detection_labels":
                continue
            if key not in collated_batch:
                collated_batch[key] = []
            collated_batch[key].append(value)

    if len(detection_labels_list) > 0:
        collated_batch["detection_labels"] = torch.cat(detection_labels_list, 0)
    else:
        collated_batch["detection_labels"] = torch.empty((0, 6), dtype=torch.float32)

    for key, value_list in collated_batch.items():
        if key != "detection_labels" and len(value_list) > 0:
            if isinstance(value_list[0], torch.Tensor):
                collated_batch[key] = torch.stack(value_list)
            elif isinstance(value_list[0], str):
                collated_batch[key] = value_list
            else:
                collated_batch[key] = default_collate(value_list)
        elif key != "detection_labels" and len(value_list) == 0:
            collated_batch[key] = []

    return collated_batch


@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""

    feed_height = 288
    feed_width = 544
    img_size = [feed_width, feed_height]

    semantic_color_coding = semantic_color_encoding(args)
    detection_color_encoding = color_encoding_woodscape_detection() # Added as requested

    val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir,
                                      path_file=args.val_file,
                                      is_train=False,
                                      config=args)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=custom_collate_fn)

    all_val_image_paths = [line.rstrip('\n') for line in open(args.val_file)]

    # Global crop coordinates for WoodScape
    total_car1_images = 6054
    WOODSCAPE_CROPPED_COORDS = dict(
        Car1=dict(FV=(114, 110, 1176, 610),
                  MVL=(343, 5, 1088, 411),
                  MVR=(185, 5, 915, 425),
                  RV=(186, 203, 1105, 630)),
        Car2=dict(FV=(160, 272, 1030, 677),
                  MVL=(327, 7, 1096, 410),
                  MVR=(175, 4, 935, 404),
                  RV=(285, 187, 1000, 572))
    )

    # --- Pre-parse HEVC/VVC logs once and calculate overall HEVC metrics ---
    overall_hevc_metrics = {
        'hevc_q40_log_bpp': [], 'hevc_q40_calc_psnr_vs_orig': [], 'hevc_q40_calc_ms_ssim_vs_orig': [],
        'hevc_q42_log_bpp': [], 'hevc_q42_calc_psnr_vs_orig': [], 'hevc_q42_calc_ms_ssim_vs_orig': [],
        'hevc_q45_log_bpp': [], 'hevc_q45_calc_psnr_vs_orig': [], 'hevc_q45_calc_ms_ssim_vs_orig': []
    }

    hevc_q40_results = {}
    hevc_q42_results = {}
    hevc_q45_results = {}

    if args.comprae_vvc_hevc_jpeg2000:
        print("\n=> Pre-parsing HEVC/VVC log files and calculating overall HEVC metrics...")
        log_q40_path = os.path.join(args.dataset_dir, "rgb_images_Q40", "log_compression_hevc_q40.txt")
        log_q42_path = os.path.join(args.dataset_dir, "rgb_images_Q42", "log_compression_hevc_q42.txt")
        log_q45_path = os.path.join(args.dataset_dir, "rgb_images_Q45", "log_compression_hevc_q45.txt")

        hevc_q40_results = parse_compression_results(log_q40_path)
        hevc_q42_results = parse_compression_results(log_q42_path)
        hevc_q45_results = parse_compression_results(log_q45_path)

        # First pass over the dataset to compute HEVC metrics
        for batch_i, inputs in enumerate(tqdm(val_loader, desc="Calculating HEVC Baselines")):
            current_image_path = all_val_image_paths[batch_i]
            if current_image_path.endswith(f"_detection.png") or current_image_path.endswith(f"_semantic.png"):
                continue

            frame_index, cam_side = os.path.basename(current_image_path).split('.')[0].split('_')
            current_image_filename_yuv = f"{frame_index}_{cam_side}.yuv"

            full_original_image_path = os.path.join(args.dataset_dir, args.rgb_images, f"{frame_index}_{cam_side}.png")
            pil_true_original_image = Image.open(full_original_image_path).convert('RGB')
            true_original_image_tensor = transforms.ToTensor()(
                pil_true_original_image.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            ).unsqueeze(0).to(args.device)

            _, pil_input_image_q40_full, pil_input_image_q42_full, pil_input_image_q45_full = pre_image_op(args, 0, frame_index, cam_side)

            input_image_q40_tensor = transforms.ToTensor()(
                pil_input_image_q40_full.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            ).unsqueeze(0).to(args.device)
            input_image_q42_tensor = transforms.ToTensor()(
                pil_input_image_q42_full.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            ).unsqueeze(0).to(args.device)
            input_image_q45_tensor = transforms.ToTensor()(
                pil_input_image_q45_full.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            ).unsqueeze(0).to(args.device)

            # HEVC Q40
            data_q40 = hevc_q40_results.get(current_image_filename_yuv)
            if data_q40:
                bpp_q40 = data_q40['Bits'] / (feed_width * feed_height)
                overall_hevc_metrics['hevc_q40_log_bpp'].append(bpp_q40)

            psnr_value_q40_vs_orig = psnr(true_original_image_tensor, input_image_q40_tensor)
            ms_ssim_q40_vs_orig = ms_ssim(true_original_image_tensor, input_image_q40_tensor, data_range=1.0).item()
            overall_hevc_metrics['hevc_q40_calc_psnr_vs_orig'].append(psnr_value_q40_vs_orig)
            overall_hevc_metrics['hevc_q40_calc_ms_ssim_vs_orig'].append(ms_ssim_q40_vs_orig)

            # HEVC Q42
            data_q42 = hevc_q42_results.get(current_image_filename_yuv)
            if data_q42:
                bpp_q42 = data_q42['Bits'] / (feed_width * feed_height)
                overall_hevc_metrics['hevc_q42_log_bpp'].append(bpp_q42)

            psnr_value_q42_vs_orig = psnr(true_original_image_tensor, input_image_q42_tensor)
            ms_ssim_q42_vs_orig = ms_ssim(true_original_image_tensor, input_image_q42_tensor, data_range=1.0).item()
            overall_hevc_metrics['hevc_q42_calc_psnr_vs_orig'].append(psnr_value_q42_vs_orig)
            overall_hevc_metrics['hevc_q42_calc_ms_ssim_vs_orig'].append(ms_ssim_q42_vs_orig)

            # HEVC Q45
            data_q45 = hevc_q45_results.get(current_image_filename_yuv)
            if data_q45:
                bpp_q45 = data_q45['Bits'] / (feed_width * feed_height)
                overall_hevc_metrics['hevc_q45_log_bpp'].append(bpp_q45)

            psnr_value_q45_vs_orig = psnr(true_original_image_tensor, input_image_q45_tensor)
            ms_ssim_q45_vs_orig = ms_ssim(true_original_image_tensor, input_image_q45_tensor, data_range=1.0).item()
            overall_hevc_metrics['hevc_q45_calc_psnr_vs_orig'].append(psnr_value_q45_vs_orig)
            overall_hevc_metrics['hevc_q45_calc_ms_ssim_vs_orig'].append(ms_ssim_q45_vs_orig)
        print("=> HEVC Baseline calculation complete.\n")

    # Average the collected HEVC metrics
    for key in overall_hevc_metrics:
        if overall_hevc_metrics[key]:
            overall_hevc_metrics[key] = np.mean(overall_hevc_metrics[key])
        else:
            overall_hevc_metrics[key] = np.nan

    # Define DL model configurations
    dl_model_configs = [
        {
            "name": "Model_A_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_chengAttenDecoder.05_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "69/68400"
        },
        # Add more DL model configurations here if needed
        # {
        #     "name": "Model_B_Factorized",
        #     "compression_decoder_type": "factorizedPriorDecoder",
        #     "pretrained_path_segment": "path/to/model_B_weights",
        #     "batch_suffix": "some_batch_number"
        # },
    ]

    all_dl_model_overall_metrics = {}

    # Load the common compression_model once
    if args.compression_quality == 0:
        compression_model = ScaleHyperprior(128, 192).to(args.device)
        checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q0.pth')
    elif args.compression_quality == 4:
        compression_model = ScaleHyperprior(128, 256).to(args.device)
        checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q4.pth')
    elif args.compression_quality >= 8:
        compression_model = Cheng2020Attention(192).to(args.device)
        checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_cheng_atten_Q3_2023.pth')
    compression_model.load_state_dict(checkpoint)
    compression_model.eval()

    for dl_config in dl_model_configs:
        model_name = dl_config["name"]
        print(f"\n" + "=" * 80)
        print(f"Evaluating DL Model: {model_name}")
        print("=" * 80)

        current_compression_decoder_name = dl_config["compression_decoder_type"]
        current_encoder_name = "encoder" # Renamed as requested
        current_bitrates_testing = dl_config["pretrained_path_segment"]
        current_batch_number = dl_config["batch_suffix"]

        run_bitrates = current_bitrates_testing + "/models/weights_" + current_batch_number
        eval_output_dir_model = os.path.join(args.output_directory, model_name)
        os.makedirs(eval_output_dir_model, exist_ok=True)

        encoder_path = os.path.join(args.pretrained_weights + run_bitrates, current_encoder_name + ".pth")
        compression_path = os.path.join(args.pretrained_weights + run_bitrates, "decoder.pth") # Assuming generic name
        depth_decoder_path = os.path.join(args.pretrained_weights + run_bitrates, "detection.pth")
        semantic_decoder_path = os.path.join(args.pretrained_weights + run_bitrates, "semantic.pth")

        # Load models for the current DL configuration
        encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
        loaded_dict_enc = torch.load(encoder_path, map_location=args.device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc, strict=False)
        encoder.eval()
        for m in encoder.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

        decoder = YoloDecoder(encoder.num_ch_enc, args).to(args.device)
        loaded_dict = torch.load(depth_decoder_path, map_location=args.device, weights_only=True)
        decoder.load_state_dict(loaded_dict)
        decoder.eval()

        semantic_decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(args.device)
        loaded_dict = torch.load(semantic_decoder_path, map_location=args.device, weights_only=True)
        semantic_decoder.load_state_dict(loaded_dict)
        semantic_decoder.eval()

        if current_compression_decoder_name == "factorizedPriorDecoder":
            if (args.network_layers == 18):
                compression_decoder = FactorizedPriorDecoderAtten(64, 256).to(args.device)
            elif (args.network_layers == 50):
                compression_decoder = FactorizedPriorDecoderAtten(64, 1024).to(args.device)
        elif current_compression_decoder_name == "chengAttenDecoder":
            compression_decoder = Cheng2020AttentionDecoder(64).to(args.device)
        elif current_compression_decoder_name == "ScaleHyperprior":
            compression_decoder = ScaleHyperpriorDecoderAtten(64, 128).to(args.device)

        loaded_dict = torch.load(compression_path)
        compression_decoder.load_state_dict(loaded_dict)
        compression_decoder.eval()

        # Initialize accumulators for the current DL model
        labels_overall_dl = []
        sample_metrics_overall_dl = []
        semantic_metric_overall_dl = IoU(args.semantic_num_classes, args.dataset, ignore_index=None)
        semantic_acc_meter_overall_dl = AverageMeter()
        dl_metrics_current_model = []

        # Initialize video writer for current model, outside the image loop
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(os.path.join(eval_output_dir_model, f"{model_name}_detection_video.mp4"), fourcc, FRAME_RATE, (feed_width, feed_height))


        # Iterate through the dataset for the current DL model
        for batch_i, inputs in enumerate(tqdm(val_loader, desc=f"Processing images for {model_name}")):
            for key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[key] = ipt.to(args.device)

            current_image_path = all_val_image_paths[batch_i]
            if current_image_path.endswith(f"_detection.png") or current_image_path.endswith(f"_semantic.png"):
                continue

            frame_index, cam_side = os.path.basename(current_image_path).split('.')[0].split('_')

            full_original_image_path = os.path.join(args.dataset_dir, args.rgb_images, f"{frame_index}_{cam_side}.png")
            pil_true_original_image = Image.open(full_original_image_path).convert('RGB')
            true_original_image_tensor = transforms.ToTensor()(
                pil_true_original_image.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            ).unsqueeze(0).to(args.device)

            with torch.no_grad():
                out_dec_full_image_for_kpis = compression_model.forward(true_original_image_tensor)
                x_hat_full_image_for_kpis = out_dec_full_image_for_kpis["x_hat"]

                num_pixels_full = true_original_image_tensor.shape[2] * true_original_image_tensor.shape[3]

                bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels_full))
                    for likelihoods in out_dec_full_image_for_kpis["likelihoods"].values()
                )
                bpp = round(bpp.item(), 3)
                psnr_dl_value = round(psnr(true_original_image_tensor, x_hat_full_image_for_kpis), 2)
                ms_ssim_dl_value = round(ms_ssim(true_original_image_tensor, x_hat_full_image_for_kpis, data_range=1.0).item(), 5)

                dl_metrics_current_model.append({
                    'PSNR': psnr_dl_value,
                    'BPP': bpp,
                    'MS-SSIM': ms_ssim_dl_value
                })

                if args.crop:
                    out_dec_cropped_for_downstream = compression_model.forward(inputs["color_aug", 0, 0])
                    input_image_for_downstream_tasks = out_dec_cropped_for_downstream["x_hat"]
                else:
                    input_image_for_downstream_tasks = x_hat_full_image_for_kpis

            features = encoder(input_image_for_downstream_tasks)
            out_dec_comp_decoder = compression_decoder(features)
            semantic_pred = semantic_decoder(features)[("semantic", 0)]

            yolo_outputs = decoder(features, img_dim=[feed_width, feed_height])["yolo_outputs"]

            current_batch_targets_raw = inputs[("detection_labels", 0)].squeeze(0).cpu()

            if current_batch_targets_raw.numel() > 0:
                labels_overall_dl.extend(current_batch_targets_raw[:, 1].tolist())

            targets_for_batch_stats = current_batch_targets_raw.clone()
            if targets_for_batch_stats.numel() > 0:
                targets_for_batch_stats[:, 2:6] = xywh2xyxy(targets_for_batch_stats[:, 2:6])
                targets_for_batch_stats[:, 2] *= img_size[0]
                targets_for_batch_stats[:, 3] *= img_size[1]
                targets_for_batch_stats[:, 4] *= img_size[0]
                targets_for_batch_stats[:, 5] *= img_size[1]

            outputs_nms = non_max_suppression(yolo_outputs,
                                              conf_thres=args.detection_conf_thres,
                                              nms_thres=args.detection_nms_thres)

            sample_metrics_overall_dl.extend(
                get_batch_statistics(outputs_nms, targets_for_batch_stats, iou_threshold=0.5, args=args))

            _, predictions_semantic = torch.max(semantic_pred.data, 1)
            semantic_labels_current_image = inputs["semantic_labels", 0, 0]
            semantic_metric_overall_dl.add(predictions_semantic, semantic_labels_current_image)

            output_name = os.path.splitext(os.path.basename(current_image_path))[0]
            name_dest_npy = os.path.join(eval_output_dir_model, f"{output_name}_semantic.npy")
            semantic_pred_data = semantic_pred.data
            _, predictions = torch.max(semantic_pred_data.squeeze(0), 0)
            predictions = predictions.byte().cpu().detach().numpy()
            np.save(name_dest_npy, predictions)

            alpha = 0.5
            color_semantic = np.array(transforms.ToPILImage()(input_image_for_downstream_tasks.cpu().squeeze(0)))
            not_background = predictions != 0
            color_semantic[not_background, ...] = (color_semantic[not_background, ...] * (1 - alpha) +
                                                   semantic_color_coding[
                                                       predictions[not_background]] * alpha)
            semantic_color_mapped_pil = Image.fromarray(color_semantic)

            name_dest_im_semantic = os.path.join(eval_output_dir_model, f"{output_name}_semantic.png") # Renamed for clarity

            pil_input_image = G.to_pil_image(input_image_for_downstream_tasks.cpu().squeeze(0))
            rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
            rgb_color_pred_concat.paste(pil_input_image, (0, 0))
            rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
            rgb_color_pred_concat.save(name_dest_im_semantic)

            # --- Detection Visualization Logic (using input image) ---
            output_name_det_viz = os.path.splitext(os.path.basename(current_image_path))[0]
            name_dest_im_det_viz = os.path.join(eval_output_dir_model, f"{output_name_det_viz}_detection.png")

            # Use the input image (scaled to feed_width x feed_height) for visualization
            img_d = inputs["color", 0, 0].cpu().detach().numpy().squeeze(0)
            img_cpu_det_viz = (np.transpose(img_d, (1, 2, 0)) * 255).astype(np.uint8)

            # Ensure it's BGR for OpenCV drawing
            if img_cpu_det_viz.shape[2] == 3:
                img_cpu_det_viz = cv2.cvtColor(img_cpu_det_viz, cv2.COLOR_RGB2BGR)

            if not outputs_nms[0] is None:
                outputs = torch.cat(outputs_nms, dim=0)
                for box in outputs:
                    # Get class name and color
                    cls_pred = int(box[6])
                    class_color = (detection_color_encoding[cls_pred]).tolist()
                    x1, y1, x2, y2, conf = box[0], box[1], box[2], box[3], box[4] # Extract x1, y1, x2, y2 and confidence

                    # Calculate cx, cy, w, h from x1,y1,x2,y2 for get_contour
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    box_for_contour_input = [cx.item(), cy.item(), w.item(), h.item()]

                    # Call get_contour with correct parameters and a fixed alpha (e.g., 0.5)
                    bbox_poly = get_contour(box_for_contour_input, alpha=0.5).exterior.coords

                    # Convert contour points to integer for cv2.drawContours
                    boxes_cv_pts = np.int0(bbox_poly)[0:4] # Take first 4 points for the rectangle
                    box_cv_format = np.int0([[b[0], b[1]] for b in boxes_cv_pts]) # Ensure format is compatible with cv2

                    cv2.drawContours(img_cpu_det_viz, [box_cv_format], 0, class_color, thickness=2)

                    # Font scale is relative to the image dimensions
                    font_scale = 1.5e-3 * img_cpu_det_viz.shape[0]
                    cv2.putText(img_cpu_det_viz, str(f"{conf:.2f}"), (np.uint16(x1) - 5, np.uint16(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

                video.write(img_cpu_det_viz) # Write the image with drawn boxes to video
                cv2.imwrite(name_dest_im_det_viz, img_cpu_det_viz) # Save the image with bounding boxes
            else:
                # If no detections, just save the input image for detection visualization
                video.write(img_cpu_det_viz)
                cv2.imwrite(name_dest_im_det_viz, img_cpu_det_viz)

            output_name_reconstructed = os.path.splitext(os.path.basename(current_image_path))[0]
            name_dest_im_reconstructed = os.path.join(eval_output_dir_model, f"{output_name_reconstructed}_reconstructed.png")
            utils.save_image(input_image_for_downstream_tasks.data, name_dest_im_reconstructed, normalize=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        video.release()

        # Calculate overall metrics for the current DL model
        overall_ap_dl_np = np.array([])
        if len(sample_metrics_overall_dl) > 0:
            true_positives_overall, pred_scores_overall, pred_labels_overall = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics_overall_dl))
            ]
            _, _, overall_ap_dl_np, _, _ = ap_per_class(
                true_positives_overall, pred_scores_overall, pred_labels_overall, np.array(labels_overall_dl)
            )
        overall_map_dl = np.nanmean(overall_ap_dl_np) if overall_ap_dl_np.size > 0 else 0.0

        _, overall_miou_dl = semantic_metric_overall_dl.value()
        avg_dl_comp = calculate_averages(dl_metrics_current_model, ['PSNR', 'BPP', 'MS-SSIM'])

        all_dl_model_overall_metrics[model_name] = {
            "mAP": overall_map_dl,
            "mIoU": overall_miou_dl,
            "PSNR": avg_dl_comp['PSNR'],
            "BPP": avg_dl_comp['BPP'],
            "MS-SSIM": avg_dl_comp['MS-SSIM']
        }
        print(f"Finished evaluation for DL Model: {model_name}")

    # --- Final Consolidated Summary Table ---
    print("\n\n" + "=" * 100)
    print(f"{'Overall Evaluation Summary':^100}")
    print("=" * 100)

    # HEVC Baseline Results
    print("\n--- HEVC Baseline (vs Original) ---")
    print(f"HEVC Q40: BPP: {overall_hevc_metrics['hevc_q40_log_bpp']:.3f} | PSNR: {overall_hevc_metrics['hevc_q40_calc_psnr_vs_orig']:.2f} | MS-SSIM: {overall_hevc_metrics['hevc_q40_calc_ms_ssim_vs_orig']:.5f}")
    print(f"HEVC Q42: BPP: {overall_hevc_metrics['hevc_q42_log_bpp']:.3f} | PSNR: {overall_hevc_metrics['hevc_q42_calc_psnr_vs_orig']:.2f} | MS-SSIM: {overall_hevc_metrics['hevc_q42_calc_ms_ssim_vs_orig']:.5f}")
    print(f"HEVC Q45: BPP: {overall_hevc_metrics['hevc_q45_log_bpp']:.3f} | PSNR: {overall_hevc_metrics['hevc_q45_calc_psnr_vs_orig']:.2f} | MS-SSIM: {overall_hevc_metrics['hevc_q45_calc_ms_ssim_vs_orig']:.5f}")
    print("-" * 100)

    # DL Model Results
    print("\n--- DL Model Results (Overall Averages vs Original) ---")
    print(f"{'Model Name':<25} | {'DL BPP':<10} | {'DL PSNR':<10} | {'DL MS-SSIM':<12} | {'mAP':<8} | {'mIoU':<8}")
    print("-" * 100)
    for model_name, metrics in all_dl_model_overall_metrics.items():
        print(f"{model_name:<25} | "
              f"{metrics['BPP']:.3f}{'':<7} | "
              f"{metrics['PSNR']:.2f}{'':<7} | "
              f"{metrics['MS-SSIM']:.5f}{'':<7} | "
              f"{metrics['mAP']:.4f}{'':<3} | "
              f"{metrics['mIoU']:.4f}")

    print("=" * 100)
    print(f"=> LoL! beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
