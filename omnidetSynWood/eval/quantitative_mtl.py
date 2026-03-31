import os
import math
import cv2
import yaml
import torch
from PIL import Image
import numpy as np
import csv
import subprocess
import tempfile
from torchvision import transforms, utils
import torchvision.transforms.functional as G
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# --- Real Imports (uncomment and use in your project) ---
# Assuming these are custom local modules from the original project
from data_loader.woodscape_loader import WoodScapeRawDataset
from train_utils.detection_utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class, get_contour
from main import collect_args  # Retained for argument collection
from models.detection_decoderORG import YoloDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import Tupperware, IoU, semantic_color_encoding  # Retained Tupperware and IoU from utils
from models.compressAI.priors import ScaleHyperpriorDecoderAtten, FactorizedPriorDecoderAtten
from models.compressAI.waseda import Cheng2020AttentionDecoder

FRAME_RATE = 1


# --- Utility Functions ---

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calculates the Peak Signal-to-Noise Ratio between two tensors."""
    if a.ndim == 3: a = a.unsqueeze(0)
    if b.ndim == 3: b = b.unsqueeze(0)
    if a.shape != b.shape: raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a, b = a.to(torch.float32), b.to(torch.float32)
    mse = F.mse_loss(a, b).item()
    return float('inf') if mse == 0 else -10 * math.log10(mse)


def psnr_to_mse(psnr_val: float) -> float:
    """Converts PSNR to MSE, assuming max pixel value is 1.0 for float images."""
    if psnr_val == float('inf'): return 0.0
    return 1.0 / (10 ** (psnr_val / 10.0))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a single image tensor to a PIL Image."""
    if tensor.is_cuda: tensor = tensor.cpu()
    if tensor.dim() == 4 and tensor.shape[0] == 1: tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)


def color_encoding_woodscape_detection():
    detection_classes = dict(vehicles=(43, 125, 255), rider=(255, 0, 0), person=(216, 45, 128),
                             traffic_sign=(255, 175, 58), traffic_light=(43, 255, 255))
    detection_color_encoding = np.zeros((5, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(detection_classes.items()):
        detection_color_encoding[i] = v
    return detection_color_encoding


def custom_collate_fn(batch):
    collated_data = {}

    # Initialize lists for items that should be collected as lists (non-tensor, or variable length tensors)
    collated_data['original_pil_resized'] = []
    collated_data['frame_id'] = []
    collated_data['image_path'] = []

    detection_labels_to_concat = []  # Temporary list to collect detection labels for concatenation

    for i, sample in enumerate(batch):
        # Handle Detection Labels:
        if ("detection_labels", 0) in sample and sample[("detection_labels", 0)] is not None:
            labels_raw_from_sample = sample[("detection_labels", 0)]  # Original shape: (1, num_objects, 6)
            # Ensure labels_for_current_sample is 2D: (num_objects, 6) or (0, 6) if empty
            labels_for_current_sample = labels_raw_from_sample.view(-1, 6)

            # Do NOT prepend batch index here. The batch index will be handled explicitly in test_online_compression
            if labels_for_current_sample.numel() > 0:
                detection_labels_to_concat.append(labels_for_current_sample)

        # Handle non-tensor items (these should remain lists of their original types)
        if "original_pil_resized" in sample:
            collated_data['original_pil_resized'].append(sample['original_pil_resized'])
        if "frame_id" in sample:
            collated_data['frame_id'].append(sample['frame_id'])
        if "image_path" in sample:
            collated_data['image_path'].append(sample['image_path'])

        # Handle all other keys (which are expected to be tensors and will be stacked)
        for key, value in sample.items():
            if key == ("detection_labels", 0) or \
                    key == "original_pil_resized" or \
                    key == "frame_id" or \
                    key == "image_path":
                continue

            if key not in collated_data:
                collated_data[key] = []
            collated_data[key].append(value)

    # After processing all samples in the batch:

    # Finalize Detection Labels: Concatenate all collected detection labels
    if len(detection_labels_to_concat) > 0:
        # This will be (total_objects_in_batch, 6): [class_id, x, y, w, h, conf]
        collated_data["detection_labels"] = torch.cat(detection_labels_to_concat, 0)
    else:
        # If no detection labels were found in the entire batch, provide an empty tensor of the correct shape (6 columns)
        collated_data["detection_labels"] = torch.empty((0, 6), dtype=torch.float32)

    # Stack the lists of tensors for other keys (e.g., "color", "semantic_labels")
    for key in list(collated_data.keys()):
        if key in ["original_pil_resized", "frame_id", "image_path", "detection_labels"]:
            continue

        value_list = collated_data[key]
        if len(value_list) > 0 and isinstance(value_list[0], torch.Tensor):
            collated_data[key] = torch.stack(value_list)
        elif len(value_list) == 0:
            del collated_data[key]

    return collated_data


# --- Online Compression Functions ---

def compress_hevc_online(original_image_tensor: torch.Tensor, qp_value: int, hevc_encoder_cfg_main: str,
                         hevc_encoder_cfg_per_sequence: str, tapp_encoder_path: str, tapp_decoder_path: str):
    """
    Compresses an image using HEVC via a simulated HM software pipeline (ffmpeg + TAppEncoder/TAppDecoder).
    The 'qp_value' is used for the HM encoder.
    """
    pil_image = tensor_to_pil(original_image_tensor)
    w, h = pil_image.size
    num_pixels = w * h

    # Define YUV format (4:2:0 is common for HEVC)
    yuv_pixel_format = "yuv420p"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_png_path = os.path.join(tmpdir, "input.png")
        input_yuv_path = os.path.join(tmpdir, "input.yuv")
        bitstream_hevc_path = os.path.join(tmpdir, "bitstream.bin")
        output_yuv_path = os.path.join(tmpdir, "output.yuv")
        output_png_path = os.path.join(tmpdir, "output.png")

        # 1. Save original image as PNG
        pil_image.save(input_png_path)

        # 2. Convert PNG to YUV using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', input_png_path,
                '-s', f"{w}x{h}",  # Specify resolution for YUV
                '-pix_fmt', yuv_pixel_format,
                '-loglevel', 'error',  # Suppress verbose output
                input_yuv_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error converting PNG to YUV: {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}
        # 3. Encode YUV to HEVC using TAppEncoder
        try:
            cmd = [
                tapp_encoder_path,
                '-c', hevc_encoder_cfg_main,  # Main config file
                '-c', hevc_encoder_cfg_per_sequence,  # Per-sequence config file
                '-i', input_yuv_path,  # Use provided TAppEncoder path
                '--QP=' + str(qp_value),  # Use --QP as in user's command
                '--ConformanceWindowMode=1',  # Add conformance window mode
                '-o', output_yuv_path,  # This is the reconstructed YUV output from TAppEncoder
                '-b', bitstream_hevc_path  # This is the bitstream output
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error encoding with TAppEncoder (Is TAppEncoderStatic in PATH and config files accessible?): {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}

        # Check if bitstream was created
        if not os.path.exists(bitstream_hevc_path) or os.path.getsize(bitstream_hevc_path) == 0:
            print(f"Warning: HEVC bitstream not created or is empty for QP {qp_value}. Skipping metrics.")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}

        # Calculate BPP from the HEVC bitstream size
        bpp = (os.path.getsize(bitstream_hevc_path) * 8) / num_pixels

        # 4. Decode HEVC bitstream to YUV using TAppDecoder (This step is technically not needed if TAppEncoder outputs YUV directly,
        # but keeping it for completeness if TAppEncoder only outputs bitstream and a separate decoder is required)
        # However, the user's shell command implies TAppEncoder outputs YUV directly to _rec.yuv,
        # so this TAppDecoder call might be redundant or incorrect based on the HM workflow.
        # Given the user's shell command, the output_yuv_path from TAppEncoder should be used directly for ffmpeg recon.
        # Let's assume TAppEncoder outputs the reconstructed YUV to output_yuv_path.

        # 5. Convert YUV back to PNG using ffmpeg
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', yuv_pixel_format,
                '-s', f"{w}x{h}",  # Must specify resolution for raw YUV input
                '-i', output_yuv_path,  # Use the YUV output from TAppEncoder
                '-loglevel', 'error',  # Suppress verbose output
                output_png_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error converting YUV to PNG: {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}

        # 6. Load reconstructed PNG and calculate metrics
        reconstructed_tensor = transforms.ToTensor()(Image.open(output_png_path).convert('RGB')).unsqueeze(0).to(
            original_image_tensor.device)

        # Ensure tensors are float32 and within [0,1] range for ms_ssim
        original_tensor_float = original_image_tensor.to(torch.float32).clamp(0.0, 1.0)
        reconstructed_tensor_float = reconstructed_tensor.to(torch.float32).clamp(0.0, 1.0)

        psnr_val = psnr(original_tensor_float, reconstructed_tensor_float)
        ms_ssim_val = ms_ssim(original_tensor_float, reconstructed_tensor_float, data_range=1.0).item()

        return {'BPP': bpp, 'PSNR': psnr_val, 'MS_SSIM': ms_ssim_val, 'MSE': psnr_to_mse(psnr_val)}


def compress_jpeg2000_online(original_image_tensor: torch.Tensor, quality_level: int):
    pil_image = tensor_to_pil(original_image_tensor)
    w, h = pil_image.size
    num_pixels = w * h

    quality_to_ffmpeg_compression_level = {
        0: 15,  # Highest quality, lowest compression
        1: 25,
        2: 50,
        3: 75,
        4: 95  # Lowest quality, highest compression
    }
    ffmpeg_compression_level = quality_to_ffmpeg_compression_level.get(quality_level, 50)  # Default to 50 if not found

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.png")
        bitstream_path = os.path.join(tmpdir, "out.jp2")
        recon_path = os.path.join(tmpdir, "rec.png")

        # Save original image
        pil_image.save(input_path)

        # Compress to JPEG2000 using FFmpeg and libopenjpeg
        # Use -compression_level and ensure pixel format is appropriate
        subprocess.run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', input_path,
            '-c:v', 'libopenjpeg',  # Changed to libopenjpeg as per common usage
            '-compression_level', str(ffmpeg_compression_level),
            '-pix_fmt', 'yuv444p',  # Commonly used for JPEG2000 in ffmpeg
            bitstream_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Added stdout/stderr redirect

        # Calculate bitrate
        bpp = (os.path.getsize(bitstream_path) * 8) / num_pixels

        # Decode it back
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', bitstream_path, recon_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Added stdout/stderr redirect

        reconstructed_tensor = transforms.ToTensor()(Image.open(recon_path).convert('RGB')).unsqueeze(0).to(
            original_image_tensor.device)

        # Ensure tensors are float32 and within [0,1] range for ms_ssim
        original_tensor_float = original_image_tensor.to(torch.float32).clamp(0.0, 1.0)
        reconstructed_tensor_float = reconstructed_tensor.to(torch.float32).clamp(0.0, 1.0)

        psnr_val = psnr(original_tensor_float, reconstructed_tensor_float)
        ms_ssim_val = ms_ssim(original_tensor_float, reconstructed_tensor_float, data_range=1.0).item()

        return {
            'BPP': bpp,
            'PSNR': psnr_val,
            'MS_SSIM': ms_ssim_val,  # Standardized key to 'MS_SSIM'
            'MSE': psnr_to_mse(psnr_val)
        }


def compress_webp_online(original_image_tensor: torch.Tensor, quality_level: int):
    """
    Compresses an image using WebP via ffmpeg.
    The 'quality_level' is mapped to ffmpeg's -q:v (quality scale) from 0 (worst) to 100 (best).
    Here, Q0 (best) maps to -q:v 45, and Q4 (worst) maps to -q:v 0.
    """
    pil_image = tensor_to_pil(original_image_tensor)
    w, h = pil_image.size
    num_pixels = w * h

    # Map our quality_level (0-4) to FFmpeg's -q:v for WebP (0-100, higher is better quality)
    # Q0 (best) -> ffmpeg_q 45
    # Q4 (worst) -> ffmpeg_q 0
    # Linear mapping: ffmpeg_q = 45 - (quality_level / 4) * 45
    ffmpeg_q_value = int(45 - (quality_level / 4) * 45)
    ffmpeg_q_value = max(0, min(100, ffmpeg_q_value))  # Ensure it's within [0, 100]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.png")
        bitstream_path = os.path.join(tmpdir, "out.webp")
        recon_path = os.path.join(tmpdir, "rec.png")

        # Save original image
        pil_image.save(input_path)

        # Compress to WebP using FFmpeg
        try:
            subprocess.run([
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', input_path,
                '-c:v', 'libwebp',  # Use libwebp encoder
                '-q:v', str(ffmpeg_q_value),  # Use -q:v for quality
                bitstream_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error compressing with WebP (ffmpeg): {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}

        # Calculate bitrate
        bpp = (os.path.getsize(bitstream_path) * 8) / num_pixels

        # Decode it back
        try:
            subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', bitstream_path, recon_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error decoding WebP (ffmpeg): {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}

        reconstructed_tensor = transforms.ToTensor()(Image.open(recon_path).convert('RGB')).unsqueeze(0).to(
            original_image_tensor.device)

        # Ensure tensors are float32 and within [0,1] range for ms_ssim
        original_tensor_float = original_image_tensor.to(torch.float32).clamp(0.0, 1.0)
        reconstructed_tensor_float = reconstructed_tensor.to(torch.float32).clamp(0.0, 1.0)

        psnr_val = psnr(original_tensor_float, reconstructed_tensor_float)
        ms_ssim_val = ms_ssim(original_tensor_float, reconstructed_tensor_float, data_range=1.0).item()

        return {
            'BPP': bpp,
            'PSNR': psnr_val,
            'MS_SSIM': ms_ssim_val,
            'MSE': psnr_to_mse(psnr_val)
        }


@torch.no_grad()
def test_online_compression(args):
    """Function to evaluate DL models and traditional codecs with on-the-fly compression."""
    # --- Basic Setup ---
    feed_height = int(args.input_height)  # Use args.input_height
    feed_width = int(args.input_width)  # Use args.input_width
    img_size = [feed_width, feed_height]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset Loading ---
    val_dataset = WoodScapeRawDataset(data_path=args.dataset_dir, path_file=args.val_file, is_train=False, config=args)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            drop_last=True, collate_fn=custom_collate_fn)
    all_val_image_paths = [line.rstrip('\n') for line in open(args.val_file)]

    # --- Data Structures for Results ---
    per_image_detailed_metrics = {}
    all_dl_model_overall_metrics = {}

    # --- Codec and Model Definitions ---
    # CRF values for HEVC will now map to QP values for HM
    hevc_qps = {'HM HEVC QP33': 33, 'HM HEVC QP36': 36, 'HM HEVC QP39': 39, 'HM HEVC QP42': 42, 'HM HEVC QP45': 45}
    jpeg2000_qs = {'JPEG2000 Q0': 0, 'JPEG2000 Q1': 1, 'JPEG2000 Q2': 2, 'JPEG2000 Q3': 3, 'JPEG2000 Q4': 4}
    webp_qs = {'WebP Q0': 0, 'WebP Q1': 1, 'WebP Q2': 2, 'WebP Q3': 3, 'WebP Q4': 4}  # New WebP quality levels

    dl_model_configs = [
        {
            "name": "RES50_Baseline_no_compression",
            "compression_decoder_type": "not_applicable",
            "pretrained_path_segment": "res50_no_compress_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "99/97500"
        },
        {
            "name": "Model_A_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_cheng_attn0.08_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "98/96300"
        },
        {
            "name": "Model_B_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_chengAttenDecoder.05_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "69/68400"
        },
        {
            "name": "Model_C_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_chengAttenDecoder.03_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "92/90600"
        },
        {
            "name": "Model_D_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_cheng_attn0.02_FR_2Classes_yolo12_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "124/122100"
        },
        {
            "name": "Model_E_ChengAtten",
            "compression_decoder_type": "chengAttenDecoder",
            "pretrained_path_segment": "res50_chengAttenDecoder.01_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
            "batch_suffix": "114/111900"
        },
    ]

    semantic_color_coding = semantic_color_encoding(args)
    detection_color_encoding = color_encoding_woodscape_detection()

    # --- Main Evaluation ---
    print("Starting evaluation...")

    # --- 1. Evaluate Traditional Codecs (once per image) ---
    if not args.skip_compression:
        print("Evaluating traditional codecs...")
        # Re-initialize val_loader to ensure a fresh iteration for traditional codecs
        val_loader_codecs = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)

        for batch_i, inputs in enumerate(tqdm(val_loader_codecs, desc="Evaluating Codecs")):
            current_image_path = all_val_image_paths[batch_i]
            image_key = os.path.basename(current_image_path)

            # Initialize entry for this image if not already present
            if image_key not in per_image_detailed_metrics:
                per_image_detailed_metrics[image_key] = {'image_name': image_key}

            # true_original_image_tensor should be the resized input image tensor
            true_original_image_tensor = inputs[("color", 0, 0)].to(device)

            # Iterate through HEVC QPs (formerly CRFs)
            for name, qp in hevc_qps.items():
                try:
                    # Pass config paths and TAppEncoder/TAppDecoder paths to the compression function
                    per_image_detailed_metrics[image_key][name] = compress_hevc_online(
                        true_original_image_tensor, qp,
                        args.hevc_encoder_cfg_main, args.hevc_encoder_cfg_per_sequence,
                        args.tapp_encoder_path, args.tapp_decoder_path  # Pass new arguments
                    )
                except Exception as e:
                    print(f"ERROR: {name} failed for {image_key}: {e}")
                    per_image_detailed_metrics[image_key][name] = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan,
                                                                   'MSE': np.nan}

            for name, q in jpeg2000_qs.items():
                try:
                    per_image_detailed_metrics[image_key][name] = compress_jpeg2000_online(true_original_image_tensor,
                                                                                           q)
                except Exception as e:
                    print(f"ERROR: {name} failed for {image_key}: {e}")
                    per_image_detailed_metrics[image_key][name] = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan,
                                                                   'MSE': np.nan}

            # New: Iterate through WebP quality levels
            for name, q in webp_qs.items():
                try:
                    per_image_detailed_metrics[image_key][name] = compress_webp_online(true_original_image_tensor, q)
                except Exception as e:
                    print(f"ERROR: {name} failed for {image_key}: {e}")
                    per_image_detailed_metrics[image_key][name] = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan,
                                                                   'MSE': np.nan}

    # --- 2. Evaluate DL Models (loop per model) ---
    for dl_config in dl_model_configs:
        model_name = dl_config["name"]
        print(f"\n{'=' * 80}\nEvaluating DL Model: {model_name}\n{'=' * 80}")

        # --- Load Models for Current Config ---
        run_bitrates = os.path.join(args.pretrained_weights, dl_config["pretrained_path_segment"], "models",
                                    f"weights_{dl_config['batch_suffix']}")
        eval_output_dir_model = os.path.join(args.output_directory, model_name)
        os.makedirs(eval_output_dir_model, exist_ok=True)

        encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(device)
        # Load state_dict filtering keys
        loaded_dict_enc = torch.load(os.path.join(run_bitrates, "encoder.pth"), map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc, strict=False)
        encoder.eval()
        for m_module in encoder.modules():
            for child in m_module.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False
                    child.running_mean = None
                    child.running_var = None

        decoder = YoloDecoder(encoder.num_ch_enc, args).to(device)
        decoder.load_state_dict(torch.load(os.path.join(run_bitrates, "detection.pth"), map_location=device))
        decoder.eval()

        semantic_decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(device)
        semantic_decoder.load_state_dict(torch.load(os.path.join(run_bitrates, "semantic.pth"), map_location=device))
        semantic_decoder.eval()

        comp_decoder_map = {
            "chengAttenDecoder": Cheng2020AttentionDecoder(64),
            "factorizedPriorDecoder": FactorizedPriorDecoderAtten(64, 1024 if args.network_layers == 50 else 256),
            "ScaleHyperprior": ScaleHyperpriorDecoderAtten(64, 128)
        }
        compression_decoder = comp_decoder_map[dl_config["compression_decoder_type"]].to(device) if dl_config[
                                                                                                        "compression_decoder_type"] != "not_applicable" else None

        if compression_decoder:  # Only load state_dict if a compression decoder is applicable
            compression_decoder.load_state_dict(
                torch.load(os.path.join(run_bitrates, "decoder.pth"), map_location=device))
            compression_decoder.eval()

        # --- Initialize Accumulators for this DL model ---
        labels_overall_dl, sample_metrics_overall_dl, comp_metrics_list = [], [], []
        semantic_metric = IoU(args.semantic_num_classes, args.dataset, ignore_index=None)
        video = cv2.VideoWriter(os.path.join(eval_output_dir_model, f"{model_name}_detection_video.mp4"),
                                cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (feed_width, feed_height))

        # --- Loop through dataset for this loaded model ---
        for batch_i, inputs in enumerate(tqdm(val_loader, desc=f"Processing images for {model_name}")):
            # Move inputs to device
            for key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[key] = ipt.to(device)
            current_image_path = all_val_image_paths[batch_i]
            image_key = os.path.basename(current_image_path)  # Use basename for consistency with dict keys

            # --- Compression Metrics ---
            true_original_image_tensor = inputs[("color", 0, 0)]

            if dl_config["compression_decoder_type"] != "not_applicable":
                features_orig = encoder(true_original_image_tensor)
                out_comp_orig = compression_decoder(features_orig)
                x_hat_orig = out_comp_orig["x_hat"]
                num_pixels_full = true_original_image_tensor.shape[2] * true_original_image_tensor.shape[3]

                bpp = sum((torch.log(lk).sum() / (-math.log(2) * num_pixels_full)) for lk in
                          out_comp_orig["likelihoods"].values()).item()
                psnr_dl, msssim_dl = psnr(true_original_image_tensor, x_hat_orig), ms_ssim(true_original_image_tensor,
                                                                                           x_hat_orig,
                                                                                           data_range=1.0).item()
                comp_metrics = {'BPP': bpp, 'PSNR': psnr_dl, 'MS_SSIM': msssim_dl,
                                'MSE': psnr_to_mse(psnr_dl)}  # Standardized key to 'MS_SSIM'
                input_image_for_downstream_tasks = x_hat_orig
            else:
                # If no compression, set compression metrics to NaN and use original image for downstream tasks
                comp_metrics = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}
                input_image_for_downstream_tasks = true_original_image_tensor

            comp_metrics_list.append(comp_metrics)

            # Populate per_image_detailed_metrics for DL model
            if image_key not in per_image_detailed_metrics:
                per_image_detailed_metrics[image_key] = {
                    'image_name': image_key}  # Should already be initialized, but safe check
            per_image_detailed_metrics[image_key][model_name] = comp_metrics

            # --- Downstream Task Evaluation ---
            with torch.no_grad():
                # Correctly determine input for downstream tasks based on args.crop
                if args.crop and ("color_aug", 0, 0) in inputs:
                    # If cropping is enabled and augmented color is available, use features from it
                    features_for_downstream = encoder(inputs[("color_aug", 0, 0)])
                else:
                    # Otherwise, use features from the reconstructed/original image
                    features_for_downstream = encoder(input_image_for_downstream_tasks)

            features = features_for_downstream  # Use the features from the chosen input image
            semantic_pred = semantic_decoder(features)[("semantic", 0)]
            yolo_outputs = decoder(features, img_dim=img_size)["yolo_outputs"]

            # --- Detection Metrics and Visualization ---
            # `inputs["detection_labels"]` is now a 6-column tensor: [class_id, x, y, w, h, conf]
            targets = inputs["detection_labels"].cpu()  # Renamed for clarity

            if targets.numel() > 0:
                # The first column (index 0) is class ID
                labels_overall_dl.extend(targets[:, 0].tolist())  # Changed from index 1 to 0 for class_id

                # Rescale target coordinates (indices shifted as class_id is now at index 0)
                targets[:, 1:5] = xywh2xyxy(targets[:, 1:5])  # Changed from 2:6 to 1:5
                targets[:, 1] *= img_size[0]  # x1
                targets[:, 2] *= img_size[1]  # y1
                targets[:, 3] *= img_size[0]  # x2
                targets[:, 4] *= img_size[1]  # y2

                outputs_nms = non_max_suppression(yolo_outputs,
                                                  conf_thres=args.detection_conf_thres,
                                                  nms_thres=args.detection_nms_thres)

                # Directly pass the 6-column targets to get_batch_statistics.
                # It will be [class_id, x1_scaled, y1_scaled, x2_scaled, y2_scaled, conf]
                sample_metrics_overall_dl.extend(
                    get_batch_statistics(outputs_nms, targets, iou_threshold=0.5, args=args))

            # --- Semantic Metrics and Visualization ---
            _, predictions_semantic = torch.max(semantic_pred.data, 1)
            semantic_metric.add(predictions_semantic, inputs[("semantic_labels", 0, 0)])

            # Visualization code
            output_name = os.path.splitext(os.path.basename(current_image_path))[0]
            eval_output_dir_model_vis = os.path.join(eval_output_dir_model, "visualizations")
            os.makedirs(eval_output_dir_model_vis, exist_ok=True)

            # Save semantic prediction
            name_dest_npy = os.path.join(eval_output_dir_model_vis, f"{output_name}_semantic.npy")
            _, predictions_np = torch.max(semantic_pred.data.squeeze(0), 0)
            predictions_np = predictions_np.byte().cpu().detach().numpy()
            np.save(name_dest_npy, predictions_np)

            # Create semantic overlay image
            alpha = 0.5
            color_semantic_img_np = np.array(transforms.ToPILImage()(inputs[("color", 0, 0)].cpu().squeeze(0)))
            not_background = predictions_np != 0
            blended_image = color_semantic_img_np.copy()
            blended_image[not_background, ...] = (color_semantic_img_np[not_background, ...] * (1 - alpha) +
                                                  semantic_color_coding[predictions_np[not_background]].astype(
                                                      color_semantic_img_np.dtype) * alpha)
            semantic_color_mapped_pil = Image.fromarray(blended_image)

            name_dest_im_semantic = os.path.join(eval_output_dir_model_vis, f"{output_name}_semantic.png")
            pil_input_image = G.to_pil_image(inputs[("color", 0, 0)].cpu().squeeze(0))
            rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
            rgb_color_pred_concat.paste(pil_input_image, (0, 0))
            rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
            rgb_color_pred_concat.save(name_dest_im_semantic)

            # Detection visualization
            name_dest_im_det_viz = os.path.join(eval_output_dir_model_vis, f"{output_name}_detection.png")
            img_d_np = inputs[("color", 0, 0)].cpu().detach().numpy().squeeze(0)
            img_d_np = np.transpose(img_d_np, (1, 2, 0))  # H, W, C
            img_cpu_display = (img_d_np * 255).astype(np.uint8).copy()  # Convert to 0-255 uint8 for OpenCV

            if not outputs_nms[0] is None:
                outputs = torch.cat(outputs_nms, dim=0)
                for box in outputs:
                    cls_pred = int(box[6])  # Class ID
                    class_color = (detection_color_encoding[cls_pred]).tolist()  # Get BGR color

                    # Convert bounding box coordinates to image scale for drawing
                    x1, y1, x2, y2 = box[0:4].cpu().numpy().tolist()
                    # `get_contour` expects scaled coordinates, so assuming box[0:4] are already scaled by nms
                    # If not, need to scale them here before passing to get_contour or drawing directly.
                    # Given it works, assume nms output is in pixel coordinates.

                    # Ensure coordinates are integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(img_cpu_display, (x1, y1), (x2, y2), class_color, thickness=2)

                    # Put confidence score
                    conf = box[4]  # Confidence is at index 4 (before class) in output of nms in detection_utils
                    cv2.putText(img_cpu_display, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                video.write(cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))  # Write to video
                cv2.imwrite(name_dest_im_det_viz, cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))  # Save image
            else:
                video.write(cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))  # Write original frame if no detections
                cv2.imwrite(name_dest_im_det_viz,
                            cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))  # Save original frame if no detections

            # Save reconstructed image (or original if compression skipped)
            name_dest_im_reconstructed = os.path.join(eval_output_dir_model_vis, f"{output_name}_reconstructed.png")
            utils.save_image(input_image_for_downstream_tasks.data, name_dest_im_reconstructed, normalize=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        video.release()  # Release video writer after processing all images for this model

        # --- Calculate and Store Overall Metrics for this DL Model ---
        true_pos, pred_scores, pred_labels = (np.concatenate(x, 0) for x in
                                              list(zip(*sample_metrics_overall_dl))) if sample_metrics_overall_dl else (
            np.array([]), np.array([]), np.array([]))

        overall_ap_dl_np = np.array([])
        if len(labels_overall_dl) > 0 and true_pos.size > 0:  # Ensure there are labels and predictions to compute AP
            _, _, overall_ap_dl_np, _, _ = ap_per_class(
                true_pos, pred_scores, pred_labels, np.array(labels_overall_dl)
            )

        overall_map_dl = np.nanmean(overall_ap_dl_np) if overall_ap_dl_np.size > 0 else 0.0

        _, overall_miou_dl = semantic_metric.value()

        # Calculate average compression metrics only if compression was not skipped
        if not args.skip_compression:
            avg_dl_comp_bpp = np.nanmean([d['BPP'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_psnr = np.nanmean([d['PSNR'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_msssim = np.nanmean([d['MS_SSIM'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_mse = np.nanmean([d['MSE'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
        else:
            avg_dl_comp_bpp = np.nan
            avg_dl_comp_psnr = np.nan
            avg_dl_comp_msssim = np.nan
            avg_dl_comp_mse = np.nan

        all_dl_model_overall_metrics[model_name] = {
            "mAP": overall_map_dl,
            "mIoU": overall_miou_dl,
            "BPP": avg_dl_comp_bpp,
            "PSNR": avg_dl_comp_psnr,
            "MS_SSIM": avg_dl_comp_msssim,  # Standardized key to 'MS_SSIM'
            "MSE": avg_dl_comp_mse
        }
        print(
            f"Finished evaluation for DL Model: {model_name}. mAP: {all_dl_model_overall_metrics[model_name]['mAP']:.4f}, mIoU: {all_dl_model_overall_metrics[model_name]['mIoU']:.4f}")

    # --- Final Consolidated Summary Table (Overall Averages) ---
    print("\n\n" + "=" * 100)
    print(f"{'Overall Evaluation Summary':^100}")
    print("=" * 100)

    # Traditional Codec Results (Average over all images)
    if not args.skip_compression:
        print("\n--- Traditional Codecs (Online Encoding/Decoding vs Original) ---")
        header_trad = ["Codec"]
        metrics_keys = ['BPP', 'PSNR', 'MS_SSIM', 'MSE']  # Standardized key in this list
        for key_suffix in metrics_keys:
            header_trad.append(key_suffix)
        print(f"{header_trad[0]:<20} | {' | '.join([f'{h:<10}' for h in header_trad[1:]])}")
        print("-" * 100)

        # Collect and average traditional codec metrics
        overall_traditional_metrics = {name: {k: [] for k in metrics_keys} for name in
                                       list(hevc_qps.keys()) + list(jpeg2000_qs.keys()) + list(webp_qs.keys())}

        for image_key, image_data in per_image_detailed_metrics.items():
            for codec_name in list(hevc_qps.keys()) + list(jpeg2000_qs.keys()) + list(webp_qs.keys()):
                if codec_name in image_data:
                    for metric_key, value in image_data[codec_name].items():
                        if metric_key in metrics_keys:  # Check for exact key match
                            overall_traditional_metrics[codec_name][metric_key].append(value)

        for codec_name, metrics_data in overall_traditional_metrics.items():
            row_values = [codec_name]
            for metric_key in metrics_keys:
                avg_val = np.nanmean(metrics_data[metric_key]) if metrics_data[
                    metric_key] else np.nan  # Use np.nanmean
                row_values.append(
                    f"{avg_val:.3f}" if metric_key == 'BPP' else f"{avg_val:.2f}" if metric_key == 'PSNR' else f"{avg_val:.5f}")
            print(f"{row_values[0]:<20} | {' | '.join([f'{v:<10}' for v in row_values[1:]])})")
        print("-" * 100)

    # DL Model Results
    print("\n--- DL Model Results (Overall Averages vs Original) ---")
    print(
        f"{'Model Name':<25} | {'BPP':<10} | {'PSNR':<10} | {'MS_SSIM':<12} | {'MSE':<10} | {'mAP':<8} | {'mIoU':<8}")  # Standardized key in table header
    print("-" * 100)
    for model_name, metrics in all_dl_model_overall_metrics.items():
        print(f"{model_name:<25} | "
              f"{metrics['BPP']:.3f}{'':<7} | "
              f"{metrics['PSNR']:.2f}{'':<7} | "
              f"{metrics['MS_SSIM']:.5f}{'':<7} | "  # Standardized key
              f"{metrics['MSE']:.5f}{'':<4} | "
              f"{metrics['mAP']:.4f}{'':<3} | "
              f"{metrics['mIoU']:.4f}")
    print("=" * 100)

    # --- Per-Image Detailed Results Comparison ---
    csv_output_path = os.path.join(args.output_directory, "per_image_metrics_comparison.csv")
    print(f"\n\nSaving detailed per-image metrics to: {csv_output_path}")

    # Construct the header for the detailed table dynamically
    all_codec_names = list(hevc_qps.keys()) + list(jpeg2000_qs.keys()) + list(webp_qs.keys())
    all_model_names = sorted([config['name'] for config in dl_model_configs])

    dynamic_headers = ["Image"]
    if not args.skip_compression:
        for codec_name in all_codec_names:
            dynamic_headers.extend(
                [f"{codec_name} (BPP)", f"{codec_name} (PSNR)", f"{codec_name} (MS_SSIM)",
                 f"{codec_name} (MSE)"])  # Standardized key

    for model_name in all_model_names:
        dynamic_headers.extend(
            [f"{model_name} (BPP)", f"{model_name} (PSNR)", f"{model_name} (MS_SSIM)",
             f"{model_name} (MSE)"])  # Standardized key
        dynamic_headers.extend([f"{model_name} (mAP)", f"{model_name} (mIoU)"])

    with open(csv_output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(dynamic_headers)  # Write header row

        # Print the data rows
        sorted_image_keys = sorted(per_image_detailed_metrics.keys())
        for image_key in sorted_image_keys:
            row_data = per_image_detailed_metrics[image_key]

            # Prepare values for current row
            values = [row_data['image_name']]  # Image name for CSV

            if not args.skip_compression:
                for codec_name in all_codec_names:
                    codec_metrics = row_data.get(codec_name, {})
                    values.append(f"{codec_metrics.get('BPP', np.nan):.3f}")
                    values.append(f"{codec_metrics.get('PSNR', np.nan):.2f}")
                    values.append(f"{codec_metrics.get('MS_SSIM', np.nan):.5f}")  # Standardized key
                    values.append(f"{codec_metrics.get('MSE', np.nan):.5f}")

            for model_name in all_model_names:
                dl_metrics = row_data.get(model_name, {})
                values.append(f"{dl_metrics.get('BPP', np.nan):.3f}")
                values.append(f"{dl_metrics.get('PSNR', np.nan):.2f}")
                values.append(f"{dl_metrics.get('MS_SSIM', np.nan):.5f}")  # Standardized key
                values.append(f"{dl_metrics.get('MSE', np.nan):.5f}")

                # Fetch overall mAP and mIoU for this model from the consolidated metrics
                # These are overall metrics, not per-image, so they will be the same for all images.
                overall_model_metrics = all_dl_model_overall_metrics.get(model_name, {})
                values.append(f"{overall_model_metrics.get('mAP', np.nan):.4f}")
                values.append(f"{overall_model_metrics.get('mIoU', np.nan):.4f}")

            csv_writer.writerow(values)  # Write data row

    print("=> Per-image metrics saved to CSV.")
    print(f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)

    HM_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Adjust as needed

    # Ensure these attributes are set on args, e.g., in your config YAML or here:
    args.hevc_encoder_cfg_main = os.path.join(HM_BASE_DIR, 'HM', 'cfg', 'encoder_intra_main.cfg')
    args.hevc_encoder_cfg_per_sequence = os.path.join(HM_BASE_DIR, 'HM', 'cfg', 'per-sequence',
                                                      'Traffic_Fisheye8K.cfg')

    # --- IMPORTANT: Set the absolute paths to your TAppEncoderStatic and TAppDecoder executables ---
    # Example paths (YOU NEED TO CHANGE THESE TO YOUR ACTUAL PATHS):
    args.tapp_encoder_path = os.path.join(HM_BASE_DIR, 'HM', 'bin',
                                          'TAppEncoderStatic')  # Adjust if your executable is named differently or located elsewhere
    args.tapp_decoder_path = os.path.join(HM_BASE_DIR, 'HM', 'bin',
                                          'TAppDecoderStatic')  # Adjust if your executable is named differently or located elsewhere

    # Example: Manually setting args for testing if not using a config file
    args.pretrained_weights = "pretrained_models/MyModel/MTL_bitrates/"
    # args.crop = False # Set to True if your data loading uses cropping and ("color_aug", 0, 0)
    args.skip_compression = False  # Add this line to control compression evaluation

    test_online_compression(args)
