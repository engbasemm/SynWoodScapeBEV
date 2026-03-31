import os
import math
import cv2
import time
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
from thop import profile
import torch.nn as nn
import io
import json
import multiprocessing


from torch.utils.data import Dataset
from data_loader.woodscape_loader import WoodScapeRawDataset


from train_utils.detection_utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class, get_contour
from main import collect_args
from models.detection_decoderORG import YoloDecoder
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder
from utils import Tupperware, IoU, semantic_color_encoding
from models.compressAI.priors import ImprovedFPDA , ScaleHyperpriorDecoderAtten, FactorizedPriorDecoderAtten, \
    FactorizedPriorDecoderAttenQtz
from models.compressAI.waseda import Cheng2020AttentionDecoder


FRAME_RATE = 1




class ImageOnlyDataset(Dataset):
    """
    A dataset class that only loads images, suitable for compression and evaluation
    when no detection or semantic annotations are needed.
    """


    def __init__(self, data_path, path_file, is_train, config):
        self.data_path = data_path
        self.path_file = path_file
        self.is_train = is_train
        self.config = config


        self.filenames = self.read_filenames(path_file)
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((config.input_height, config.input_width))


    def read_filenames(self, path_file):
        with open(path_file, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        return filenames


    def _get_image_path(self, filename):
        return os.path.join(self.data_path, filename)


    def __getitem__(self, index):
        outputs = {}
        filename = self.filenames[index]


        # Load color image
        image_path = self._get_image_path(filename)
        original_pil = Image.open(image_path).convert('RGB')
        outputs[("color", 0, 0)] = self.to_tensor(self.resize(original_pil))
        outputs["original_pil_resized"] = self.resize(original_pil)
        outputs["frame_id"] = index
        outputs["image_path"] = image_path


        # For ImageOnlyDataset, explicitly set detection and semantic labels to None
        outputs[("detection_labels", 0)] = None
        outputs[("semantic_labels", 0, 0)] = None


        return outputs


    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.filenames)




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


    collated_data['original_pil_resized'] = []
    collated_data['frame_id'] = []
    collated_data['image_path'] = []


    detection_labels_to_concat = []


    for i, sample in enumerate(batch):
        if ("detection_labels", 0) in sample and sample[("detection_labels", 0)] is not None:
            labels_raw_from_sample = sample[("detection_labels", 0)]
            labels_for_current_sample = labels_raw_from_sample.view(-1, 6)


            if labels_for_current_sample.numel() > 0:
                detection_labels_to_concat.append(labels_for_current_sample)


        if "original_pil_resized" in sample:
            collated_data['original_pil_resized'].append(sample['original_pil_resized'])
        if "frame_id" in sample:
            collated_data['frame_id'].append(sample['frame_id'])
        if "image_path" in sample:
            collated_data['image_path'].append(sample['image_path'])


        for key, value in sample.items():
            if key == ("detection_labels", 0) or \
                    key == "original_pil_resized" or \
                    key == "frame_id" or \
                    key == "image_path":
                continue


            if key not in collated_data:
                collated_data[key] = []
            collated_data[key].append(value)


    if len(detection_labels_to_concat) > 0:
        collated_data["detection_labels"] = torch.cat(detection_labels_to_concat, 0)
    else:
        collated_data["detection_labels"] = torch.empty((0, 6), dtype=torch.float32)


    for key in list(collated_data.keys()):
        if key in ["original_pil_resized", "frame_id", "image_path", "detection_labels"]:
            continue


        value_list = collated_data[key]
        if len(value_list) > 0 and isinstance(value_list[0], torch.Tensor):
            collated_data[key] = torch.stack(value_list)
        elif len(value_list) == 0:
            del collated_data[key]


    return collated_data




def compress_hevc_online(original_image_tensor: torch.Tensor, qp_value: int, hevc_encoder_cfg_main: str,
                         hevc_encoder_cfg_per_sequence: str, tapp_encoder_path: str, tapp_decoder_path: str,
                         output_vis_path=None):
    """
    Compresses an image using HEVC via a simulated HM software pipeline (ffmpeg + TAppEncoder/TAppDecoder).
    The 'qp_value' is used for the HM encoder.
    Note: TAppEncoder/TAppDecoder are command-line tools that primarily operate on files.
    The use of `tempfile.TemporaryDirectory()` here means files are written to a temporary location,
    which on Linux systems is often a RAM-backed filesystem (`/tmp` or `/dev/shm`),
    providing a performance benefit akin to in-memory operations.
    """
    pil_image = tensor_to_pil(original_image_tensor)
    w, h = pil_image.size
    num_pixels = w * h


    yuv_pixel_format = "yuv420p"


    with tempfile.TemporaryDirectory() as tmpdir:
        input_png_path = os.path.join(tmpdir, "input.png")  # Original image saved as PNG
        input_yuv_path = os.path.join(tmpdir, "input.yuv")  # PNG converted to YUV for encoder
        bitstream_hevc_path = os.path.join(tmpdir, "bitstream.bin")  # Compressed bitstream
        decoded_yuv_path = os.path.join(tmpdir, "decoded.yuv")  # YUV reconstructed by decoder
        output_png_path = os.path.join(tmpdir, "output.png")  # Final reconstructed PNG for metrics


        pil_image.save(input_png_path)


        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', input_png_path,
                '-s', f"{w}x{h}",
                '-pix_fmt', yuv_pixel_format,
                '-loglevel', 'error',
                input_yuv_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error converting PNG to YUV: {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': np.nan,
                    'decoding_time': np.nan}
        try:
            # --- HEVC Encoding ---
            start_enc_time = time.time()
            cmd_enc = [
                tapp_encoder_path,
                '-c', hevc_encoder_cfg_main,
                '-c', hevc_encoder_cfg_per_sequence,
                '-i', input_yuv_path,
                '--QP=' + str(qp_value),
                '--ConformanceWindowMode=1',
                '-o', os.devnull,  # Suppress reconstructed YUV from encoder
                '-b', bitstream_hevc_path
            ]
            subprocess.run(cmd_enc, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            encoding_time = time.time() - start_enc_time
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error encoding with TAppEncoder (Is TAppEncoderStatic in PATH and config files accessible?): {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': np.nan,
                    'decoding_time': np.nan}


        if not os.path.exists(bitstream_hevc_path) or os.path.getsize(bitstream_hevc_path) == 0:
            print(f"Warning: HEVC bitstream not created or is empty for QP {qp_value}. Skipping metrics.")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': np.nan,
                    'decoding_time': np.nan}


        bpp = (os.path.getsize(bitstream_hevc_path) * 8) / num_pixels


        # --- HEVC Decoding ---
        try:
            start_dec_time = time.time()
            cmd_dec = [tapp_decoder_path, '-b', bitstream_hevc_path, '-o', decoded_yuv_path]
            subprocess.run(cmd_dec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            decoding_time = time.time() - start_dec_time
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error decoding with TAppDecoder (Is TAppDecoderStatic in PATH?): {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': encoding_time,
                    'decoding_time': np.nan}


        try:
            # Convert decoded YUV back to PNG for metric calculation
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', yuv_pixel_format,
                '-s', f"{w}x{h}",
                '-i', decoded_yuv_path,
                '-loglevel', 'error',
                output_png_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error converting YUV to PNG: {e}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': encoding_time,
                    'decoding_time': decoding_time}


        reconstructed_pil = Image.open(output_png_path).convert('RGB')
        if output_vis_path:
            os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
            reconstructed_pil.save(output_vis_path)
        reconstructed_tensor = transforms.ToTensor()(reconstructed_pil).unsqueeze(0).to(original_image_tensor.device)


        original_tensor_float = original_image_tensor.to(torch.float32).clamp(0.0, 1.0)
        reconstructed_tensor_float = reconstructed_tensor.to(torch.float32).clamp(0.0, 1.0)


        psnr_val = psnr(original_tensor_float, reconstructed_tensor_float)
        ms_ssim_val = ms_ssim(original_tensor_float, reconstructed_tensor_float, data_range=1.0).item()


        return {'BPP': bpp, 'PSNR': psnr_val, 'MS_SSIM': ms_ssim_val, 'MSE': psnr_to_mse(psnr_val),
                'encoding_time': encoding_time, 'decoding_time': decoding_time}




def compress_jpeg2000_online(original_image_tensor: torch.Tensor, quality_level: int, output_vis_path=None):
    """
    Compresses an image using JPEG2000 via FFmpeg, leveraging temporary files
    which can reside in RAM (e.g., /dev/shm on Linux).
    """
    pil_image = tensor_to_pil(original_image_tensor)
    w, h = pil_image.size
    num_pixels = w * h


    quality_to_ffmpeg_compression_level = {
        0: 10,  # Highest quality
        1: 15,
        2: 25,
        3: 50,
        4: 75,
        5: 95   # Lowest quality
    }
    ffmpeg_compression_level = quality_to_ffmpeg_compression_level.get(quality_level, 50)


    with tempfile.TemporaryDirectory() as tmpdir:
        input_png_path = os.path.join(tmpdir, "input.png")
        encoded_j2k_path = os.path.join(tmpdir, "encoded.j2k")
        output_png_path = os.path.join(tmpdir, "output.png")
        try:
            pil_image.save(input_png_path)


            # --- JPEG2000 Encoding ---
            start_enc_time = time.time()
            encode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', input_png_path,
                '-c:v', 'libopenjpeg', '-compression_level', str(ffmpeg_compression_level),
                encoded_j2k_path
            ]
            subprocess.run(encode_cmd, check=True, capture_output=True, text=True)
            encoding_time = time.time() - start_enc_time


            bitstream_size_bytes = os.path.getsize(encoded_j2k_path)
            bpp = (bitstream_size_bytes * 8) / num_pixels


            # --- JPEG2000 Decoding ---
            start_dec_time = time.time()
            decode_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', encoded_j2k_path,
                output_png_path
            ]
            subprocess.run(decode_cmd, check=True, capture_output=True, text=True)
            decoding_time = time.time() - start_dec_time


            reconstructed_pil = Image.open(output_png_path).convert('RGB')
            if output_vis_path:
                os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
                reconstructed_pil.save(output_vis_path)
            reconstructed_tensor = transforms.ToTensor()(reconstructed_pil).unsqueeze(0).to(original_image_tensor.device)


            original_tensor_float = original_image_tensor.to(torch.float32).clamp(0.0, 1.0)
            reconstructed_tensor_float = reconstructed_tensor.to(torch.float32).clamp(0.0, 1.0)


            psnr_val = psnr(original_tensor_float, reconstructed_tensor_float)
            ms_ssim_val = ms_ssim(original_tensor_float, reconstructed_tensor_float, data_range=1.0).item()


            return {
                'BPP': bpp,
                'PSNR': psnr_val,
                'MS_SSIM': ms_ssim_val,
                'MSE': psnr_to_mse(psnr_val),
                'encoding_time': encoding_time,
                'decoding_time': decoding_time
            }


        except subprocess.CalledProcessError as e:
            print(f"Error during JPEG2000 processing: {e.stderr}")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': np.nan,
                    'decoding_time': np.nan}
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please ensure it's installed and in your PATH.")
            return {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan, 'encoding_time': np.nan,
                    'decoding_time': np.nan}




def process_image_codecs_worker(task_args):
    """
    A worker function for multiprocessing. Compresses a single image with all
    traditional codec configurations (HEVC and JPEG2000).
    """
    (image_key, original_image_tensor, hevc_qps, jpeg2000_qs,
     hevc_encoder_cfg_main, hevc_encoder_cfg_per_sequence, tapp_encoder_path, tapp_decoder_path,
     base_output_dir, current_dataset_name) = task_args


    # The tensor is expected to be on the CPU.
    image_metrics = {'image_name': image_key}
    output_name = os.path.splitext(image_key)[0]


    # Create a common folder for original images for the dataset
    originals_output_dir = os.path.join(base_output_dir, current_dataset_name, "originals")
    os.makedirs(originals_output_dir, exist_ok=True)
    original_output_path = os.path.join(originals_output_dir, f"{output_name}_original.png")
    if not os.path.exists(original_output_path):
        tensor_to_pil(original_image_tensor).save(original_output_path)


    # HEVC Compression
    for name, qp in hevc_qps.items():
        try:
            eval_output_dir_codec = os.path.join(base_output_dir, current_dataset_name, name)
            vis_output_dir = os.path.join(eval_output_dir_codec, "visualizations")
            safe_codec_name = name.replace(' ', '_')
            output_vis_path = os.path.join(vis_output_dir, f"{output_name}_rsz_rcon_{safe_codec_name}.png")


            image_metrics[name] = compress_hevc_online(
                original_image_tensor, qp,
                hevc_encoder_cfg_main, hevc_encoder_cfg_per_sequence,
                tapp_encoder_path, tapp_decoder_path,
                output_vis_path=output_vis_path)
        except Exception as e:
            print(f"ERROR in worker: {name} failed for {image_key}: {e}")
            image_metrics[name] = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan,
                                   'encoding_time': np.nan, 'decoding_time': np.nan}


    # JPEG2000 Compression
    for name, q in jpeg2000_qs.items():
        try:
            eval_output_dir_codec = os.path.join(base_output_dir, current_dataset_name, name)
            vis_output_dir = os.path.join(eval_output_dir_codec, "visualizations")
            safe_codec_name = name.replace(' ', '_')
            output_vis_path = os.path.join(vis_output_dir, f"{output_name}_rsz_rcon_{safe_codec_name}.png")


            image_metrics[name] = compress_jpeg2000_online(original_image_tensor, q, output_vis_path=output_vis_path)
        except Exception as e:
            print(f"ERROR in worker: {name} failed for {image_key}: {e}")
            image_metrics[name] = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan,
                                   'encoding_time': np.nan, 'decoding_time': np.nan}


    return image_key, image_metrics




dl_model_configs = [
    {
        "name": "RES50_Baseline_no_compression",
        "compression_decoder_type": "not_applicable",
        "pretrained_path_segment": "res50_no_compress_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
        "batch_suffix": "99/97500", "resnet_type": "RES50",
        "fixed_bpp": None
    },
    # Cheng Attention Models
    {
        "name": "Model_A_ChengAtten",
        "compression_decoder_type": "chengAttenDecoder",
        "pretrained_path_segment": "res50_cheng_attn0.08_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
        #"batch_suffix": "107/105000", "resnet_type": "RES50",
        "batch_suffix": "103/101100", "resnet_type": "RES50",
        "fixed_bpp": None
    },
    {
        "name": "Model_B_ChengAtten",
        "compression_decoder_type": "chengAttenDecoder",
        "pretrained_path_segment": "res50_chengAttenDecoder.05_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
        "batch_suffix": "105/103800", "resnet_type": "RES50",
        "fixed_bpp": None
    },
    {
        "name": "Model_C_ChengAtten",
        "compression_decoder_type": "chengAttenDecoder",
        "pretrained_path_segment": "res50_cheng_attn0.02_FR_2Classes_yolo12_dbg128_conf30_20_mtl_regul_0.1_anchort5",
        "batch_suffix": "124/122100", "resnet_type": "RES50",
        "fixed_bpp": None
    },
    {
        "name": "Model_D_ChengAtten",
        "compression_decoder_type": "chengAttenDecoder",
        "pretrained_path_segment": "res50_chengAttenDecoder.01_FR_2Classes_yolo3_dbg128_conf30_20_mtl_regul_0.1_anchort5",
        "batch_suffix": "114/111900", "resnet_type": "RES50",
        "fixed_bpp": None
    }
]




def _save_detection_visualization(original_image_tensor, outputs_nms, detection_color_encoding, base_output_dir,
                                  output_name_base, video_writer=None):
    """Saves the detection results as an image and optionally writes to a video."""
    vis_output_dir = os.path.join(base_output_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    output_path = os.path.join(vis_output_dir, f"{output_name_base}_detection.png")


    img_d_np = original_image_tensor.cpu().detach().numpy().squeeze(0)
    img_d_np = np.transpose(img_d_np, (1, 2, 0))
    img_cpu_display = (img_d_np * 255).astype(np.uint8).copy()


    if outputs_nms is not None and outputs_nms[0] is not None:
        outputs = torch.cat(outputs_nms, dim=0)
        for box in outputs:
            cls_pred = int(box[6])
            class_color = (detection_color_encoding[cls_pred]).tolist()


            x1, y1, x2, y2 = box[0:4].cpu().numpy().tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            cv2.rectangle(img_cpu_display, (x1, y1), (x2, y2), class_color, thickness=2)


            conf = box[4]
            cv2.putText(img_cpu_display, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)


    if video_writer:
        video_writer.write(cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))


    cv2.imwrite(output_path, cv2.cvtColor(img_cpu_display, cv2.COLOR_RGB2BGR))




def _save_semantic_visualization(original_image_tensor, semantic_pred, semantic_color_coding, base_output_dir,
                                 output_name_base, feed_width, feed_height):
    """Saves the semantic segmentation results as a raw .npy file and a blended .png image."""
    vis_output_dir = os.path.join(base_output_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)


    # Save raw predictions as .npy
    name_dest_npy = os.path.join(vis_output_dir, f"{output_name_base}_semantic.npy")
    _, predictions_np = torch.max(semantic_pred.data.squeeze(0), 0)
    predictions_np = predictions_np.byte().cpu().detach().numpy()
    np.save(name_dest_npy, predictions_np)


    # Create and save blended visualization
    alpha = 0.5
    color_semantic_img_np = np.array(transforms.ToPILImage()(original_image_tensor.cpu().squeeze(0)))
    not_background = predictions_np != 0
    blended_image = color_semantic_img_np.copy()
    blended_image[not_background, ...] = (color_semantic_img_np[not_background, ...] * (1 - alpha) +
                                          semantic_color_coding[predictions_np[not_background]].astype(
                                              color_semantic_img_np.dtype) * alpha)
    semantic_color_mapped_pil = Image.fromarray(blended_image)


    name_dest_im_semantic = os.path.join(vis_output_dir, f"{output_name_base}_semantic.png")
    pil_input_image = G.to_pil_image(original_image_tensor.cpu().squeeze(0))
    rgb_color_pred_concat = Image.new('RGB', (feed_width, feed_height + feed_height))
    rgb_color_pred_concat.paste(pil_input_image, (0, 0))
    rgb_color_pred_concat.paste(semantic_color_mapped_pil, (0, pil_input_image.height))
    rgb_color_pred_concat.save(name_dest_im_semantic)




@torch.no_grad()
def test_online_compression(args, current_dataset_name, current_data_path, current_val_file, has_semantic_annotations,
                            has_detection_annotations):
    """Function to evaluate DL models and traditional codecs with on-the-fly compression."""
    feed_height = int(args.input_height)
    feed_width = int(args.input_width)
    img_size = [feed_width, feed_height]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for dataset: {current_dataset_name}")


    if current_dataset_name.lower() == "woodscape":
        val_dataset = WoodScapeRawDataset(data_path=current_data_path, path_file=current_val_file, is_train=False,
                                          config=args)
    elif current_dataset_name.lower() == "fisheye8k_image_only":
        val_dataset = ImageOnlyDataset(data_path=current_data_path, path_file=current_val_file, is_train=False,
                                       config=args)
    else:
        raise ValueError(f"Unsupported dataset type for loading: {current_dataset_name}")


    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True,
                            drop_last=True, collate_fn=custom_collate_fn)
    all_val_image_paths = [line.rstrip('\n') for line in open(current_val_file)]


    per_image_detailed_metrics = {}
    all_dl_model_overall_metrics = {}
    overall_codec_metrics = {}


    hevc_qps = {'HM HEVC QP31': 31, 'HM HEVC QP33': 33, 'HM HEVC QP36': 36, 'HM HEVC QP39': 39, 'HM HEVC QP42': 42, 'HM HEVC QP45': 45}
    jpeg2000_qs = {'JPEG2000 Q0': 0, 'JPEG2000 Q1': 1, 'JPEG2000 Q2': 2, 'JPEG2000 Q3': 3, 'JPEG2000 Q4': 4, 'JPEG2000 Q5': 5}


    semantic_color_coding = semantic_color_encoding(args)
    detection_color_encoding = color_encoding_woodscape_detection()


    print(f"Starting evaluation for dataset: {current_dataset_name}...")


    if not args.skip_compression:
        if args.enable_multiprocessing:
            print(f"Evaluating traditional codecs in parallel for {current_dataset_name}...")
            val_loader_codecs = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2,
                                           pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)


            tasks = []
            print("  -> Preparing tasks for parallel processing...")
            for batch_i, inputs in enumerate(tqdm(val_loader_codecs, desc="  Preparing Codec Tasks")):
                current_image_path = all_val_image_paths[batch_i]
                image_key = os.path.basename(current_image_path)
                true_original_image_tensor = inputs[("color", 0, 0)].cpu()


                task_args = (
                    image_key, true_original_image_tensor, hevc_qps, jpeg2000_qs,
                    args.hevc_encoder_cfg_main, args.hevc_encoder_cfg_per_sequence,
                    args.tapp_encoder_path, args.tapp_decoder_path,
                    args.output_directory, current_dataset_name)
                tasks.append(task_args)


            num_codec_workers = max(1, (os.cpu_count() or 1) // 4)
            print(f"  -> Spawning {num_codec_workers} processes for codec evaluation...")


            with multiprocessing.Pool(processes=num_codec_workers) as pool:
                results = list(tqdm(pool.imap_unordered(process_image_codecs_worker, tasks),
                                    total=len(tasks), desc="  Processing Traditional Codecs"))


            for image_key, image_metrics in results:
                per_image_detailed_metrics[image_key] = image_metrics
        else:
            print(f"Evaluating traditional codecs sequentially for {current_dataset_name}...")
            val_loader_codecs = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                           pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)


            for batch_i, inputs in enumerate(tqdm(val_loader_codecs, desc="  Processing Traditional Codecs")):
                current_image_path = all_val_image_paths[batch_i]
                image_key = os.path.basename(current_image_path)
                true_original_image_tensor = inputs[("color", 0, 0)].to(device)


                if image_key not in per_image_detailed_metrics:
                    per_image_detailed_metrics[image_key] = {'image_name': image_key}


                # Create a common folder for original images for the dataset
                originals_output_dir = os.path.join(args.output_directory, current_dataset_name, "originals")
                os.makedirs(originals_output_dir, exist_ok=True)
                original_output_path = os.path.join(originals_output_dir, f"{output_name}_original.png")
                if not os.path.exists(original_output_path):
                    tensor_to_pil(true_original_image_tensor).save(original_output_path)


                output_name = os.path.splitext(image_key)[0]


                for name, qp in hevc_qps.items():
                    eval_output_dir_codec = os.path.join(args.output_directory, current_dataset_name, name)
                    vis_output_dir = os.path.join(eval_output_dir_codec, "visualizations")
                    safe_codec_name = name.replace(' ', '_')
                    output_vis_path = os.path.join(vis_output_dir, f"{output_name}_rsz_rcon_{safe_codec_name}.png")


                    per_image_detailed_metrics[image_key][name] = compress_hevc_online(
                        true_original_image_tensor, qp, args.hevc_encoder_cfg_main,
                        args.hevc_encoder_cfg_per_sequence, args.tapp_encoder_path, args.tapp_decoder_path,
                        output_vis_path=output_vis_path)


                for name, q in jpeg2000_qs.items():
                    eval_output_dir_codec = os.path.join(args.output_directory, current_dataset_name, name)
                    vis_output_dir = os.path.join(eval_output_dir_codec, "visualizations")
                    safe_codec_name = name.replace(' ', '_')
                    output_vis_path = os.path.join(vis_output_dir, f"{output_name}_rsz_rcon_{safe_codec_name}.png")
                    per_image_detailed_metrics[image_key][name] = compress_jpeg2000_online(
                        true_original_image_tensor, q, output_vis_path=output_vis_path)


        for codec_type_dict in [hevc_qps, jpeg2000_qs]:
            for name in codec_type_dict.keys():
                bpps = []
                psnrs = []
                msssims = []
                mses = []
                encoding_times = []
                decoding_times = []
                for image_key in per_image_detailed_metrics:
                    metrics = per_image_detailed_metrics[image_key].get(name)
                    if metrics and not np.isnan(metrics['BPP']):
                        bpps.append(metrics['BPP'])
                        psnrs.append(metrics['PSNR'])
                        msssims.append(metrics['MS_SSIM'])
                        mses.append(metrics['MSE'])
                        if 'encoding_time' in metrics:
                            encoding_times.append(metrics['encoding_time'])
                        if 'decoding_time' in metrics:
                            decoding_times.append(metrics['decoding_time'])


                overall_codec_metrics[name] = {
                    "BPP": np.nanmean(bpps) if bpps else np.nan,
                    "PSNR": np.nanmean(psnrs) if psnrs else np.nan,
                    "MS_SSIM": np.nanmean(msssims) if msssims else np.nan,
                    "MSE": np.nanmean(mses) if mses else np.nan,
                    "Encoding Time (s)": np.nanmean(encoding_times) if encoding_times else np.nan,
                    "Decoding Time (s)": np.nanmean(decoding_times) if decoding_times else np.nan,
                }
        print(f"\nFinished evaluation for Traditional Codecs on {current_dataset_name}.")


    for dl_config in dl_model_configs:
        model_name = dl_config["name"]
        print(f"\n{'=' * 80}\nEvaluating DL Model: {model_name} for dataset: {current_dataset_name}\n{'=' * 80}")


        run_bitrates = os.path.join(args.pretrained_weights, dl_config["pretrained_path_segment"], "models",
                                    f"weights_{dl_config['batch_suffix']}")
        eval_output_dir_model = os.path.join(args.output_directory, current_dataset_name,
                                             model_name)
        os.makedirs(eval_output_dir_model, exist_ok=True)


        # --- 1. Instantiate all models ---
        network_layers = int(dl_config["resnet_type"].replace("RES", ""))
        encoder = ResnetEncoder(num_layers=network_layers, pretrained=False).to(device)


        compression_decoder = None
        decoder_type = dl_config["compression_decoder_type"]
        if decoder_type == "chengAttenDecoder":
            compression_decoder = Cheng2020AttentionDecoder(64).to(device)
        elif decoder_type == "factorizedPriorDecoder":
            compression_decoder = FactorizedPriorDecoderAtten(64, 1024 if network_layers == 50 else 256).to(device)
        elif decoder_type == "factorizedPriorDecoderQuantz":
            compression_decoder = FactorizedPriorDecoderAttenQtz(64, 1024 if network_layers == 50 else 256).to(device)
        elif decoder_type == "ImprovedFPDA":
            N = dl_config.get("N", 256)  # Default to a sensible value
            M = dl_config.get("M", 256)  # Default to a sensible value
            use_refinement = dl_config.get("use_refinement_head", False)
            # Use features from ResNet layer3 (index 3), which has 1024 channels for ResNet-50
            in_channels = encoder.num_ch_enc[3]
            compression_decoder = ImprovedFPDA(N=N, M=M, out_height=feed_height, out_width=feed_width,
                                               in_channels_encoder=in_channels,
                                               use_refinement_head=use_refinement).to(device)


        decoder = None  # This is the detection decoder
        if has_detection_annotations:
            decoder = YoloDecoder(encoder.num_ch_enc, args).to(device)


        semantic_decoder = None
        if has_semantic_annotations:
            semantic_decoder = SemanticDecoder(encoder.num_ch_enc, n_classes=args.semantic_num_classes).to(device)


        # --- 2. Load weights ---
        loaded_dict_enc = torch.load(os.path.join(run_bitrates, "encoder.pth"), map_location=device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc, strict=False)


        if compression_decoder is not None:
            compression_decoder.load_state_dict(torch.load(os.path.join(run_bitrates, "decoder.pth"), map_location=device))


        if decoder is not None:
            decoder.load_state_dict(torch.load(os.path.join(run_bitrates, "detection.pth"), map_location=device))


        if semantic_decoder is not None:
            semantic_decoder.load_state_dict(torch.load(os.path.join(run_bitrates, "semantic.pth"), map_location=device))


        # --- 3. Set to eval mode ---
        encoder.eval()
        for m_module in encoder.modules():
            if isinstance(m_module, nn.BatchNorm2d):
                m_module.track_running_stats = False
                m_module.running_mean = None
                m_module.running_var = None
        if compression_decoder is not None:
            compression_decoder.eval()
        if decoder is not None:
            decoder.eval()
        if semantic_decoder is not None:
            semantic_decoder.eval()


        # --- 4. Hardware Resource Calculation ---
        # Model Size (from files)
        total_size_bytes = 0
        encoder_path = os.path.join(run_bitrates, "encoder.pth")
        if os.path.exists(encoder_path):
            total_size_bytes += os.path.getsize(encoder_path)


        decoder_path = os.path.join(run_bitrates, "decoder.pth")
        if dl_config["compression_decoder_type"] != "not_applicable" and os.path.exists(decoder_path):
            total_size_bytes += os.path.getsize(decoder_path)
        total_size_mb = total_size_bytes / (1024 * 1024)


        # FLOPs and Parameters (using thop)
        dummy_input = torch.randn(1, 3, feed_height, feed_width).to(device)
        enc_flops, enc_params = profile(encoder, inputs=(dummy_input,), verbose=False)


        comp_flops, comp_params = 0, 0
        if compression_decoder is not None:
            with torch.no_grad():
                dummy_features = encoder(dummy_input)
            comp_flops, comp_params = profile(compression_decoder, inputs=(dummy_features,), verbose=False)
        total_flops_g = (enc_flops + comp_flops) / 1e9
        total_params_m = (enc_params + comp_params) / 1e6


        labels_overall_dl, sample_metrics_overall_dl, comp_metrics_list, encoding_times_dl, decoding_times_dl = [], [], [], [], []
        semantic_metric = IoU(args.semantic_num_classes, args.dataset,
                              ignore_index=None) if has_semantic_annotations else None


        video = None
        if has_detection_annotations:
            video = cv2.VideoWriter(os.path.join(eval_output_dir_model, f"{model_name}_detection_video.mp4"),
                                    cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (feed_width, feed_height))


        for batch_i, inputs in enumerate(
                tqdm(val_loader, desc=f"Processing images for {model_name} on {current_dataset_name}")):
            for key, ipt in inputs.items():
                if isinstance(ipt, torch.Tensor):
                    inputs[key] = ipt.to(device)
            current_image_path = all_val_image_paths[batch_i]
            image_key = os.path.basename(current_image_path)


            true_original_image_tensor = inputs[("color", 0, 0)]


            bpp = np.nan  # Default BPP
            if dl_config["compression_decoder_type"] != "not_applicable":
                # Time encoding
                torch.cuda.synchronize(device)
                start_time = time.time()
                features_orig = encoder(true_original_image_tensor)
                torch.cuda.synchronize(device)
                encoding_times_dl.append(time.time() - start_time)


                # Time decoding
                torch.cuda.synchronize(device)
                start_time = time.time()
                out_comp_orig = compression_decoder(features_orig)
                x_hat_orig = out_comp_orig["x_hat"]
                torch.cuda.synchronize(device)
                decoding_times_dl.append(time.time() - start_time)


                num_pixels_full = true_original_image_tensor.shape[2] * true_original_image_tensor.shape[3]


                # For models with fixed BPP (e.g., Factorized models with quantization), use the config value.
                if dl_config.get("fixed_bpp") is not None:
                    bpp = dl_config["fixed_bpp"]
                # For models that calculate BPP via entropy coding (e.g., ChengAtten), calculate it.
                elif "likelihoods" in out_comp_orig and out_comp_orig["likelihoods"]:
                    bpp = sum((torch.log(lk).sum() / (-math.log(2) * num_pixels_full)) for lk in
                              out_comp_orig["likelihoods"].values()).item()
                elif "num_bits" in out_comp_orig:
                    bpp = out_comp_orig["num_bits"].item() / num_pixels_full


                psnr_dl, msssim_dl = psnr(true_original_image_tensor, x_hat_orig), ms_ssim(true_original_image_tensor,
                                                                                           x_hat_orig,
                                                                                           data_range=1.0).item()
                comp_metrics = {'BPP': bpp, 'PSNR': psnr_dl, 'MS_SSIM': msssim_dl,
                                'MSE': psnr_to_mse(psnr_dl)}
                input_image_for_downstream_tasks = x_hat_orig
            else:
                # Baseline model (no compression decoder)
                torch.cuda.synchronize(device)
                start_time = time.time()
                # The "encoding" for baseline is just the ResNet forward pass
                features_for_downstream_tasks = encoder(true_original_image_tensor)
                torch.cuda.synchronize(device)
                encoding_times_dl.append(time.time() - start_time)
                decoding_times_dl.append(0)  # No decoding step


                comp_metrics = {'BPP': np.nan, 'PSNR': np.nan, 'MS_SSIM': np.nan, 'MSE': np.nan}
                input_image_for_downstream_tasks = true_original_image_tensor
                features_for_downstream = features_for_downstream_tasks


            comp_metrics_list.append(comp_metrics)


            if image_key not in per_image_detailed_metrics:
                per_image_detailed_metrics[image_key] = {'image_name': image_key}
            per_image_detailed_metrics[image_key][model_name] = comp_metrics


            # For compressed models, we need to re-encode the reconstructed image
            # to get features for the downstream tasks. For baseline, we already have them.
            if dl_config["compression_decoder_type"] != "not_applicable":
                with torch.no_grad():
                    features_for_downstream = encoder(input_image_for_downstream_tasks)


            features = features_for_downstream


            if has_semantic_annotations and semantic_decoder is not None:
                semantic_pred = semantic_decoder(features)[("semantic", 0)]
            else:
                semantic_pred = None


            yolo_outputs = None
            if has_detection_annotations and decoder is not None:
                yolo_outputs = decoder(features, img_dim=img_size)["yolo_outputs"]


            if has_detection_annotations:
                targets = inputs["detection_labels"].cpu()


                if targets.numel() > 0 and yolo_outputs is not None:
                    labels_overall_dl.extend(targets[:, 1].tolist())


                    targets[:, 2:6] = xywh2xyxy(targets[:, 2:6])
                    targets[:, 2] *= img_size[0]
                    targets[:, 3] *= img_size[1]
                    targets[:, 4] *= img_size[0]
                    targets[:, 5] *= img_size[1]


                    outputs_nms = non_max_suppression(yolo_outputs,
                                                      conf_thres=args.detection_conf_thres,
                                                      nms_thres=args.detection_nms_thres)


                    sample_metrics_overall_dl.extend(
                        get_batch_statistics(outputs_nms, targets, iou_threshold=0.5, args=args))
                else:
                    outputs_nms = None


                # Refactored visualization call
                output_name_base = os.path.splitext(os.path.basename(current_image_path))[0]
                safe_model_name = model_name.replace(' ', '_')
                _save_detection_visualization(inputs[("color", 0, 0)], outputs_nms, detection_color_encoding,
                                              eval_output_dir_model, f"{output_name_base}_{safe_model_name}", video)
            else:
                pass


            if has_semantic_annotations and semantic_metric is not None and semantic_pred is not None:
                _, predictions_semantic = torch.max(semantic_pred.data, 1)
                semantic_metric.add(predictions_semantic, inputs[("semantic_labels", 0, 0)])


                # Refactored visualization call
                output_name_base = os.path.splitext(os.path.basename(current_image_path))[0]
                safe_model_name = model_name.replace(' ', '_')
                _save_semantic_visualization(inputs[("color", 0, 0)], semantic_pred, semantic_color_coding,
                                             eval_output_dir_model, f"{output_name_base}_{safe_model_name}", feed_width, feed_height)
            else:
                pass


            output_name = os.path.splitext(os.path.basename(current_image_path))[0]
            eval_output_dir_model_vis = os.path.join(eval_output_dir_model, "visualizations")
            os.makedirs(eval_output_dir_model_vis, exist_ok=True)


            # Save original image to common folder
            originals_output_dir = os.path.join(args.output_directory, current_dataset_name, "originals")
            os.makedirs(originals_output_dir, exist_ok=True)
            original_output_path = os.path.join(originals_output_dir, f"{output_name}_original.png")
            if not os.path.exists(original_output_path):
                utils.save_image(true_original_image_tensor.data, original_output_path, normalize=True)


            safe_model_name = model_name.replace(' ', '_')
            name_dest_im_reconstructed = os.path.join(eval_output_dir_model_vis, f"{output_name}_rsz_rcon_{safe_model_name}.png")
            utils.save_image(input_image_for_downstream_tasks.data, name_dest_im_reconstructed, normalize=True)


            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        if video:
            video.release()


        overall_map_dl = np.nan
        if has_detection_annotations:
            true_pos, pred_scores, pred_labels = (np.concatenate(x, 0) for x in
                                                  list(
                                                      zip(*sample_metrics_overall_dl))) if sample_metrics_overall_dl else (
                np.array([]), np.array([]), np.array([]))


            overall_ap_dl_np = np.array([])
            if len(labels_overall_dl) > 0 and true_pos.size > 0:
                _, _, overall_ap_dl_np, _, _ = ap_per_class(
                    true_pos, pred_scores, pred_labels, np.array(labels_overall_dl)
                )
            overall_map_dl = np.nanmean(overall_ap_dl_np) if overall_ap_dl_np.size > 0 else 0.0


        overall_miou_dl = np.nan
        if has_semantic_annotations and semantic_metric is not None:
            _, overall_miou_dl = semantic_metric.value()


        if not args.skip_compression:
            avg_dl_comp_bpp = np.nanmean([d['BPP'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_psnr = np.nanmean([d['PSNR'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_msssim = np.nanmean([d['MS_SSIM'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
            avg_dl_comp_mse = np.nanmean([d['MSE'] for d in comp_metrics_list]) if comp_metrics_list else np.nan
        else:
            avg_dl_comp_bpp, avg_dl_comp_psnr, avg_dl_comp_msssim, avg_dl_comp_mse = np.nan, np.nan, np.nan, np.nan


        avg_encoding_time_dl = np.nanmean(encoding_times_dl) if encoding_times_dl else np.nan
        avg_decoding_time_dl = np.nanmean(decoding_times_dl) if decoding_times_dl else np.nan


        all_dl_model_overall_metrics[model_name] = {
            "mAP": overall_map_dl,
            "mIoU": overall_miou_dl,
            "BPP": avg_dl_comp_bpp,
            "PSNR": avg_dl_comp_psnr,
            "MS_SSIM": avg_dl_comp_msssim,
            "MSE": avg_dl_comp_mse,
            "Model Size (MB)": total_size_mb,
            "Params (M)": total_params_m,
            "GFLOPs": total_flops_g,
            "Encoding Time (s)": avg_encoding_time_dl,
            "Decoding Time (s)": avg_decoding_time_dl
        }
        print(
            f"Finished evaluation for DL Model: {model_name} on {current_dataset_name}. mAP: {all_dl_model_overall_metrics[model_name]['mAP']:.4f}, mIoU: {all_dl_model_overall_metrics[model_name]['mIoU']:.4f}")


    return per_image_detailed_metrics, all_dl_model_overall_metrics, overall_codec_metrics




if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)


    HM_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


    args.hevc_encoder_cfg_main = os.path.join(HM_BASE_DIR, 'HM', 'cfg', 'encoder_intra_main.cfg')
    args.hevc_encoder_cfg_per_sequence = os.path.join(HM_BASE_DIR, 'HM', 'cfg', 'per-sequence',
                                                      'Traffic_Fisheye8K.cfg')


    args.tapp_encoder_path = os.path.join(HM_BASE_DIR, 'HM', 'bin',
                                          'TAppEncoderStatic')
    args.tapp_decoder_path = os.path.join(HM_BASE_DIR, 'HM', 'bin',
                                          'TAppDecoderStatic')


    args.pretrained_weights = "pretrained_models/MyModel/MTL_bitrates/"
    args.skip_compression = False


    datasets_to_evaluate = [
        {
            "name": "Fisheye8k_Image_Only",
            "data_path": args.fishey8k_dataset_dir,
            "val_file": args.fishey8k_val_file,
            "has_semantic": False,
            "has_detection": False
        },
        {
            "name": "Woodscape",
            "data_path": args.dataset_dir,
            "val_file": args.val_file,
            "has_semantic": True,
            "has_detection": True
        }
    ]


    all_datasets_per_image_detailed_metrics = {}
    all_datasets_overall_dl_model_metrics = {}
    all_datasets_overall_traditional_codec_metrics = {}


    for dataset_config in datasets_to_evaluate:
        dataset_name = dataset_config["name"]
        data_path = dataset_config["data_path"]
        val_file = dataset_config["val_file"]
        has_semantic = dataset_config["has_semantic"]
        has_detection = dataset_config["has_detection"]


        print(f"\n\n{'*' * 100}\nStarting evaluation for dataset: {dataset_name}\n{'*' * 100}")


        per_image_metrics, overall_dl_metrics, overall_codec_metrics = test_online_compression(
            args, dataset_name, data_path, val_file, has_semantic, has_detection
        )


        all_datasets_per_image_detailed_metrics[dataset_name] = per_image_metrics
        all_datasets_overall_dl_model_metrics[dataset_name] = overall_dl_metrics
        all_datasets_overall_traditional_codec_metrics[dataset_name] = overall_codec_metrics


    print("\n\n" + "=" * 120)
    print(f"{'Consolidated Overall Evaluation Summary Across All Datasets':^120}")
    print("=" * 120)


    for dataset_name, overall_metrics in all_datasets_overall_dl_model_metrics.items():
        print(f"\n--- DL Model Results for Dataset: {dataset_name} ---")
        header = (f"{'Model Name':<25} | {'BPP':<8} | {'PSNR':<8} | {'MS_SSIM':<10} | {'mAP':<8} | {'mIoU':<8} | "
                  f"{'Size(MB)':<8} | {'Params(M)':<10} | {'GFLOPs':<8} | {'Enc Time(s)':<12} | {'Dec Time(s)':<12}")
        print(header)
        print("-" * len(header))
        for model_name, metrics in overall_metrics.items():
            resnet_type = "N/A"
            for config_item in dl_model_configs:
                if config_item["name"] == model_name:
                    resnet_type = config_item.get("resnet_type", "N/A")
                    break


            print(f"{model_name:<25} | "
                  f"{metrics.get('BPP', np.nan):<8.3f} | "
                  f"{metrics.get('PSNR', np.nan):<8.2f} | "
                  f"{metrics.get('MS_SSIM', np.nan):<10.5f} | "
                  f"{metrics.get('mAP', np.nan):<8.4f} | "
                  f"{metrics.get('mIoU', np.nan):<8.4f} | "
                  f"{metrics.get('Model Size (MB)', np.nan):<8.2f} | "
                  f"{metrics.get('Params (M)', np.nan):<10.2f} | "
                  f"{metrics.get('GFLOPs', np.nan):<8.2f} | "
                  f"{metrics.get('Encoding Time (s)', np.nan):<12.4f} | "
                  f"{metrics.get('Decoding Time (s)', np.nan):<12.4f}")
        print("-" * len(header))


    for dataset_name, overall_metrics in all_datasets_overall_traditional_codec_metrics.items():
        print(f"\n--- Traditional Codec Results for Dataset: {dataset_name} ---")
        header = (f"{'Codec Name':<25} | {'BPP':<10} | {'PSNR':<10} | {'MS_SSIM':<12} | "
                  f"{'Enc Time(s)':<12} | {'Dec Time(s)':<12}")
        print(header)
        print("-" * len(header))
        for codec_name, metrics in overall_metrics.items():
            print(f"{codec_name:<25} | "
                  f"{metrics.get('BPP', np.nan):<10.3f} | "
                  f"{metrics.get('PSNR', np.nan):<10.2f} | "
                  f"{metrics.get('MS_SSIM', np.nan):<12.5f} | "
                  f"{metrics.get('Encoding Time (s)', np.nan):<12.4f} | "
                  f"{metrics.get('Decoding Time (s)', np.nan):<12.4f}")
        print("-" * len(header))


    print("=" * 120)


    html_data_output = {
        "datasets": {}
    }


    for dataset_name in all_datasets_overall_dl_model_metrics.keys():
        dl_models_for_dataset = []
        for model_name, metrics in all_datasets_overall_dl_model_metrics[dataset_name].items():
            resnet_type = "N/A"
            for config_item in dl_model_configs:
                if config_item["name"] == model_name:
                    resnet_type = config_item.get("resnet_type", "N/A")
                    break


            # Clean up metrics for JSON serialization
            cleaned_metrics = {k: (v if not np.isnan(v) else None) for k, v in metrics.items()}
            cleaned_metrics["Model Name"] = model_name
            cleaned_metrics["ResNet Type"] = resnet_type
            dl_models_for_dataset.append(cleaned_metrics)


        traditional_codecs_for_dataset = []
        for codec_name, metrics in all_datasets_overall_traditional_codec_metrics[dataset_name].items():
            cleaned_metrics = {k: (v if not np.isnan(v) else None) for k, v in metrics.items()}
            cleaned_metrics["Codec"] = codec_name
            traditional_codecs_for_dataset.append(cleaned_metrics)


        html_data_output["datasets"][dataset_name] = {
            "dl_models": dl_models_for_dataset,
            "traditional_codecs": traditional_codecs_for_dataset
        }


    print("\n<---PLOT_DATA_START--->")
    print(json.dumps(html_data_output, indent=4))
    print("<---PLOT_DATA_END--->")


    print(f"=> Done!")



