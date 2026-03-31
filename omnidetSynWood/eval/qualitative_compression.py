"""
Qualitative and Quantitiative compression

# usage: ./qualitative_compression.py --config data/params.yaml

# author: Basem barakat <basem.barakat@valeo.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import os
import math
import random

import cv2
import yaml
from PIL import Image
from torchvision import transforms, utils
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchmetrics import PeakSignalNoiseRatio
#from eval.qualitative_semantic import pre_image_op
import torch.nn as nn
from main import collect_args
from models.detection_decoder import YoloDecoder
from models.resnet import ResnetEncoder
from train_utils.detection_utils import *
from utils import Tupperware
from models.compressAI.priors import ScaleHyperpriorDecoderAtten ,FactorizedPriorDecoder,ScaleHyperpriorOrg,ScaleHyperpriorDecoder , ScaleHyperpriorNoEntropy , ScaleHyperprior , FactorizedPriorDecoderAtten
from models.compressAI.waseda import Cheng2020Anchor , Cheng2020Attention , Cheng2020AttentionDecoder
import torchvision.transforms as T

FRAME_RATE = 1


def pre_image_op(args, index, frame_index, cam_side ):
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

    return get_image(args, index, cropped_coords, frame_index, cam_side )
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def get_image(args, index, cropped_coords, frame_index, cam_side ):
    recording_folder = args.rgb_images if index == 0 else "previous_images"
    file = f"{frame_index}_{cam_side}.png" if index == 0 else f"{frame_index}_{cam_side}_prev.png"
    path = os.path.join(args.dataset_dir, recording_folder, file)

    path_q37 = os.path.join(args.dataset_dir, "rgb_images_Q37", file)
    path_q32 = os.path.join(args.dataset_dir, "rgb_images_Q32", file)
    path_q42 = os.path.join(args.dataset_dir, "rgb_images_JPEG2000_Q3", file)
    image = Image.open(path).convert('RGB')
    image_q37 = Image.open(path_q37).convert('RGB')
    image_q32 = Image.open(path_q32).convert('RGB')
    image_q42 = Image.open(path_q42).convert('RGB')

    image_fisheye8k = image

    if args.fishey8k_dataset_enable:

        file = f"camera1_A_{int(get_image.fisheye8K_couter )}.png"
        fishey8k_dataset_path = os.path.join(os.getcwd(), args.fishey8k_dataset_dir,file)
        #fishey8k_dataset_path = os.path.join(args.dataset_dir, args.fishey8k_dataset_dir, file)
        image_fisheye8k = Image.open(fishey8k_dataset_path).convert('RGB')
        get_image.fisheye8K_couter  = get_image.fisheye8K_couter  + 1

    if args.crop:
        return image.crop(cropped_coords) , image_q32.crop(cropped_coords) , image_q37.crop(cropped_coords) , image_q42.crop(cropped_coords) , image_fisheye8k
        #return image.crop(cropped_coords) , image_q32, image_q37 , image_q42

    return image , image_q32 , image_q37 , image_fisheye8k

get_image.fisheye8K_couter = 0;

def color_encoding_woodscape_detection():
    detection_classes = dict(vehicles=(43, 125, 255), rider=(255, 0, 0), person=(216, 45, 128),
                             traffic_sign=(255, 175, 58), traffic_light=(43, 255, 255))
    detection_color_encoding = np.zeros((5, 3), dtype=np.uint8)
    for i, (k, v) in enumerate(detection_classes.items()):
        detection_color_encoding[i] = v
    return detection_color_encoding

@torch.no_grad()
def test_simple(args):
    """Function to predict for a single image or folder of images"""
    if not os.path.isdir(args.output_directory):
        os.mkdir(args.output_directory)

    decoder_name = "chengAttenDecoder" #"factorizedPriorDecoder"  #"chengAttenDecoder"  "" #
    encoder_name = "Res18Encoder_" +decoder_name


    encoder_path = os.path.join(args.pretrained_weights, encoder_name + ".pth")
    compression_path = os.path.join(args.pretrained_weights, decoder_name +".pth")


    print("=> Loading pretrained encoder")
    encoder = ResnetEncoder(num_layers=args.network_layers, pretrained=False).to(args.device)
    loaded_dict_enc = torch.load(encoder_path, map_location=args.device)

    # extract the height and width of image that this model was trained with
    feed_height = 288 # loaded_dict_enc['height']
    feed_width = 544 #loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()
    for m in encoder.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None


    print("=> Loading pretrained compression decoder")
    if decoder_name == "factorizedPriorDecoder":
        compression_decoder = FactorizedPriorDecoderAtten(64, 128).to(args.device)
    elif decoder_name == "chengAttenDecoder":
        compression_decoder =  Cheng2020AttentionDecoder(64).to(args.device)
    elif decoder_name == "ScaleHyperprior":
        compression_decoder = ScaleHyperpriorDecoderAtten(64,128).to(args.device)

    loaded_dict = torch.load(compression_path)
    compression_decoder.load_state_dict(loaded_dict)
    compression_decoder.eval()


    if args.dl_compress_decompress == True:
        if args.compression_quality == 0:
            compression_model = ScaleHyperprior(128, 192) .to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q0.pth')
        elif args.compression_quality == 4:
            compression_model = ScaleHyperprior(128, 256).to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q4.pth')
        elif args.compression_quality >= 8:
            #compression_model = ScaleHyperprior(192, 320) .to(args.device)
            #checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_scale_prior_Q8_2023.pth')
            compression_model = Cheng2020Attention(192) .to(args.device)
            checkpoint = torch.load('models/AEs/results_models_resnet_ae/compression_cheng_atten_Q3_2023.pth')

        compression_model.load_state_dict(checkpoint)
        #compression_model.update()
        compression_model.eval()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.join(args.output_directory, f"{args.video_name}.mp4")
    video = cv2.VideoWriter(video_name, fourcc, FRAME_RATE, (feed_width, feed_height))

    image_paths = [line.rstrip('\n') for line in open(args.val_file)]
    print(f"=> Predicting on {len(image_paths)} validation images")

    for idx, image_path in enumerate(image_paths):
        if image_path.endswith(f"_detection.png"):
            continue
        frame_index, cam_side = image_path.split('.')[0].split('_')
        input_image , input_image_q32 , input_image_q37,input_image_q42 , image_fisheye8k = pre_image_op(args, 0, frame_index, cam_side )

        if args.fishey8k_dataset_enable:
            input_image_fisheye8K = image_fisheye8k.resize((feed_width, feed_height), Image.LANCZOS)
            input_image_fisheye8K = transforms.ToTensor()(input_image_fisheye8K).unsqueeze(0)
            input_image_fisheye8K = input_image_fisheye8K.to(args.device)

            output_name_fisheye = f"camera1_A_{int(get_image.fisheye8K_couter)}"
            name_dest_im_fishey8k = os.path.join(args.output_directory , 'fisheye8K' , f"{output_name_fisheye}_org_rsz.png")

            #utils.save_image(input_image_fisheye8K.data, name_dest_im_fishey8k, normalize=True)



        input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = input_image.to(args.device)

        input_image_q32 = input_image_q32.resize((feed_width, feed_height), Image.LANCZOS)
        input_image_q32 = transforms.ToTensor()(input_image_q32).unsqueeze(0)
        input_image_q32 = input_image_q32.to(args.device)

        input_image_q37 = input_image_q37.resize((feed_width, feed_height), Image.LANCZOS)
        input_image_q37 = transforms.ToTensor()(input_image_q37).unsqueeze(0)
        input_image_q37 = input_image_q37.to(args.device)

        input_image_q42 = input_image_q42.resize((feed_width, feed_height), Image.LANCZOS)
        input_image_q42 = transforms.ToTensor()(input_image_q42).unsqueeze(0)
        input_image_q42 = input_image_q42.to(args.device)

        if args.dl_compress_decompress == True:
            with torch.no_grad():
                '''policy = T.AutoAugmentPolicy.IMAGENET
                augmenter = T.AutoAugment(policy)
                input_image = augmenter(input_image)'''
                augmenter = T.ColorJitter(#brightness=(0.8, 1.2),
                                       contrast=(0.5))
                                       #saturation=(0.8, 1.2),
                                       #hue=(-0.1, 0.1))
                #input_image = augmenter(input_image)
                '''out_enc = compression_model.compress(input_image)
                out_dec = compression_model.decompress(out_enc["strings"], out_enc["shape"])

                torch.clamp(out_dec["x_hat"], min=0, max=1 )
                x = input_image.squeeze(0)
                num_pixels = x.size(0) * x.size(1) * x.size(2)
                bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                print(" - psnr:", round(psnr(input_image, out_dec["x_hat"]), 5),
                " - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 5),
                "compressed size:", sum(len(s[0]) for s in out_enc["strings"]) , "bpp:",bpp)'''

                out_dec = compression_model.forward(input_image)
                x = input_image.squeeze(0)
                num_pixels = x.size(0) * x.size(1) * x.size(2)

                bpp = sum(
                    (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                    for likelihoods in out_dec["likelihoods"].values()
                )
                bpp = round ( bpp.item() , 3)
                psnr_value = round(psnr(x, out_dec["x_hat"]) , 2)
                psnr_value_q32 = round(psnr(x , input_image_q32.squeeze(0)), 5)
                psnr_value_q37 = round(psnr(x , input_image_q37.squeeze(0)), 5)
                psnr_value_q42 = round(psnr(x, input_image_q42.squeeze(0)), 3)
                mse = F.mse_loss(input_image, out_dec["x_hat"]).item()
                '''print(" psnr_dl: ",psnr_value,"- psnr_q32:", psnr_value_q32,"- psnr_q37:", psnr_value_q37 ,
                      "- psnr_vvc_42:", psnr_value_q42 ," - bpp :" ,bpp ,
                      " - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 5),
                      " - ms-ssim_q37:", round(ms_ssim(input_image_q37,input_image, data_range=1.0).item(), 5),
                      " - ms-ssim_q32:", round(ms_ssim(input_image_q32, input_image, data_range=1.0).item(), 5),
                      " - ms-ssim_q42:", round(ms_ssim(input_image_q42, input_image, data_range=1.0).item(), 5) )'''

                print("psnr:", psnr_value," - bpp :" ,bpp ,
                      "     - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 3))

                #print("psnr:", psnr_value_q42,
                #      "     - ms-ssim:", round(ms_ssim(input_image_q42, input_image, data_range=1.0).item(), 5))

                input_image = out_dec["x_hat"].to(args.device)
                #reconstructed_x = compression_model.forward(input_image)
                #input_image = reconstructed_x["x_hat"].to(args.device)

        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_im = os.path.join(args.output_directory, f"{output_name}_detection.png")

        # PREDICTION

        features = encoder(input_image_fisheye8K)
        out_dec_fisheye8k = compression_decoder(features)
        x_fisheye8k = input_image_fisheye8K.squeeze(0)
        num_pixels = x_fisheye8k.size(0) * x_fisheye8k.size(1) * x_fisheye8k.size(2)
        bpp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_dec_fisheye8k["likelihoods"].values()
        )
        bpp = round(bpp.item(), 3)
        psnr_fishey8k_value = round(psnr(x_fisheye8k, out_dec_fisheye8k["x_hat"]), 2)
        mse_fishey8k = F.mse_loss(input_image_fisheye8K, out_dec_fisheye8k["x_hat"]).item()
        print("- psnr_fishey8k_value:", psnr_fishey8k_value,
              "- mse_fishey8k:", mse_fishey8k ,
              "     - ms-ssim:", round(ms_ssim(input_image_fisheye8K, out_dec_fisheye8k["x_hat"], data_range=1.0).item(), 3),
              "bpp" , bpp)


        features = encoder(input_image)
        out_dec = compression_decoder(features)



        x = input_image.squeeze(0)
        num_pixels = x.size(0) * x.size(1) * x.size(2)



        #saving names
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        output_name_fisheye = f"camera1_A_{int(get_image.fisheye8K_couter )}"
        name_dest_im = os.path.join(args.output_directory + encoder_name, f"{output_name}_reconstructed.png")

        #name_dest_im_fishey8k = os.path.join(args.output_directory + encoder_name, f"{output_name}_reconstructed.png")
        name_dest_im_fishey8k = os.path.join(args.output_directory + encoder_name, f"{output_name_fisheye}_reconstructed_fisheye.png")

        utils.save_image(out_dec["x_hat"].to(args.device).data, name_dest_im, normalize=True)
        utils.save_image(out_dec_fisheye8k["x_hat"].to(args.device).data, name_dest_im_fishey8k, normalize=True)
        bpp = 0
        '''sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_dec["likelihoods"].values()
        )
        bpp = round(bpp.item(), 3)'''


        psnr_value = round(psnr(x, out_dec["x_hat"]), 2)
        mse = F.mse_loss(input_image, out_dec["x_hat"]).item()

        psnralt = PeakSignalNoiseRatio().to(args.device)
        psnr_valu_alt= psnralt(x, out_dec["x_hat"])

        print("mse:" , mse , "psnr_alt",psnr_valu_alt , " - psnr:", psnr_value, " - bpp :", bpp,
              "     - ms-ssim:", round(ms_ssim(input_image, out_dec["x_hat"], data_range=1.0).item(), 3))
        psnr_value_q32 = round(psnr(x, input_image_q32.squeeze(0)), 5)
        psnr_value_q37 = round(psnr(x, input_image_q37.squeeze(0)), 5)
        psnr_value_q42 = round(psnr(x, input_image_q42.squeeze(0)), 3)

        print("- psnr_q32:", psnr_value_q32,"- psnr_q37:", psnr_value_q37 ,
              "- psnr_vvc_42:", psnr_value_q42 ,
              " - ms-ssim_q37:", round(ms_ssim(input_image_q37,input_image, data_range=1.0).item(), 5),
              " - ms-ssim_q32:", round(ms_ssim(input_image_q32, input_image, data_range=1.0).item(), 5),
              " - ms-ssim_q42:", round(ms_ssim(input_image_q42, input_image, data_range=1.0).item(), 5) )
        input_image = out_dec["x_hat"].to(args.device)
        #outputs = decoder(features, img_dim=[feed_width, feed_height])
        #outputs = non_max_suppression(outputs["yolo_outputs"],
        #                              conf_thres=args.detection_conf_thres,
        #                              nms_thres=args.detection_nms_thres)

        img_d = input_image[0].cpu().detach().numpy()
        img_d = np.transpose(img_d, (1, 2, 0))
        img_cpu = np.zeros(img_d.shape, img_d.dtype)
        img_cpu[:, :, :] = img_d[:, :, :] * 255
        cv2.imwrite(name_dest_im, cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR))


        print(f"=> Processed {idx + 1} of {len(image_paths)} images - saved prediction to {name_dest_im}")

    video.release()
    print(f"=> LoL! beautiful video created and dumped to disk. \n"
          f"=> Done!")


if __name__ == '__main__':
    config = collect_args()
    params = yaml.safe_load(open(config.config))
    args = Tupperware(params)
    test_simple(args)
