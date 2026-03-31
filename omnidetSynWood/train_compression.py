"""
Semantic segmentation training for OmniDet.

# author: Varun Ravi Kumar <rvarun7777@gmail.com>

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import gc
import argparse
import torch
from pytorch_msssim import ssim, ms_ssim
from torchmetrics import PeakSignalNoiseRatio
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader.synwoodscape_loader import SynWoodScapeRawDataset
from data_loader.woodscape_loader import WoodScapeRawDataset
from colorama import Fore, Style
from models.resnet import ResnetEncoder
from losses.ratedistortion_loss import RateDistortionLoss
from models.compression_decoder import gAutoencoder,DecoderIn256 , UnetDecoder , DecoderIn512 , DecoderIn64
from utils import TrainUtils
from torch.nn import functional as F
from models.compressAI.ldmic import SVS_JPC, Multi_LDMIC_checkboard, Multi_JointContextTransfer
from models.compression_decoder_vae import ResNet_VAE
from models.compressAI.priors import ImprovedFPDA ,FactorizedPriorDecoder,ScaleHyperprior,ScaleHyperpriorDecoder , ScaleHyperpriorNoEntropy , FactorizedPriorDecoderAtten , FactorizedPriorDecoderAttenQtz
from models.compressAI.waseda import Cheng2020Anchor , Cheng2020Attention , Cheng2020AttentionDecoder
from models.compressAI.codecs import HM
from models.compressAI.pretrained import  load_pretrained
from torchvision import transforms
from models.VAEs.models.vanilla_vae import VanillaVAE
from models.VAEs.models.wae_mmd import  WAE_MMD
from models.VAEs.models.beta_vae import  BetaVAE
from utils_ae import *

import matplotlib.pyplot as plt

torch.set_grad_enabled(True)

class CompressionInit(TrainUtils):


    def __init__(self, args):
        super().__init__(args)
        self.compression_decoder = self.args.compression_decoder

        self.first_time = True

        #setup HEVC codec
        cwd = os.getcwd()
        codec_arg =  argparse.Namespace(build_dir=cwd+'/../HM/bin/',config=cwd+'/../HM/cfg/encoder_intra_high_throughput_rext.cfg'  , rgb=False)
        self.codec = HM (codec_arg)

        self.best_compression_loss = float('inf')

        #setup DL codec based
        self.enable_compression = self.args.enable_compression

        if self.enable_compression == False:
            self.dl_compress_decompress = self.args.dl_compress_decompress
        else:
            self.dl_compress_decompress = False

        self.train_cifar_compression = self.args.train_cifar_compression
        self.load_compression_model = self.args.load_compression_model
        self.compression_quality = self.args.compression_quality
        self.compression_criterion = RateDistortionLoss()

        gc.collect()
        torch.cuda.empty_cache()


class CompressionModel(CompressionInit):
    def __init__(self, args):
        super().__init__(args)

        # --- Init model ---
        if self.enable_compression == False :
            self.models["encoder"] = ResnetEncoder(num_layers=self.args.network_layers, pretrained=True).to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.encoder_channels = self.models["encoder"].num_ch_enc
            if self.dl_compress_decompress == True:
                if self.compression_decoder == "factorized_prior":
                    if self.compression_quality == 0:
                         self.compression_model = ScaleHyperprior(128, 192).to(self.device)
                         checkpoint = torch.load('./models/AEs/results_models_resnet_ae/compression_scale_prior_Q0.pth')
                    elif self.compression_quality == 4:
                        self.compression_model = ScaleHyperprior(128, 256).to(self.device)
                        checkpoint = torch.load('./models/AEs/results_models_resnet_ae/compression_scale_prior_Q4.pth')
                    elif self.compression_quality >= 8:
                        self.compression_model = ScaleHyperprior(192, 320).to(self.device)
                        checkpoint = torch.load('./models/AEs/results_models_resnet_ae/compression_scale_prior_Q8.pth')

                    self.compression_model.load_state_dict(checkpoint)

        else:
            if self.compression_decoder == "factorized_prior":
                if self.compression_quality == 0:
                    self.compression_model = ScaleHyperprior(128, 192).to(self.device)
                else:
                    #self.compression_model = ScaleHyperpriorNoEntropy(192, 320).to(self.device)
                    self.compression_model = ScaleHyperprior(192, 320).to(self.device)
                    #self.compression_model = Cheng2020Anchor(192).to(self.device)
                    #self.compression_model = Cheng2020Attention(192).to(self.device)

                self.models["compression"] = self.compression_model
                self.parameters_to_train += list(self.models["compression"].parameters())
            elif self.compression_decoder == "SVS_JPC":
                self.compression_model = gAutoencoder(
                    ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                    SVS_JPC(N=192, M=192, decode_atten=Multi_JointContextTransfer)).to(self.device)
            elif self.compression_decoder == "Multi_LDMIC_checkboard":
                self.compression_model = gAutoencoder(
                    ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                    Multi_LDMIC_checkboard(N=192, M=192, decode_atten=Multi_JointContextTransfer, training=True)).to(
                    self.device)
            elif self.compression_decoder == "VAEAutoencoder":
                '''
                self.compression_model =  BetaVAE(  in_channels= 3,  latent_dim= 128,  loss_type= 'B',
                                                    gamma= 10.0,  max_capacity= 25,  Capacity_max_iter= 10000).to(self.device)
                self.models["compression"] = self.compression_model
                self.parameters_to_train += list(self.models["compression"].parameters())
                self.compression_criterion = self.compression_model.loss_function
                '''
                '''
                self.compression_model =  WAE_MMD(3, 128,reg_weight=5000,kernel_type='rbf').to(self.device)
                self.models["compression"] = self.compression_model
                self.parameters_to_train += list(self.models["compression"].parameters())
                self.compression_criterion = self.compression_model.loss_function

                '''
                self.compression_model = VanillaVAE(3, 128).to(self.device)
                self.models["compression"] = self.compression_model
                self.parameters_to_train += list(self.models["compression"].parameters())
                self.compression_criterion = self.compression_model.loss_function

            #CNN_fc_hidden1, CNN_fc_hidden2 = 1024*1, 1024*1
                #CNN_embed_dim = 256*6*3  # latent dim extracted by 2D CNN
                #res_size = 224  # ResNet image size
                #dropout_p = 0.2  # dropout probability
                #self.compression_model = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p,
                #        CNN_embed_dim=CNN_embed_dim).to(self.device)
                #self.models["compression"] = self.compression_model
                #self.parameters_to_train += list(self.models["compression"].parameters())

            elif self.compression_decoder == "chengAttenDecoder":
                num_layers = self.args.network_layers
                if (num_layers == 18) :
                    self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                                                  Cheng2020AttentionDecoder(128)).to(self.device)
                elif(num_layers == 50):
                    self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                                                  Cheng2020AttentionDecoder(64)).to(self.device)
            elif self.compression_decoder == "factorizedPriorDecoder":
                num_layers = self.args.network_layers
                if (num_layers == 18):
                    self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),FactorizedPriorDecoderAtten(64,256)).to(self.device)

                if (num_layers == 50):
                    self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),FactorizedPriorDecoderAtten(64,1024)).to(self.device)

            elif self.compression_decoder == "ImprovedFPDA_8bit":
                    num_layers = self.args.network_layers
                    if (num_layers == 18):
                        self.compression_model = gAutoencoder(
                            ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                            ImprovedFPDA(256, 256, bits=8)).to(self.device)

                    if (num_layers == 50):
                        self.compression_model = gAutoencoder(
                            ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                            ImprovedFPDA(256, 256, in_channels_encoder= 1024)).to(self.device)

            elif self.compression_decoder == "FactorizedPriorDecoderAttenQtz":
                    num_layers = self.args.network_layers
                    if (num_layers == 18):
                        self.compression_model = gAutoencoder(
                            ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                            FactorizedPriorDecoderAttenQtz(64, 256)).to(self.device)

                    if (num_layers == 50):
                        self.compression_model = gAutoencoder(
                            ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                            FactorizedPriorDecoderAttenQtz(64, 1024)).to(self.device)

            elif self.compression_decoder == "ScaleHyperpriorDecoder":
                self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),ScaleHyperpriorDecoder(64,128)).to(self.device)
            elif self.compression_decoder == "unet":
                self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),UnetDecoder(2)).to(self.device)
            elif self.compression_decoder == "resnet_in_64":
                self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                                                   DecoderIn64()).to(self.device)
            elif self.compression_decoder == "resnet_in_256":
                self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                                                   DecoderIn256()).to(self.device)
            elif self.compression_decoder == "resnet_in_512":
                self.compression_model = gAutoencoder(ResnetEncoder(num_layers=self.args.network_layers, pretrained=True),
                                                   DecoderIn512()).to(self.device)
            else:
                raise NotImplementedError

            if self.compression_decoder != "factorized_prior" and self.compression_decoder !="VAEAutoencoder":
                self.models["encoder"] = self.compression_model.encoder
    #            for m in self.models["encoder"].modules():
    #                for child in m.children():
    #                    if type(child) == nn.BatchNorm2d:
    #                        child.track_running_stats = False
    #                        child.running_mean = None
    #                        child.running_var = None

                self.models["decoder"] = self.compression_model.decoder
            # ---------------- Pre-trained AE --------------
            if self.train_cifar_compression == True :

                #autencoder=    torch.load('./models/AEs/results_models_resnet_ae/compression.pth')
                if self.compression_decoder != "factorized_prior" and self.compression_decoder !="VAEAutoencoder":
                    self.models["decoder"] = self.models["decoder"].to(self.device)
                    self.models["encoder"] = self.models["encoder"].to(self.device)

                self.simple_train(self.compression_decoder )

                if self.compression_decoder == "factorized_prior":
                    torch.save(self.models["compression"].state_dict(),
                               './models/AEs/results_models_resnet_ae/factorizedprior.pth')
                else:
                    torch.save(self.models["encoder"].state_dict(), './models/AEs/results_models_resnet_ae/ResNetEncoder.pth')
                    torch.save(self.models["decoder"].state_dict(), './models/AEs/results_models_resnet_ae/ResNetDecoderIn512v2.pth')
                #self.models["encoder"] = torch.load('./models/AEs/results_models_resnet_ae/UnetEncoder.pth')

            if (self.load_compression_model == True) :
                if self.compression_decoder == "factorized_prior" or (self.compression_decoder == "factorizedPriorDecoder"):
                    compression_path = os.path.join(self.args.pretrained_weights, "factorizedPriorDecoder.pth")
                    #loaded_dict = torch.load('./models/AEs/results_models_resnet_ae/factorizedprior.pth')
                    loaded_dict = torch.load(compression_path)
                    self.compression_model.decoder.load_state_dict(loaded_dict)
                else:
                    loaded_dict = torch.load('./models/AEs/results_models_resnet_ae/ResNetDecoderIn64.pth')
                    self.models["decoder"] .load_state_dict(loaded_dict)
                    #loaded_dict = torch.load('./models/AEs/results_models_resnet_ae/ResNetEncoderIn64.pth')
                    #self.models["encoder"].load_state_dict(loaded_dict)

            if self.compression_decoder != "factorized_prior":        #self.models["decoder"] = torch.load('./models/AEs/results_models_resnet_ae/ResNetDecoder512_WoodScape.pth')
                self.models["decoder"] = self.models["decoder"].to(self.device)
                self.parameters_to_train += list(self.models["decoder"].parameters())
                self.parameters_to_train += list(self.models["encoder"].parameters())
                self.encoder_channels = self.models["encoder"].num_ch_enc




        if args.use_multiple_gpu:
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            if  self.enable_compression == True :
                self.models["decoder"] = torch.nn.DataParallel(self.models["decoder"])



        print(f"=> Training on the {self.args.dataset.upper()} dataset \n"
              f"=> Training model named: {self.args.model_name} \n"
              f"=> Models and tensorboard events files are saved to: {self.args.output_directory} \n"
              f"=> Training is using the cuda device id: {self.args.cuda_visible_devices} \n"
              f"=> Loading {self.args.dataset} training and validation dataset")

        # --- Load Data ---
        self.train_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                                 path_file=args.train_file,
                                                 is_train=True,
                                                 config=args)
        collate_train = self.train_dataset.collate_fn if "detection" in self.args.train else None
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=False,
                                       drop_last=True,
                                       collate_fn=collate_train)

        print(f"{Fore.RED}=> Total number of training examples: {len(self.train_dataset)}{Style.RESET_ALL}")

        val_dataset = SynWoodScapeRawDataset(data_path=args.dataset_dir,
                                          path_file=args.val_file,
                                          is_train=False,
                                          config=args)

        collate_val = val_dataset.collate_fn if "detection" in self.args.train else None
        self.val_loader = DataLoader(val_dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=False,
                                     drop_last=True,
                                     collate_fn=collate_val)

        self.val_iter = iter(self.val_loader)

        print(f"{Fore.YELLOW}=> Total number of validation examples: {len(val_dataset)}{Style.RESET_ALL}")

        self.num_total_steps = len(self.train_dataset) // args.batch_size * args.epochs


        self.configure_optimizers()

    def pre_init(self):
        if self.args.pretrained_weights:
            self.load_model()

        #self.save_args()

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def compression_train(self):
        # Add test images and graph to TensorBoard
        if self.enable_compression == False:
            return 0

        global_loss = float('inf')
        losses = dict()
        self.step = 0
        for self.epoch in range(self.args.epochs):
            self.training_loss = 0
            self.ssim_train = 0
            # switch to train mode
            self.set_train()
            data_loading_time = 0
            gpu_time = 0
            before_op_time = time.time()

            for batch_idx, inputs in enumerate(self.train_loader):
                current_time = time.time()
                data_loading_time += (current_time - before_op_time)
                before_op_time = current_time
                # -- PUSH INPUTS DICT TO DEVICE --
                self.inputs_to_device(inputs)
                x = inputs["color", 0, 0]
                #utils.save_image(x.data, 'results/original.png', normalize=True)
                #resized = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
                #utils.save_image(resized.data, 'results/original_resized.png', normalize=True)

                # -- COMPUTE GRADIENT AND DO OPTIMIZER STEP --
                self.optimizer.zero_grad()

                # -- RESNET AUTO ENCODER ---#
                outputs = self.compression_model.forward(x)

                #utils.save_image(outputs.data, 'results/test.png', normalize=True)


                #losses["msssim_loss"].backward()
                #self.optimizer.step()


                losses = self.compression_criterion(outputs, inputs["color", 0, 0])

                losses["compression_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.compression_model.parameters(), 1)
                if 'x_hat' in outputs:
                    ms_ssim_val = ms_ssim(outputs['x_hat'], x, data_range=1,
                                          size_average=True)  #ms_ssim_metric(outputs.data, x.data)
                else:
                    ms_ssim_val = ms_ssim( outputs, x, data_range=1, size_average=True ) #ms_ssim_metric(outputs.data, x.data)
                losses["msssim_loss"] = 1 -ms_ssim_val
                self.optimizer.step()

                self.training_loss += (losses["compression_loss"].item() * self.args.batch_size)
                #losses["ssim_loss"] = 1 - (ssim(x.data, outputs.data, data_range=1, size_average=False) ) # return (N,))
                #self.ssim_train += (losses["ssim_loss"].mean())

                duration = time.time() - before_op_time
                gpu_time += duration

                if batch_idx % self.args.log_frequency == 0:
                    self.log_time(batch_idx, duration, losses["compression_loss"].cpu().data, data_loading_time, gpu_time)
                    self.compression_statistics("train", inputs, outputs, losses)
                    data_loading_time = 0
                    gpu_time = 0

                if self.step % self.args.val_frequency == 0 and self.step != 0:
                    # -- SAVE DETECTION MODEL WITH BEST WEIGHTS BASED ON VALIDATION mAP --
                    self.compression_val(self.epoch )

                self.step += 1
                before_op_time = time.time()

            # Validate on each step, save model on improvements
            if(self.training_loss  < self.best_compression_loss ) and (self.epoch > 10):
                if self.compression_decoder == "factorized_prior":
                    #torch.save(self.models["compression"].state_dict(), './models/AEs/results_models_resnet_ae/compression_cheng_atten_Q8_2023.pth')
                    torch.save(self.models["compression"].state_dict(),
                               './models/AEs/results_models_resnet_ae/compression_scale_prior_Q0_2023.pth')
                else:
                    torch.save(self.models["encoder"].state_dict(), './models/AEs/results_models_resnet_ae/Encoder.pth')
                    torch.save(self.models["decoder"].state_dict(), './models/AEs/results_models_resnet_ae/Decoder.pth')
                self.best_compression_loss = self.training_loss
                print("Saving better model with loss:" + str(self.training_loss))




            self.lr_scheduler.step()

        print("Training complete!")



    @torch.no_grad()
    def compression_val(self , epoch_num):

        if self.compression_decoder != "factorized_prior":
            for m in self.models["encoder"].modules():
                for child in m.children():
                    if type(child) == nn.BatchNorm2d:
                        child.track_running_stats = False
                        child.running_mean = None
                        child.running_var = None

        """Validate the compression model"""
        self.set_eval()

        val_losses = dict()

        for val_inputs in self.val_loader:
            self.inputs_to_device(val_inputs)

            if self.compression_decoder != "factorized_prior":
                compressed_data = self.models["encoder"].forward(val_inputs["color", 0, 0])
                outputs = self.models["decoder"].forward(compressed_data)
            else:
                outputs = self.compression_model.forward(val_inputs["color", 0, 0])

            #utils.save_image(outputs.data, 'results/' + str(epoch_num) + '.png', normalize=True)

            val_losses = self.compression_criterion(outputs, val_inputs["color", 0, 0])
            if type(outputs) is dict:
                if 'x_hat' in outputs:
                    outputs = outputs["x_hat"]
            val_losses["ssim_loss"] = ssim(val_inputs["color", 0, 0].data, outputs.data, data_range=1, size_average=False)  # return (N,)
            val_losses["ssim_loss"] = 1 - val_losses["ssim_loss"]

            psnr = PeakSignalNoiseRatio().to(self.device)
            val_losses["psnr"] = psnr(outputs, val_inputs["color", 0, 0])


            ms_ssim_val = ms_ssim(outputs, val_inputs["color", 0, 0], data_range=1, size_average=True)  # ms_ssim_metric(outputs.data, x.data)
            val_losses["msssim_loss"] = 1 - ms_ssim_val

            _, predictions = torch.max(outputs.data, 1)
            #self.metric.add(predictions, val_inputs["color", 0, 0])
        #outputs["class_iou"], outputs["mean_iou"] = self.metric.value()

        # Compute stats for the tensorboard
        self.compression_statistics("val", val_inputs, outputs, val_losses)
        #self.metric.reset()
        val_compression = val_losses["compression_loss"].data

        del val_inputs, val_losses
        #del val_losses
        self.set_train()

        return val_compression

    # create default evaluator for doctests
    def save_best_compression_weights(self):
        # 2D Compression validation on each step and save model on improvements.
        compression_loss = self.compression_val( self.epoch)
        if (compression_loss < self.best_compression_loss):
            self.best_compression_loss = compression_loss
            torch.save(self.models["encoder"].state_dict(), './models/AEs/results_models_resnet_ae/Encoder.pth')
            torch.save(self.models["decoder"].state_dict(), './models/AEs/results_models_resnet_ae/Decoder.pth')
            print("Saving better model with loss:" + str(compression_loss))




    def compression_statistics(self, mode, inputs, outputs, losses) -> None:

        writer = self.writers[mode]

        orig_grid = make_grid(inputs["color", 0, 0].data, nrow=self.args.batch_size)
        grid = []
        writer.add_image('Original images', orig_grid, self.step)
        #writer.add_graph(self.compression_model ,inputs["color", 0, 0].data ,use_strict_trace=False)
        #writer.add_graph(self.models["encoder"], inputs["color", 0, 0].data)
        #writer.add_graph(self.models["decoder"], compressed_data.data)
        if (type(outputs) is dict ) :
            if  ('x_hat' in outputs):
                grid = make_grid(outputs["x_hat"].data, nrow=self.args.batch_size)
        else:
            grid = make_grid(outputs.data, nrow=self.args.batch_size)

        if mode == 'train':
            writer.add_image('reconstructed trained images', grid, self.step)
        else :
            writer.add_image('reconstructed validated images', grid, self.step)

        for loss, value in losses.items():
            writer.add_scalar(f"{loss}", value.mean(), self.step)


    def simple_train(self , compression_decoder):

        self.compression_decoder = compression_decoder

        repeat_patch = True
        repeat_batch_count = 100000
        batch_size = 16


        # Define transform
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((64, 64)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
             
        CIFAR_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=True,
                                                 download=True, transform=transform)

        test_set = torchvision.datasets.CIFAR10(root='../data/CIFAR10', train=False,
                                                download=True, transform=transform)
        train_set, val_set = train_val_split(CIFAR_set, 0.1)
        '''
        #       =========================  CelebA Dataset  =========================

        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize((64, 64)),
                                               transforms.ToTensor(), ])


        TRAIN_DATA_PATH = "data/celeba/celeba/"
        TEST_DATA_PATH = "data/celeba/celeba/"

        # custome data
        train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform)
        test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform)

        trainsubsetlen = 4000
        testsubsetlen = 1000
        indicestrain = np.arange(trainsubsetlen)
        indicestrain = np.random.permutation(indicestrain)

        indicestest = np.arange(testsubsetlen)
        indicestest = np.random.permutation(indicestest)

        # select train/test/val, for demo I am using 70,15,15
        train_indices = indicestrain[:int(0.7 * trainsubsetlen)]
        test_indices = indicestest[int(0.85 * testsubsetlen):]

        train_set = torch.utils.data.Subset(train_data, train_indices)
        val_set = torch.utils.data.Subset(test_data, test_indices)
        test_set = val_set
'''

        #print("Number of parameters in model: {0}".format(compression().num_params))

        param_names = ('init_lr', 'batch_size', 'weight_decay')
        parameters = OrderedDict(
            run=[0.001, batch_size, 0.001],
        )

        m = RunManager()
        num_epochs = 200

        for hparams in RunBuilder.get_runs_from_params(param_names, parameters):

            # Instantiate a network model
            self.ae = self.compression_model.to(self.device)

            # Construct a DataLoader object with training data
            train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=10, shuffle=False)
            test_images, _ = next(iter(test_loader))
            test_noisy_images = image_noiser(test_images)
            # Define optimizer
            optimizer = optim.SGD(self.ae.parameters(), lr=hparams.init_lr, momentum=0.9, weight_decay=hparams.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 60, 0.95)

            #self.inputs_to_device(test_images)
            #self.inputs_to_device(test_noisy_images)
            # Setup run instance

            m.begin_run(hparams, self.ae, test_images.to(self.device), test_noisy_images.to(self.device))
            print('Now training model with hyperparameters: init_lr={0}, batch_size={1}, weight_decay={2}'
                  .format(hparams.init_lr, hparams.batch_size, hparams.weight_decay))

            # Start training loop
            for epoch in range(num_epochs):
                m.begin_epoch()

                # Train the model
                for i, batch in enumerate(train_loader):
                    images, _ = batch
                    images = images.to(self.device)
                    # Zero all gradients
                    optimizer.zero_grad()
                    # Calculating the loss
                    preds = self.ae(images)
                    if self.compression_decoder == "VAEAutoencoder":
                        loss = self.compression_criterion(*preds , M_N = 0.00025)
                        loss = loss['loss']
                    else:
                        loss = self.compression_criterion(preds, images)
                        loss = loss['compression_loss']


                    if i % 10 == 0:
                        with torch.no_grad():
                            val_images, _ = next(iter(val_loader))
                            val_images = val_images.to(self.device)
                            #val_images = images
                            val_preds = self.ae(val_images)
                            if self.compression_decoder == "VAEAutoencoder":
                                val_loss = self.compression_criterion(*val_preds, M_N = 0.00025)
                                val_loss = val_loss["loss"]
                            else:
                                val_loss = self.compression_criterion(val_preds, val_images)
                                val_loss = val_loss["compression_loss"]




                            m.track_loss(val_loss, val_images.size(0), mode='val')
                        print('Epoch {0}, iteration {1}: train loss {2}, val loss {3}'.format(epoch + 1,
                                                                                              i * hparams.batch_size,
                                                                                              round(loss.item(), 6),
                                                                                              round(val_loss.item(),
                                                                                                    6)))

                    # Backpropagate
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1)

                    # Update the weights
                    optimizer.step()

                    if repeat_patch == True:
                        for k in range (repeat_batch_count):
                            preds = self.ae(images)
                            if self.compression_decoder == "VAEAutoencoder":
                                #loss = F.mse_loss(preds[1],preds[0])
                                loss = self.compression_criterion(*preds, M_N = 0.00025)
                                loss = loss ['loss']

                                preds = preds [0]
                            else:
                                loss = self.compression_criterion(preds, images)
                                loss = loss["compression_loss"]


                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1)
                            optimizer.step()
                            print('Epoch {0}:' , round(loss.item(), 6))
                            if self.first_time == True:
                                plt.interactive(False)
                                fig, ax = plt.subplots(2, 1)
                                #preds = preds['x_hat']
                                grid_img = torchvision.utils.make_grid(preds, nrow=16)
                                preds = grid_img.permute(1, 2, 0)
                                im1 = ax[1].imshow(preds.cpu())

                                grid_img_orig = torchvision.utils.make_grid(images, nrow=16)
                                preds_orig = grid_img_orig.permute(1, 2, 0)
                                im2 = ax[0].imshow(preds_orig.cpu())
                                #plt.show(im2)
                                #plt.show(im1)
                                #imgplot = plt.imshow(preds.cpu())
                                self.first_time = False
                            else:
                                #preds = preds['x_hat']
                                grid_img = torchvision.utils.make_grid(preds, nrow=16)
                                preds = grid_img.permute(1, 2, 0)
                                preds = np.clip(preds.cpu(), 0, 1)
                                im1.set_data(preds)
                                fig.canvas.draw_idle()
                                plt.pause(.1)


                    m.track_loss(loss, images.size(0), mode='train')

                m.end_epoch()

            # torch.save(ae, './models/150epochs_' + str(hparams) + '.pth')
            m.end_run()
            print("Model has finished training.\n")
            scheduler.step()

        m.save('results_final')
        print("Training completed.")

    def test_same_image(self,images):
        # Calculating the loss
        preds = self.ae(images)
        if self.first_time == True:
            fig, ax = plt.subplots(1, 1)
            grid_img = torchvision.utils.make_grid(preds, nrow=16)
            preds = grid_img.permute(1, 2, 0)
            im = ax.imshow(preds.cpu())
            # imgplot = plt.imshow(preds.cpu())
            self.first_time = False
        else:
            grid_img = torchvision.utils.make_grid(preds, nrow=16)
            preds = grid_img.permute(1, 2, 0)
            im.set_data(preds.cpu())
            fig.canvas.draw_idle()
            plt.pause(0.1)