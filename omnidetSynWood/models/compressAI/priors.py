# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch.quantization as tq
from torch.ao.nn.quantized import FloatFunctional
import math
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F # May be needed by sub-modules
from typing import List, Tuple, Dict, Optional, Any # For type hints
# pylint: disable=E0611,E0401

from .entropy_models import EntropyBottleneck, GaussianConditional
from .layers import GDN

from models.AEs.utils import conv, deconv, update_registered_buffers
from .ans import BufferedRansEncoder, RansDecoder
from .entropy_models import EntropyBottleneck, GaussianConditional
from .layers import GDN, MaskedConv2d
from .layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)


# pylint: enable=E0611,E0401


__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "ScaleHyperprior",
    "MeanScaleHyperprior"
]


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)


class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def compress_decompress(self,x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        strings =  [y_strings]
        shape = y.size()[-2:]

        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class FactorizedPriorDecoder1(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        x_hat = self.g_s(x[1])

        return {
            "x_hat": x_hat
        }
'''
class FactorizedPriorDecoderAtten(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        if(x[1].size(dim=1) == 64):
            x_hat = self.g_s(x[1])
        elif (x[1].size(dim=1)  == 256):
            x_hat = self.g_s(x[0])

        return {
            "x_hat": x_hat
        }
'''

#quantization implementation #1 ( works well with 8bit only)

# --- Learnable Quantization Modules ---
class RoundSTE(torch.autograd.Function):
    """Straight-Through Estimator for torch.round()"""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output  # Pass gradient straight through


class LearnableAffineQuantizer(nn.Module):
    """
    Learnable Affine Quantizer (LSQ/LSQ+ style).
    Quantizes and dequantizes an input tensor using learnable scale and zero-point.
    """

    def __init__(self,
                 num_bits: int = 8,
                 per_channel: bool = False,
                 num_channels: Optional[int] = None,
                 symmetric: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.device_param = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        if per_channel and num_channels is None:
            raise ValueError("num_channels must be provided for per-channel quantization.")
        self.num_channels = num_channels

        self.qmin = 0.0
        self.qmax = 2.0 ** num_bits - 1.0

        # Determine shape for learnable parameters (e.g., for B,C,H,W C is dim 1)
        param_shape = (1, num_channels, 1, 1) if per_channel and num_channels and num_channels > 0 else (1,)

        # Learnable scale - initialized to 1.0.
        # LSQ often initializes scale based on data statistics (e.g., 2*mean(abs(weights))/sqrt(qmax_effective_range))
        self.scale = nn.Parameter(torch.full(param_shape, 1.0, device=self.device_param, dtype=torch.float32))

        if self.symmetric:
            # For symmetric quantization, zero_point is fixed.
            # If qmin=0 (unsigned), and we want real 0 to map to the middle of the quantized range:
            fixed_zero_point_val = round((self.qmin + self.qmax) / 2.0)
            self.register_buffer('zero_point', torch.full(param_shape, fixed_zero_point_val, device=self.device_param,
                                                          dtype=torch.float32))
        else:  # Asymmetric (LSQ+ style), learn zero_point
            # Initial zero_point, should ideally be calibrated. Defaulting to mid-range.
            initial_zp_val = round((self.qmin + self.qmax) / 2.0)
            self.zero_point = nn.Parameter(
                torch.full(param_shape, initial_zp_val, device=self.device_param, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure parameters are on the same device as input x
        # This is good practice if the module might be moved after initialization
        current_scale = self.scale.to(x.device)
        current_zero_point = self.zero_point.to(x.device)

        # LSQ often uses grad_scale_factor = 1.0 / math.sqrt(x.numel() * self.qmax) for scale gradient
        # For simplicity, we rely on PyTorch's autograd with STE for now.
        # A more complete LSQ implementation would have a custom backward for scale.

        s_positive = current_scale.abs() + 1e-8  # Ensure scale is positive

        # Quantize
        quant_val_float = x / s_positive + current_zero_point
        quant_val_rounded = RoundSTE.apply(quant_val_float)  # Apply STE for rounding
        quant_clamped = torch.clamp(quant_val_rounded, self.qmin, self.qmax)

        # Dequantize
        dequantized_val = (quant_clamped - current_zero_point) * s_positive
        return dequantized_val

    def extra_repr(self):
        return (f'num_bits={self.num_bits}, per_channel={self.per_channel}, symmetric={self.symmetric}, '
                f'qmin={self.qmin}, qmax={self.qmax}')


# --- Synthesis Transform (Now takes dequantized features from LearnableQuantizer) ---
class SynthesisTransformGs(nn.Module):
    def __init__(self, N: int, args: Optional[Any] = None):
        super().__init__()
        self.N = N;
        self.args = args
        self.block1 = AttentionBlock(N);
        self.block2 = ResidualBlock(N, N)
        self.block3_upsample = ResidualBlockUpsample(N, N, 2);
        self.block4 = ResidualBlock(N, N)
        self.block5_upsample = ResidualBlockUpsample(N, N, 2);
        self.block6_upsample = ResidualBlockUpsample(N, N, 2)
        self.block7 = ResidualBlock(N, N);
        self.block8_subpel = subpel_conv3x3(N, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Takes dequantized features
        x = self.block1(x);
        x = self.block2(x);
        x = self.block3_upsample(x);
        x = self.block4(x)
        x = self.block5_upsample(x);
        x = self.block6_upsample(x);
        x = self.block7(x);
        x = self.block8_subpel(x)
        return x


class FactorizedPriorDecoderAtten(CompressionModel):
    def __init__(self, N: int, M: int, args: Optional[Any] = None, **kwargs: Any):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        self.N = N;
        self.M = M;
        self.args = args

        if M != N:
            self.initial_conv = nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1)
        else:
            self.initial_conv = nn.Identity()

        self.g_s = SynthesisTransformGs(N, args=args)

        self.enable_feature_quant = getattr(args, 'enable_reconstruction_feature_quant', False)
        self.num_quant_bits_recon = getattr(args, 'num_quant_bits_reconstruction', 8)
        self.debug_quant_recon = getattr(args, 'debug_reconstruction_quantization', False)
        self.quantize_recon_per_channel = getattr(args, 'quantize_recon_per_channel', True)
        self.quantize_recon_symmetric = getattr(args, 'quantize_recon_symmetric', False)
        self.device_for_quant = getattr(args, 'device',
                                        torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))

        self.quantizer = None
        if self.enable_feature_quant and self.num_quant_bits_recon > 0:
            num_channels_for_quantizer = N  # Quantizer operates on features with N channels
            self.quantizer = LearnableAffineQuantizer(
                num_bits=self.num_quant_bits_recon,
                per_channel=self.quantize_recon_per_channel,
                num_channels=num_channels_for_quantizer if self.quantize_recon_per_channel else None,
                symmetric=self.quantize_recon_symmetric,
                device=self.device_for_quant
            )
            if self.debug_quant_recon:
                print(f"FactorizedPriorDecoderAtten: Initialized LearnableAffineQuantizer.")
                if hasattr(self.quantizer, 'extra_repr'): print(f"  Quantizer details: {self.quantizer.extra_repr()}")
        else:
            self.quantizer = nn.Identity()  # If not quantizing, pass features through

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, backbone_features_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(backbone_features_list, list) or len(backbone_features_list) < 4:
            raise ValueError(f"FactorizedPriorDecoderAtten expects a list of at least 4 backbone features.")
        latent_representation = backbone_features_list[3]

        should_print_module_debug = False
        if self.debug_quant_recon:
            if hasattr(self.args, 'current_step') and hasattr(self.args, 'log_frequency'):
                if self.args.current_step == 0 or \
                        (self.args.log_frequency > 0 and self.args.current_step % self.args.log_frequency == 0):
                    should_print_module_debug = True
            else:
                should_print_module_debug = True

        if should_print_module_debug:
            print(
                f"FactorizedPriorDecoderAtten (Step {getattr(self.args, 'current_step', 'N/A')}): Input feature map for initial_conv (shape {latent_representation.shape})")
        if latent_representation.size(1) != self.M:
            raise ValueError(f"Selected latent has {latent_representation.size(1)} ch, but model expects M={self.M}.")

        features_to_process = self.initial_conv(latent_representation)

        if self.quantizer is not None and not isinstance(self.quantizer, nn.Identity):
            if should_print_module_debug:
                quant_type = "Per-Channel" if self.quantize_recon_per_channel else "Per-Tensor"
                quant_symm = "Symmetric" if self.quantize_recon_symmetric else "Asymmetric (LSQ+ style)"
                print(
                    f"  Applying Learnable Quantization (shape {features_to_process.shape}) to {self.num_quant_bits_recon} bits ({quant_type}, {quant_symm}) before g_s.")
                spatial_factor_sq = self.downsampling_factor ** 2;
                N_channels_for_bpp = features_to_process.size(1)
                bpp_before = (N_channels_for_bpp * 32) / spatial_factor_sq
                bpp_after = (N_channels_for_bpp * self.num_quant_bits_recon) / spatial_factor_sq
                print(f"    Theoretical bpp before (float32 features): {bpp_before:.4f}")
                print(f"    Theoretical bpp after (quantized features): {bpp_after:.4f}")
                print(f"    Theoretical Compression Ratio (bits): {32.0 / self.num_quant_bits_recon:.2f}x")

            dequantized_features = self.quantizer(features_to_process)
            x_reconstructed = self.g_s(dequantized_features)
        else:  # No quantization or quantizer is Identity
            x_reconstructed = self.g_s(features_to_process)

        return {"x_hat": x_reconstructed}




class FactorizedPriorDecoderAttenQunatzE2E(CompressionModel):
    def __init__(self, N: int, M: int, args: Optional[Any] = None, **kwargs: Any):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        self.N = N
        self.M = M
        self.args = args

        if M != N:
            self.initial_conv = nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1)
        else:
            self.initial_conv = nn.Identity()

        self.g_s = nn.Sequential(
            AttentionBlock(N), ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2), ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N), subpel_conv3x3(N, 3, 2),
        )

        if args is not None:
            self.enable_feature_quant = getattr(args, 'enable_reconstruction_feature_quant', False)
            self.num_quant_bits_recon = getattr(args, 'num_quant_bits_reconstruction', 8)
            self.debug_quant_recon = getattr(args, 'debug_reconstruction_quantization', False)
            self.quantize_recon_per_channel = getattr(args, 'quantize_recon_per_channel', False) # New flag
            self.device_for_quant = getattr(args, 'device', torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
        else:
            self.enable_feature_quant = True; self.num_quant_bits_recon = 8
            self.debug_quant_recon = False; self.quantize_recon_per_channel = True
            self.device_for_quant = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    @property
    def downsampling_factor(self) -> int: return 2 ** 4

    def forward(self, backbone_features_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(backbone_features_list, list) or len(backbone_features_list) < 4:
            raise ValueError(f"FactorizedPriorDecoderAtten expects a list of at least 4 backbone features.")

        latent_representation = backbone_features_list[3]

        should_print_module_debug = False
        if self.debug_quant_recon:
            if hasattr(self.args, 'current_step') and hasattr(self.args, 'log_frequency'):
                if self.args.current_step == 0 or \
                   (self.args.log_frequency > 0 and self.args.current_step % self.args.log_frequency == 0):
                    should_print_module_debug = True
            else:
                should_print_module_debug = True

        if should_print_module_debug:
            print(f"FactorizedPriorDecoderAtten (Step {getattr(self.args, 'current_step', 'N/A')}): Input feature map for g_s (shape {latent_representation.shape})")

        if latent_representation.size(1) != self.M:
            raise ValueError(f"Selected latent has {latent_representation.size(1)} ch, but model expects M={self.M}.")

        features_to_decode = self.initial_conv(latent_representation)

        if self.enable_feature_quant and self.num_quant_bits_recon > 0:
            quant_type = "Per-Channel" if self.quantize_recon_per_channel else "Per-Tensor"
            if should_print_module_debug:
                print(f"  Quantizing features (shape {features_to_decode.shape}) to {self.num_quant_bits_recon} bits ({quant_type}) before g_s.")
                spatial_factor_sq = self.downsampling_factor**2
                N_channels_for_bpp = features_to_decode.size(1)
                bpp_before = (N_channels_for_bpp * 32) / spatial_factor_sq
                bpp_after = (N_channels_for_bpp * self.num_quant_bits_recon) / spatial_factor_sq
                print(f"    Theoretical bpp before: {bpp_before:.4f} (based on {N_channels_for_bpp} ch, 32-bit float, {self.downsampling_factor}x spatial downsampling)")
                print(f"    Theoretical bpp after: {bpp_after:.4f} (quantized to {self.num_quant_bits_recon} bits)")
                print(f"    Theoretical Compression Ratio (bits): {32.0 / self.num_quant_bits_recon:.2f}x for these features.")

            quantized_features = quantize_dequantize(
                features_to_decode,
                num_bits=self.num_quant_bits_recon,
                per_channel=self.quantize_recon_per_channel, # Pass the new flag
                device_to_use=self.device_for_quant,
                debug_print=should_print_module_debug
            )
            x_reconstructed = self.g_s(quantized_features)
        else:
            x_reconstructed = self.g_s(features_to_decode)

        return {"x_hat": x_reconstructed}



# Built in TQ quantization , its working ( very low bit rates)
# --- Synthesis Transform (Now QAT-Compatible) ---
class SynthesisTransformGsQtz(nn.Module):
    def __init__(self, N: int, args: Optional[Any] = None):
        super().__init__()
        self.N = N
        self.args = args
        self.block1 = AttentionBlock(N)
        # Use the QAT-compatible ResidualBlock
        self.block2 = ResidualBlock(N, N)
        self.block3_upsample = ResidualBlockUpsample(N, N, 2)
        self.block4 = ResidualBlock(N, N)
        self.block5_upsample = ResidualBlockUpsample(N, N, 2)
        self.block6_upsample = ResidualBlockUpsample(N, N, 2)
        self.block7 = ResidualBlock(N, N)
        self.block8_subpel = subpel_conv3x3(N, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3_upsample(x)
        x = self.block4(x)
        x = self.block5_upsample(x)
        x = self.block6_upsample(x)
        x = self.block7(x)
        x = self.block8_subpel(x)
        return x


# --- Helper function to create a QConfig for a custom bit-width ---
def get_custom_qat_qconfig(bits: int = 8, backend: str = 'fbgemm') -> tq.QConfig:
    """
    Creates a QAT configuration for a specified number of bits.
    """
    qmin_activ, qmax_activ = 0, 2 ** bits - 1
    qmin_weights, qmax_weights = -2 ** (bits - 1), 2 ** (bits - 1) - 1

    activation_fake_quant = tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=qmin_activ, quant_max=qmax_activ,
        dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False
    )
    weight_fake_quant = tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=qmin_weights, quant_max=qmax_weights,
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
    )
    return tq.QConfig(activation=activation_fake_quant, weight=weight_fake_quant)


class FactorizedPriorDecoderAttenQtz(CompressionModel):
    def __init__(self, N: int, M: int, args: Optional[Any] = None,
                 quant_bits: Optional[int] = 6, backend: str = 'fbgemm', **kwargs: Any):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        self.N = N
        self.M = M
        self.args = args

        if M != N:
            self.initial_conv = nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1)
        else:
            self.initial_conv = nn.Identity()

        self.g_s = SynthesisTransformGsQtz(N, args=args)
        self.enable_feature_quant = getattr(args, 'enable_reconstruction_feature_quant', False)
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # --- Integrated QAT Preparation ---
        if quant_bits is not None:
            self.train()  # QAT must be configured in training mode.
            if quant_bits == 8:
                self.qconfig = tq.get_default_qat_qconfig(backend)
            else:
                self.qconfig = get_custom_qat_qconfig(bits=quant_bits, backend=backend)

            # prepare_qat modifies the model in-place to insert observers
            tq.prepare_qat(self, inplace=True)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, backbone_features_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(backbone_features_list, list) or len(backbone_features_list) < 4:
            raise ValueError(f"FactorizedPriorDecoderAtten expects a list of at least 4 backbone features.")
        latent_representation = backbone_features_list[3]

        if latent_representation.size(1) != self.M:
            raise ValueError(f"Selected latent has {latent_representation.size(1)} ch, but model expects M={self.M}.")

        features_to_process = self.initial_conv(latent_representation)

        if self.enable_feature_quant:
            quantized_features = self.quant(features_to_process)
            reconstructed_quantized = self.g_s(quantized_features)
            x_reconstructed = self.dequant(reconstructed_quantized)
        else:
            x_reconstructed = self.g_s(features_to_process)

        return {"x_hat": x_reconstructed}

class FactorizedPriorDecoder(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        x_hat = self.g_s(x[1])

        return {
            "x_hat": x_hat
        }


class FactorizedPriorNoEntropy(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    def forward(self, x):
        y = self.g_a(x)
        x_hat = self.g_s(y)

        return {
            "x_hat": x_hat
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def compress_decompress(self,x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        strings =  [y_strings]
        shape = y.size()[-2:]

        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ScaleHyperpriorOrg(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}



class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, stride=1, kernel_size=3),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperpriorNoEntropy(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N, stride=1, kernel_size=3),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        scales_hat = self.h_s(z)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ScaleHyperpriorDecoder(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3, stride=1, kernel_size=3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            #conv(N, M, stride=1, kernel_size=3),
            conv(N, M),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        #z = self.h_a(torch.abs(x[1]))
        z_hat, z_likelihoods = self.entropy_bottleneck(x[1])
        scales_hat = self.h_s(z_hat)
        #scales_hat = self.h_s(z)
        y_hat, y_likelihoods = self.gaussian_conditional(x[0], scales_hat)
        x_hat = self.g_s(y_hat)



        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperpriorDecoderAtten(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
        self.h_s = nn.Sequential(
            conv3x3(N, N, padding= 1),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 1),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 1),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N  ,  padding=1),
        )

        '''self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            #conv(N, M, stride=1, kernel_size=3),
            conv(N, M),
            nn.ReLU(inplace=True),
        )'''

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        #z = self.h_a(torch.abs(x[1]))
        z_hat, z_likelihoods = self.entropy_bottleneck(x[0])
        scales_hat = self.h_s(z_hat)
        #scales_hat = self.h_s(z)
        y_hat, y_likelihoods = self.gaussian_conditional(x[0], scales_hat)
        x_hat = self.g_s(y_hat)



        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class ScaleHyperpriorDecoder1(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_s = nn.Sequential(
            deconv(M, N, stride=1, kernel_size=3),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N, stride=1),
            GDN(N, inverse=True),
            deconv(N, 3, stride=1, kernel_size=3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            #conv(N, M),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        z = self.h_a(torch.abs(x[0]))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        #scales_hat = self.h_s(z)
        y_hat, y_likelihoods = self.gaussian_conditional(x[0], scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            #nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            conv(3, N, kernel_size=5, stride=2 ),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        x_hat = self.g_s(y_hat)
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        #x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )


        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv


class JointAutoregressiveHierarchicalPriorsDecoder(MeanScaleHyperprior):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            #nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            conv(3, N, kernel_size=5, stride=2 ),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        #y = self.g_a(x)
        #z = self.h_a(y)
        if(self.N == 64):
            input =x[0]
        else:
            input = x[2]
        z = self.h_a(input)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            #y, "noise" if self.training else "dequantize"
            input, "noise" if self.training else "dequantize"
        )
        x_hat = self.g_s(y_hat)
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #_, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        _, y_likelihoods = self.gaussian_conditional(input, scales_hat, means=means_hat)
        #x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )


        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
#===================================================



class ImprovedFPDA(MeanScaleHyperprior):
    """
    A robust autoencoder for joint compression and perception, inspired by
    state-of-the-art learned compression models. This version is designed for
    stability and high-fidelity reconstruction by operating on earlier, richer
    features from the main backbone.
    """
    def __init__(self, N=192, M=192, in_channels_encoder=1024, out_height=288, out_width=544, use_refinement_head=True, **kwargs):
        super().__init__(N=N, M=M, **kwargs)


        self.out_height = out_height
        self.out_width = out_width
        self.use_refinement_head = use_refinement_head


        # Analysis transform: Compresses ResNet features to a latent representation `y`.
        # Takes features from an earlier ResNet layer (e.g., layer3, 1024 channels).
        self.g_a = nn.Sequential(
            ResidualBlock(in_channels_encoder, N),
            nn.Conv2d(N, N, kernel_size=3, stride=2, padding=1),  # Downsample
            ResidualBlock(N, N),
            nn.Conv2d(N, M, kernel_size=3, stride=2, padding=1),  # Downsample
            AttentionBlock(M)
        )


        # Refined Synthesis transform with more frequent attention, based on your suggestions.
        # Total upsampling is 64x (6 stages of 2x) to match the downsampling path.
        self.g_s = nn.Sequential(
            AttentionBlock(M),
            ResidualBlock(M, N),


            # Upsample 1 & 2 -> 1/16 resolution
            subpel_conv3x3(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            AttentionBlock(N), # Mid-level attention
            ResidualBlock(N, N),


            # Upsample 3 & 4 -> 1/4 resolution
            subpel_conv3x3(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, N, 2),
            AttentionBlock(N), # Mid-level attention
            ResidualBlock(N, N),


            # Upsample 5 & 6 -> Full resolution
            subpel_conv3x3(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )


        # Optional refinement head to sharpen details post-reconstruction
        if self.use_refinement_head:
            self.refinement_head = nn.Sequential(
                ResidualBlock(3, 32),
                nn.Conv2d(32, 3, kernel_size=3, padding=1)
            )


        # Hyperprior analysis transform: Compresses `y` into a hyper-latent `z`.
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )


        # Hyperprior synthesis transform: Reconstructs parameters for `y` from `z`.
        self.h_s = nn.Sequential(
            # Set output_padding to 0 to correctly reverse the h_a downsampling.
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(N, M * 3 // 2, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 3 // 2, M * 2, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, backbone_features_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Use features from layer3 of ResNet-50 (1/16 resolution, 1024 channels).
        # This is a more stable choice than the deepest layer.
        features_to_compress = backbone_features_list[3]


        y = self.g_a(features_to_compress)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)


        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_reconstructed = self.g_s(y_hat)


        # Optional refinement head to sharpen details
        if self.use_refinement_head:
            x_reconstructed = self.refinement_head(x_reconstructed)


        # The standard upsample-and-crop pattern to handle aspect ratio distortion from the encoder.
        if x_reconstructed.shape[2] != self.out_height or x_reconstructed.shape[3] != self.out_width:
            h_large, w_large = x_reconstructed.shape[2:]
            h_start = (h_large - self.out_height) // 2
            w_start = (w_large - self.out_width) // 2
            x_reconstructed = x_reconstructed[:, :, h_start:h_start + self.out_height, w_start:w_start + self.out_width]


        return {
            "x_hat": x_reconstructed,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
