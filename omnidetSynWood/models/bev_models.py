# models/bev_models.py
"""
Unified BEV + Compression Multi-Task Network.

This file contains the core modules for a multi-task learning framework that
jointly handles Bird's-Eye-View (BEV) map generation and multi-camera image
compression.

Key components:
- `Multi_Task_MultiCamera_Compression`: A learned compression model that operates
  on per-camera latent features, using a hyperprior and optional cross-view
  attention for context transfer.
- `MultiScaleBEVBackbone`: A feature fusion network that takes multi-scale
  features from all cameras and a fused latent representation to produce a
  unified BEV feature map.
- `BEVHead`: A decoder that generates the final BEV image from the fused
  BEV features.
- `BEV_MTL_Compression`: A top-level wrapper that integrates the compression
  and BEV generation pipelines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyasn1.codec.ber.encoder import NullEncoder
from pytorch_msssim import ssim


from models.compressAI.layers import GDN, MaskedConv2d , AttentionBlock
from models.compressAI.priors import CompressionModel, GaussianConditional
from models.compressAI.entropy_models import EntropyBottleneck
_COMPRESSION_IMPL_AVAILABLE = True


# -----------------------------------------
# Helpers
# -----------------------------------------
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=kernel_size // 2)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, output_padding=stride - 1)


class SubpelConv3x3(nn.Module):
    """
    A sub-pixel convolution layer that combines a convolution with PixelShuffle.
    This is often used as a learnable upsampling layer to avoid checkerboard artifacts.
    """

    def __init__(self, in_ch, out_ch, r=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * (r ** 2), 3, padding=1)
        self.shuffle = nn.PixelShuffle(r)

    def forward(self, x):
        return self.shuffle(self.proj(x))


def ste_round(x):
    # Straight-through estimator rounding (keeps gradient)
    return (x.round() - x).detach() + x


# -----------------------------------------
# Residual block
# -----------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))


# -----------------------------------------
# CrossViewAttention (fuse across V)
# -----------------------------------------
class CrossViewAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        # x: [B, V, C, H, W]
        B, V, C, H, W = x.shape
        h = self.num_heads
        dh = C // h
        HW = H * W

        def proj(module):
            t = module(x.view(B * V, C, H, W))      # (B*V, C, H, W)
            t = t.view(B, V, C, HW)                 # (B, V, C, HW)
            t = t.view(B, V, h, dh, HW).permute(0, 2, 1, 4, 3)  # (B,h,V,HW,dh)
            return t

        Q = proj(self.q_proj)
        K = proj(self.k_proj)
        Vv = proj(self.v_proj)

        Qloc = Q.mean(dim=2, keepdim=True)  # (B,h,1,HW,dh)
        scores = torch.einsum('bhqwd,bhvwd->bhqwv', Qloc, K).squeeze(2)  # (B,h,HW,V)
        attn = F.softmax(scores * self.scale, dim=-1)
        out = torch.einsum('bhwv,bhvwd->bhwd', attn, Vv)  # (B,h,HW,dh)
        out = out.view(B, h, HW, dh).permute(0, 1, 3, 2).reshape(B, h * dh, H, W)  # (B,C,H,W)
        return self.out_proj(out)


# -----------------------------------------
# EfficientAttention (for context transfer)
# -----------------------------------------
class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32,
                 head_count=8, value_channels=64):
        super().__init__()
        assert key_channels % head_count == 0, "key_channels must be divisible by head_count"
        assert value_channels % head_count == 0, "value_channels must be divisible by head_count"
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.head_count = head_count
        self.dk = key_channels // head_count
        self.dv = value_channels // head_count
        self.scale = (self.dk) ** -0.5

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        # target: (B, Cq, H, W), input: (B, Ci, H, W)
        B, _, H, W = input.shape
        HW = H * W
        k = self.keys(input).reshape(B, self.head_count, self.dk, HW)   # (B, heads, dk, HW)
        q = self.queries(target).reshape(B, self.head_count, self.dk, HW)  # (B, heads, dk, HW)
        v = self.values(input).reshape(B, self.head_count, self.dv, HW)  # (B, heads, dv, HW)

        k = F.softmax(k, dim=-1)    # normalize over spatial dim
        q = (q * self.scale)
        q = F.softmax(q, dim=-2)    # normalize over channel-sub-dim

        context = torch.einsum("bhdn,bhen->bhde", k, v)   # (B, heads, dk, dv)
        out = torch.einsum("bhde,bhdn->bhen", context, q)  # (B, heads, dv, HW)
        out = out.reshape(B, self.value_channels, H, W)
        return self.reprojection(out)


# -----------------------------------------
# PositionalEncoding2D and TransformerBlock
# -----------------------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("channels must be divisible by 4")
        self.channels = channels

    def forward(self, H, W, device):
        y_pos, x_pos = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        div_term = torch.exp(torch.arange(0, self.channels // 2, 2, device=device) *
                             -(torch.log(torch.tensor(10000.0)) / (self.channels // 2)))
        pe = torch.zeros(1, self.channels, H, W, device=device)
        pe[0, 0::4, :, :] = torch.sin(x_pos[None, None, :, :] * div_term[:, None, None])
        pe[0, 1::4, :, :] = torch.cos(x_pos[None, None, :, :] * div_term[:, None, None])
        pe[0, 2::4, :, :] = torch.sin(y_pos[None, None, :, :] * div_term[:, None, None])
        pe[0, 3::4, :, :] = torch.cos(y_pos[None, None, :, :] * div_term[:, None, None])
        return pe

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1, pool_ratio=4):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.pos_enc = PositionalEncoding2D(dim)
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        ph = max(1, H // self.pool_ratio)
        pw = max(1, W // self.pool_ratio)
        pooled = F.adaptive_avg_pool2d(x, (ph, pw))
        pooled = pooled + self.pos_enc(ph, pw, pooled.device)
        seq = pooled.flatten(2).permute(0, 2, 1)   # (B, ph*pw, C)
        seq = self.transformer(seq)
        out = seq.permute(0, 2, 1).view(B, C, ph, pw)
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

class C2f(nn.Module):
    """A more efficient feature fusion block inspired by YOLOv8's C2f module."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2 * self.c)
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.ModuleList(ResidualBlock(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.relu(self.bn(self.cv1(x))).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# -----------------------------------------
# Hyperprior (lightweight)
# -----------------------------------------
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=kernel_size // 2)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, output_padding=stride - 1)


class Hyperprior(CompressionModel):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int=192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )
        if out_planes == 2 * in_planes:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        params = self.hyper_decoder(z_hat)
        if out_z:
            return params, z_likelihoods, z_hat
        else:
            return params, z_likelihoods

    def compress(self, y):
        z = self.hyper_encoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.hyper_decoder(z_hat)
        return params, z_hat, z_strings #{"strings": z_string, "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        params = self.hyper_decoder(z_hat)
        return params, z_hat


# -----------------------------------------
# Multi-Joint Context Transfer (per-camera context aggregation)
# -----------------------------------------
class Multi_JointContextTransfer(nn.Module):
    def __init__(self, channels, head_count=2):
        super().__init__()
        self.rb = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.attn = EfficientAttention(key_in_channels=channels,
                                       query_in_channels=channels,
                                       key_channels=max(8, (channels // 8) * head_count),
                                       head_count=head_count,
                                       value_channels=max(16, (channels // 4) * head_count))
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            ResidualBlock(channels)
        )

    def forward(self, x, num_camera):
        """
        x: (B_total, C, H, W) where B_total = B * num_camera
        returns: (B_total, C, H, W)
        """
        B_total, C, H, W = x.shape
        if num_camera <= 0: raise ValueError("num_camera must be > 0")
        if B_total % num_camera != 0: raise ValueError("Batch must be divisible by num_camera")
        B = B_total // num_camera

        rb_x = self.rb(x)                         # (B_total, C, H, W)
        rb = rb_x.view(B, num_camera, C, H, W)    # (B, V, C, H, W)
        identity = x.view(B, num_camera, C, H, W)
        compact_list = []
        for cam_idx in range(num_camera):
            rb_cur = rb[:, cam_idx, :, :, :]     # (B, C, H, W)
            if num_camera > 1:
                others = [rb[:, i, :, :, :] for i in range(num_camera) if i != cam_idx]
                others = torch.stack(others, dim=1)   # (B, V-1, C, H, W)
                aggregate_rb = others.mean(dim=1)     # (B, C, H, W)
            else:
                aggregate_rb = torch.zeros_like(rb_cur)
            A = self.attn(rb_cur, aggregate_rb)      # (B, C, H, W)
            refined = self.refine(torch.cat([A, rb_cur], dim=1))
            compact = identity[:, cam_idx, :, :, :] + refined
            compact_list.append(compact)
        stacked = torch.stack(compact_list, dim=1)     # (B, V, C, H, W)
        out = stacked.view(B * num_camera, C, H, W)
        return out


# -----------------------------------------
# Compression module: expects per-camera latent y
# -----------------------------------------
class Multi_Task_MultiCamera_Compression(nn.Module):
    def __init__(self, in_channels=192, N=128, M=192, decode_atten=Multi_JointContextTransfer):
        """
        Multi-camera compression model with optional cross-view attention.

        This module takes a list of per-camera latent representations and performs
        learned compression. It includes an entropy model based on a hyperprior and
        a context model for accurate probability estimation.

        The reconstruction path can optionally use `Multi_JointContextTransfer` attention
        blocks to share information across camera views, leading to a more coherent
        and refined reconstruction, which is used for the compression loss.
        """
        super().__init__()
        self.in_channels = in_channels
        self.N = N
        self.M = M

        # Joint context transfer and split decoder path (for refined branch)
        self.atten_3 = decode_atten(M)
        self.decoder_1 = nn.Sequential(
            SubpelConv3x3(M, N, r=2),
            AttentionBlock(N),
            GDN(N, inverse=True),
            SubpelConv3x3(N, N, r=2),
            GDN(N, inverse=True),
            AttentionBlock(N),
        )
        self.atten_4 = NullEncoder #decode_atten(N)
        self.decoder_2 = nn.Sequential(
            SubpelConv3x3(N, N, r=2),
            AttentionBlock(N),
            GDN(N, inverse=True),
            SubpelConv3x3(N, N, r=2),
            GDN(N, inverse=True),
            SubpelConv3x3(N, 3, r=2),
        )

        # hyperprior + entropy parameter net
        self.hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M * 2)
        self.context_prediction = MaskedConv2d(M, M * 2, kernel_size=5, padding=2, stride=1)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 4, M * 10 // 3, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1), nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1)
        )
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x_list, use_atten_3=True, use_atten_4=False):
        """
        x_list: list of per-camera latent y tensors [B, M, H, W]
        use_atten_3 / use_atten_4: booleans to toggle refined stages
        Returns dictionary with:
            - "x_hat": x_hat_plain_list (used by compression loss by default)
            - "x_hat_plain": plain recon list
            - "x_hat_refined": refined recon list (used as BEV auxiliary)
            - "latents": original y latents list
            - optional: "latents_post_attn3", "mid_post_attn4"
            - "likelihoods": { "y": [...], "z": [...] }
        """
        assert isinstance(x_list, (list, tuple)), "expected list of per-camera latents"
        num_cams = len(x_list)
        B = x_list[0].shape[0]

        # 1) Compute params, likelihoods, and STE latents for all cameras
        y_ste_list = []
        y_latents = []
        y_likelihoods_list = []
        z_likelihoods_list = []

        for y in x_list:
            params, z_likelihoods = self.hyperprior(y)
            y_hat = self.gaussian_conditional.quantize(y, "noise" if self.training else "dequantize")
            ctx_params = self.context_prediction(y_hat)
            params = F.interpolate(params, size=ctx_params.size()[-2:])
            gaussian_params = self.entropy_parameters(torch.cat([params, ctx_params], dim=1))
            means_hat, scales_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

            # straight-through like quantization for decoder path
            y_ste = ste_round(y - means_hat) + means_hat

            y_ste_list.append(y_ste)
            y_latents.append(y)
            y_likelihoods_list.append(y_likelihoods)
            z_likelihoods_list.append(z_likelihoods)

        # ------- Plain path (no attention) -------
        # decode plain directly (use the split decoder to match refined shapes)
        mid_plain_list = [self.decoder_1(y_ste) for y_ste in y_ste_list]
        x_hat_plain_list = [self.decoder_2(mid) for mid in mid_plain_list]

        # ------- Refined path (with attention) -------
        # copy list to apply attention (atten_3) optionally
        y_for_refine = y_ste_list
        latents_post_attn3 = None
        mid_post_attn4 = None

        if use_atten_3 and num_cams > 1:
            stacked = torch.cat(y_for_refine, dim=0)                   # (B * V, M, H, W)
            stacked = self.atten_3(stacked, num_cams)                  # (B * V, M, H, W)
            y_for_refine = list(torch.split(stacked, B, dim=0))        # list len V
            latents_post_attn3 = y_for_refine                          # store for diagnostics

        mid_refined_list = [self.decoder_1(y_ste) for y_ste in y_for_refine]

        '''if use_atten_4 and num_cams > 1:
            stacked_mid = torch.cat(mid_refined_list, dim=0)           # (B * V, N, H1, W1)
            stacked_mid = self.atten_4(stacked_mid, num_cams)          # (B * V, N, H1, W1)
            mid_refined_list = list(torch.split(stacked_mid, B, dim=0))
            mid_post_attn4 = mid_refined_list                          # store for diagnostics'''

        x_hat_refined_list = [self.decoder_2(mid) for mid in mid_refined_list]

        # By default keep "x_hat" pointing to the plain branch so compression loss remains stable.
        # The refined branch is available as "x_hat_refined" for BEV and monitoring.
        return {
            "x_hat": x_hat_plain_list,
            "x_hat_plain": x_hat_plain_list,
            "x_hat_refined": x_hat_refined_list,
            "latents": y_ste_list,
            "latents_post_attn3": latents_post_attn3,
            "mid_post_attn4": mid_post_attn4,
            "likelihoods": {"y": y_likelihoods_list, "z": z_likelihoods_list}
        }


# -----------------------------------------
# BEV backbone & head (supports both 5D per-camera input, 4D concatenated input,
# plus optional aux concatenated scale and optional proj scale)
# -----------------------------------------
class MultiScaleBEVBackbone(nn.Module):
    def __init__(self, per_camera_in_channels_list, num_cameras, attn_channels=96, fuse_mid=256,
                 num_res_blocks=3, use_cross_view=True, proj_scale_channels=None,
                 aux_concat_scale_channels=None):
        """
        per_camera_in_channels_list: list of channels for each scale (per camera).
        num_cameras: number of cameras V (used to create concat lateral convs).
        proj_scale_channels: channels of the appended projected-latent scale (4D, not multiplied by V).
        aux_concat_scale_channels: per-camera channels of an auxiliary concatenated scale
                                   (expects 4D input [B, V*C_aux, H, W]).
        """
        super().__init__()
        self.use_cross_view = use_cross_view
        self.num_cameras = num_cameras
        self.per_camera_in_channels_list = list(per_camera_in_channels_list)
        self.proj_scale_channels = proj_scale_channels
        self.aux_concat_scale_channels = aux_concat_scale_channels

        # lateral convs for per-camera input (when input is 5D [B,V,C,H,W])
        self.lateral_convs_camera = nn.ModuleList([
            nn.Conv2d(in_ch, attn_channels, 1) for in_ch in self.per_camera_in_channels_list
        ])

        # lateral convs for concatenated input (when input is 4D [B, V*C, H, W])
        self.lateral_convs_concat = nn.ModuleList([
            nn.Conv2d(in_ch * num_cameras, attn_channels, 1) for in_ch in self.per_camera_in_channels_list
        ])

        # auxiliary concatenated scale (4D [B, V*C_aux, H, W])
        if aux_concat_scale_channels is not None:
            self.lateral_conv_aux = nn.Conv2d(aux_concat_scale_channels * num_cameras, attn_channels, 1)
        else:
            self.lateral_conv_aux = None

        # projected-latent scale (4D [B, proj_ch, H, W])
        if proj_scale_channels is not None:
            self.lateral_conv_proj = nn.Conv2d(proj_scale_channels, attn_channels, 1)
        else:
            self.lateral_conv_proj = None

        # count total scales passed into post_fusion
        total_scales = len(self.per_camera_in_channels_list)
        if self.lateral_conv_aux is not None: total_scales += 1
        if self.lateral_conv_proj is not None: total_scales += 1

        self.cross_view = CrossViewAttention(attn_channels) if use_cross_view else None

        # post fusion expects attn_channels * total_scales input
        self.post_fusion = nn.Sequential(
            nn.Conv2d(attn_channels * total_scales, fuse_mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_mid),
            nn.ReLU(inplace=True),
            C2f(fuse_mid, fuse_mid, n=2, shortcut=True), # Replaced TransformerBlock with C2f
            *[ResidualBlock(fuse_mid) for _ in range(num_res_blocks)]
        )

    def forward(self, multi_scale_features):
        """
        multi_scale_features: ordered list of per-scale tensors:
          - First: regular backbone scales (each either 5D [B,V,C,H,W] or 4D [B, V*C, H, W]), count == len(per_camera_in_channels_list).
          - Optional next: auxiliary concatenated scale (4D [B, V*C_aux, H, W]) if enabled.
          - Optional last: projected-latent scale (4D [B, proj_ch, H, W]) if enabled.
        """
        fused = []
        target_size = multi_scale_features[0].shape[-2:]
        base_scales = len(self.per_camera_in_channels_list)
        has_aux = self.lateral_conv_aux is not None
        has_proj = self.lateral_conv_proj is not None

        for i, feat in enumerate(multi_scale_features):
            if i < base_scales:
                # normal backbone scale
                if feat.dim() == 5:
                    # per-camera separated: [B, V, C, H, W]
                    B, V, C, H, W = feat.shape
                    feat_proj = self.lateral_convs_camera[i](feat.view(B * V, C, H, W)).view(B, V, -1, H, W)
                    if self.cross_view is not None:
                        fused_view = self.cross_view(feat_proj)   # (B, attn_channels, H, W)
                    else:
                        fused_view = feat_proj.mean(dim=1)
                elif feat.dim() == 4:
                    # concatenated across cameras: [B, V*C, H, W]
                    feat_proj = self.lateral_convs_concat[i](feat)  # (B, attn_channels, H, W)
                    fused_view = feat_proj
                else:
                    raise ValueError(f"Unexpected feature dims for scale {i}: {feat.shape}")

            elif has_aux and i == base_scales:
                # auxiliary concatenated scale (x_hat_refined concatenated across cameras)
                if feat.dim() != 4:
                    raise ValueError("Aux concatenated scale must be 4D [B, V*C_aux, H, W]")
                fused_view = self.lateral_conv_aux(feat)

            elif has_proj and i == base_scales + (1 if has_aux else 0):
                # projected-latent scale
                if feat.dim() != 4:
                    raise ValueError("Projected-latent scale must be 4D [B, proj_ch, H, W]")
                fused_view = self.lateral_conv_proj(feat)

            else:
                raise ValueError(f"Unexpected index {i} for provided multi_scale_features.")

            # resize to target size if needed
            if fused_view.shape[-2:] != target_size:
                fused_view = F.interpolate(fused_view, size=target_size, mode='bilinear', align_corners=False)
            fused.append(fused_view)

        stitched = torch.cat(fused, dim=1)
        return self.post_fusion(stitched)


# -----------------------------------------
# BEV head
# -----------------------------------------
class BEVHead(nn.Module):
    def __init__(self, in_channels, mid_channels=128, out_channels=3):
        super().__init__()
        self.target_height = 420  # Target BEV height from dataset config
        self.target_width = 300  # Target BEV width from dataset config

        # Calculate a width that corrects the aspect ratio for the given input height (144)
        # Target aspect H/W = 420/300 = 1.4. New width = 144 / 1.4 = 102.8 -> 103
        self.aspect_fix_width = 103

        # A decoder that refines features after aspect ratio correction and then upsamples.
        self.decoder = nn.Sequential(
            # Refine features at the new aspect ratio
            AttentionBlock(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),

            GDN(mid_channels, inverse=True),

            # Upsample by 3x. This gets us close to the target resolution.
            # Input: (144, 103) -> Output: (432, 309)
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, kernel_size=3, stride=3, padding=1, output_padding=2),
            GDN(mid_channels // 2, inverse=True),
            #AttentionBlock(mid_channels // 2),
            # Final projection to output channels
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # The backbone provides a feature map of shape (B, C_in, 144, 272)

        # Step 1: Correct the aspect ratio first using interpolation.
        # Input: (B, C, 144, 272) -> Output: (B, C, 144, 103)
        x_aspect_corrected = F.interpolate(x, size=(x.shape[2], self.aspect_fix_width), mode='bilinear',
                                           align_corners=False)

        # Step 2: Pass through the decoder. Output shape is now ~432x309.
        x = self.decoder(x_aspect_corrected)

        # Step 3: Final interpolation to the exact target size. This is now a very small adjustment.
        x = F.interpolate(x, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
        return x


# ------------------------------
# Latent fusion (learned)
# ------------------------------
class SpatialAwareLatentFusion(nn.Module):
    def __init__(self, num_cameras, proj_channels, out_channels, head_count=4):
        super().__init__()
        self.num_cameras = num_cameras
        self.pos_enc = PositionalEncoding2D(proj_channels)
        self.attention_fusion = Multi_JointContextTransfer(
            channels=proj_channels, head_count=head_count)
        self.out_proj = nn.Conv2d(proj_channels, out_channels, kernel_size=1)

    def forward(self, proj_list):
        enhanced_list = []
        for proj in proj_list:
            H, W = proj.shape[-2:]
            pos_enc = self.pos_enc(H, W, proj.device)
            enhanced = proj + pos_enc
            enhanced_list.append(enhanced)

        stacked = torch.cat(enhanced_list, dim=0)
        fused_stacked = self.attention_fusion(stacked, self.num_cameras)
        fused = torch.stack(fused_stacked.chunk(self.num_cameras, dim=0), dim=1).mean(dim=1)
        return self.out_proj(fused)


# -----------------------------------------
# Unified wrapper integrating everything
# -----------------------------------------
class BEV_MTL_Compression(nn.Module):
    def __init__(self,
                 num_cameras=4,
                 per_camera_in_channels_list=(64, 256, 512, 1024, 2048),
                 bev_proj_channels=64,
                 compression_M=192,
                 backbone_attn_channels=96,
                 backbone_fuse_mid=256,
                 use_cross_view=True):
        """
        per_camera_in_channels_list: channel counts for each backbone scale (per camera).
        compression_M: compression latent channels (M).
        """
        super().__init__()
        self.num_cameras = num_cameras
        self.per_camera_in_channels_list = list(per_camera_in_channels_list)
        deepest_in_ch = self.per_camera_in_channels_list[-1]  # e.g. 2048
        self.compression_M = compression_M
        self.compression_N = 128  # Corresponds to mid_planes in Hyperprior

        # Compression module expects latents with channels == compression_M
        self.compression = Multi_Task_MultiCamera_Compression(in_channels=compression_M, N=128, M=compression_M)

        # Project from backbone deepest channels (e.g. 2048) -> compression_M channels
        self.backbone_proj = nn.Conv2d(deepest_in_ch, compression_M, kernel_size=1)

        # Project the primary latent (y) to BEV channels
        self.latent_proj = nn.Conv2d(self.compression_M, bev_proj_channels, kernel_size=1)
        self.latent_fusion = SpatialAwareLatentFusion(num_cameras=self.num_cameras,
                                                      proj_channels=bev_proj_channels,
                                                      out_channels=bev_proj_channels)


        # Build BEV backbone (now with aux x_hat_refined concatenated scale)
        self.backbone = MultiScaleBEVBackbone(
            per_camera_in_channels_list=self.per_camera_in_channels_list,
            num_cameras=self.num_cameras,
            attn_channels=backbone_attn_channels,
            fuse_mid=backbone_fuse_mid,
            num_res_blocks=3,
            use_cross_view=use_cross_view,
            proj_scale_channels=bev_proj_channels,
            aux_concat_scale_channels=3,          # <-- x_hat_refined per camera (RGB)
        )

        self.bev_head = BEVHead(backbone_fuse_mid, mid_channels=backbone_fuse_mid // 2, out_channels=3)

    def forward(self, features_per_camera, compression_use_atten_3=True, compression_use_atten_4=True,
                bev_use_refined_aux=True):
        """
        features_per_camera: list(len=num_cameras) where each item is a list of per-scale feature maps
            features_per_camera[v][s] -> tensor [B, C_s, H_s, W_s]

        compression_use_atten_3 / compression_use_atten_4: toggle refined compression branches.
        bev_use_refined_aux: if True, BEV receives x_hat_refined concatenated across cameras as auxiliary input.
                             if False, BEV receives the plain path (still compression["x_hat"] remains plain).
        """
        if len(features_per_camera) != self.num_cameras:
            raise ValueError(f"Expected features for {self.num_cameras} cameras, got {len(features_per_camera)}")

        # Step A: prepare deepest features per camera and project them to compression_M channels
        deepest_feats = [feats[-1] for feats in features_per_camera]  # list of [B, C_deep, h, w]
        projected_latents = [self.backbone_proj(y) for y in deepest_feats]  # list of [B, M, h, w]

        # Step B: run compression module (expects latents already)
        comp_out = self.compression(projected_latents,
                                    use_atten_3=compression_use_atten_3,
                                    use_atten_4=compression_use_atten_4)  # returns many fields

        latents = comp_out["latents"]  # list of [B, M, h, w]

        # Step C: project latents to BEV channels and fuse across cameras
        proj_latents = [self.latent_proj(y) for y in latents]  # list of [B, bev_proj_channels, h, w]
        proj_fused = self.latent_fusion(proj_latents)          # [B, bev_proj_channels, h, w]

        # Step D: prepare per-scale concatenated backbone features for BEV backbone
        per_scale_concat = []
        num_scales = len(features_per_camera[0])
        for s in range(num_scales):
            per_cam_feats = [features_per_camera[v][s] for v in range(self.num_cameras)]
            concat = torch.cat(per_cam_feats, dim=1)  # [B, num_cameras * C_s, H_s, W_s]
            per_scale_concat.append(concat)

        # Step E: append auxiliary concatenated x_hat_refined across cameras (if desired)
        if bev_use_refined_aux:
            xhat_refined_list = comp_out.get("x_hat_refined", None)
            if xhat_refined_list is None:
                raise RuntimeError("BEV requested refined aux but compression did not provide x_hat_refined.")
            # concat across cameras -> [B, V*3, Hx, Wx]
            aux_concat = torch.cat(xhat_refined_list, dim=1)
            per_scale_concat.append(aux_concat)

        # Step F: append proj_fused as an extra "scale"
        target_size = per_scale_concat[0].shape[-2:]
        proj_resized = F.interpolate(proj_fused, size=target_size, mode='bilinear', align_corners=False)
        per_scale_concat.append(proj_resized)

        # Step G: BEV backbone & head
        bev_latent = self.backbone(per_scale_concat)
        bev_map = self.bev_head(bev_latent)

        return {
            "compression": comp_out,
            "bev": bev_map,
            "proj_latent": proj_fused
        }


# -----------------------------------------
# Loss helpers
# -----------------------------------------
class StitchingLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - torch.clamp(ssim(pred, target, data_range=1.0), 0.0, 1.0)
        return (1 - self.alpha) * l1_loss + self.alpha * ssim_loss

class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - torch.clamp(ssim(pred, target, data_range=1.0), 0.0, 1.0)
        return (1 - self.alpha) * l1_loss + self.alpha * ssim_loss
