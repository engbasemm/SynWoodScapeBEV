import torch
import torch.nn as nn
from typing import List

class MultiViewFusion(nn.Module):
    """
    A simple module to fuse features from multiple camera views by
    concatenating them and applying a convolution. This serves as a fast
    and simple baseline for multi-view fusion.
    """
    def __init__(self, in_channels, num_cameras=4, **kwargs):
        super().__init__()
        # The fusion convolution will take features from all cameras concatenated
        # along the channel dimension.
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_cameras, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, multi_view_features: List[torch.Tensor]):
        """
        Fuses features from multiple camera views.
        Args:
            multi_view_features (list[torch.Tensor]): A list of feature tensors, one for each camera.
                                                      Each tensor has shape (B, C, H, W).
        Returns:
            torch.Tensor: The fused feature map of shape (B, C, H, W).
        """
        # Concatenate features along the channel dimension
        concatenated_features = torch.cat(multi_view_features, dim=1)
        
        # Apply the fusion convolution
        return self.fusion_conv(concatenated_features)

class TransformerMultiViewFusion(nn.Module):
    """
    A module to fuse features from multiple camera views using a transformer-based
    attention mechanism, as described in the paper. This is more powerful but also
    more computationally expensive than the simple fusion.
    """
    def __init__(self, in_channels, num_cameras=4, embed_dim=256, num_heads=8):
        super().__init__()
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim

        # Input projection for each camera's features
        self.input_projection = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Transformer Encoder Layer for self-attention across camera views
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output convolution to project fused features back to original channel depth
        self.output_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, multi_view_features: List[torch.Tensor], return_attention: bool = False):
        """
        Fuses features from multiple camera views.
        Args:
            multi_view_features (list[torch.Tensor]): A list of feature tensors, one for each camera.
                                                      Each tensor has shape (B, C, H, W).
            return_attention (bool): If True, returns the attention maps from the transformer layers.
        Returns:
            torch.Tensor: The fused feature map of shape (B, C, H, W).
            list[torch.Tensor] (optional): A list of attention maps if return_attention is True.
        """
        if not multi_view_features:
            return None
        
        batch_size, _, height, width = multi_view_features[0].shape
        
        # 1. Project and flatten features from all views
        projected_features = [self.input_projection(feat) for feat in multi_view_features]
        flat_features = [p.flatten(2).permute(0, 2, 1) for p in projected_features]

        # 2. Stack features for transformer: (B, num_cameras, H*W, D)
        stacked_features = torch.stack(flat_features, dim=1)
        
        # Reshape for transformer: (B*H*W, num_cameras, D)
        transformer_input = stacked_features.permute(0, 2, 1, 3).reshape(batch_size * height * width, self.num_cameras, self.embed_dim)

        # 3. Apply transformer for cross-view attention, manually iterating to get weights
        attention_maps = []
        transformer_output = transformer_input
        for layer in self.transformer_encoder.layers:
            # Re-implementing the TransformerEncoderLayer forward pass to get attention weights.
            # The self_attn module is a MultiheadAttention layer.
            sa_out, attn_weights = layer.self_attn(transformer_output, transformer_output, transformer_output, need_weights=True)
            transformer_output = transformer_output + layer.dropout1(sa_out)
            transformer_output = layer.norm1(transformer_output)
            
            # Feed-forward part
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(transformer_output))))
            transformer_output = transformer_output + layer.dropout2(ff_out)
            transformer_output = layer.norm2(transformer_output)
            
            attention_maps.append(attn_weights)

        fused_sequence = self.transformer_encoder.norm(transformer_output) if self.transformer_encoder.norm else transformer_output

        # 4. Average the attended features and reshape back
        mean_fused_sequence = fused_sequence.mean(dim=1)
        fused_features_unflat = mean_fused_sequence.reshape(batch_size, height * width, self.embed_dim).permute(0, 2, 1).reshape(batch_size, self.embed_dim, height, width)

        # 5. Project back to original channel dimension
        final_output = self.output_conv(fused_features_unflat)

        if return_attention:
            return final_output, attention_maps
        else:
            return final_output