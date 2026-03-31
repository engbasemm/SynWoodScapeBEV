"""
View Transformer module for OmniDet, based on Lift-Splat-Shoot.

# author: Basem Barakat

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.
"""

import torch
import torch.nn as nn

class ViewTransformerLSS(nn.Module):
    """
    Transforms features from image space to BEV space using the Lift-Splat-Shoot methodology.
    """
    def __init__(self, grid_config, in_channels, out_channels, D, debug=False):
        """
        Args:
            grid_config (dict): Configuration for the BEV grid.
                'xbound': [min_x, max_x, resolution_x]
                'ybound': [min_y, max_y, resolution_y]
                'zbound': [min_z, max_z, resolution_z]
            in_channels (int): Number of input channels from the image feature map.
            out_channels (int): Number of output channels for the BEV feature map.
            D (int): Number of discrete depth points to sample along each camera ray.
            debug (bool): Whether to print debug information.
        """
        super().__init__()
        self.grid_config = grid_config
        self.D = D
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.debug = debug
        # Create BEV grid coordinates
        xbounds = self.grid_config['xbound']
        ybounds = self.grid_config['ybound']
        
        dx = torch.tensor(xbounds[2])
        dy = torch.tensor(ybounds[2])
        
        self.x_grid = torch.arange(xbounds[0], xbounds[1], dx)
        self.y_grid = torch.arange(ybounds[0], ybounds[1], dy)
        
        # Frustum point cloud generation
        self.depth_net = nn.Conv2d(in_channels, self.D + self.in_channels, kernel_size=1, padding=0)

        # BEV feature map generation
        self.bev_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def get_geometry(self, intrinsics, extrinsics, H, W):
        """
        Generates the 3D points for the frustum.
        """
        B = intrinsics.shape[0]
        device = intrinsics.device

        # Create depth and pixel coordinate grids
        depths = torch.linspace(self.grid_config['dbound'][0], self.grid_config['dbound'][1], self.D, device=device).view(1, self.D, 1, 1)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        if self.debug:
            print(f"    [get_geometry] depths range: min={depths.min():.2f}, max={depths.max():.2f}")
            print(f"    [get_geometry] pix_coords (x) range: min={xs.min():.2f}, max={xs.max():.2f}")
            print(f"    [get_geometry] pix_coords (y) range: min={ys.min():.2f}, max={ys.max():.2f}")


        ones = torch.ones((B, H, W), device=device, dtype=torch.float32)
        xs = xs.expand(B, H, W)
        ys = ys.expand(B, H, W)
        # pix_coords shape: (B, 3, H, W)
        pix_coords = torch.stack([xs, ys, ones], dim=1)

        # Unproject to 3D camera coordinates
        inv_K = torch.inverse(intrinsics)
        cam_points_unit_flat = inv_K @ pix_coords.flatten(2)
        if self.debug:
            print(f"    [get_geometry] inv_K (sample 0):\n{inv_K[0]}")
            print(f"    [get_geometry] cam_points_unit_flat (x) range: min={cam_points_unit_flat[:,0].min():.2f}, max={cam_points_unit_flat[:,0].max():.2f}")
            print(f"    [get_geometry] cam_points_unit_flat (y) range: min={cam_points_unit_flat[:,1].min():.2f}, max={cam_points_unit_flat[:,1].max():.2f}")
            print(f"    [get_geometry] cam_points_unit_flat (z) range: min={cam_points_unit_flat[:,2].min():.2f}, max={cam_points_unit_flat[:,2].max():.2f}")

        # Reshape to add depth dimension for broadcasting
        cam_points_unit = cam_points_unit_flat.unsqueeze(2)

        # Scale by depth values to create frustum
        depths_reshaped = depths.view(1, 1, self.D, 1)
        cam_points = depths_reshaped * cam_points_unit  # Resulting shape: (B, 3, D, H*W)
        cam_points = cam_points.view(B, 3, self.D, H, W)
        if self.debug:
            print(f"    [get_geometry] cam_points (x) range: min={cam_points[:,0].min():.2f}, max={cam_points[:,0].max():.2f}")
            print(f"    [get_geometry] cam_points (y) range: min={cam_points[:,1].min():.2f}, max={cam_points[:,1].max():.2f}")
            print(f"    [get_geometry] cam_points (z) range: min={cam_points[:,2].min():.2f}, max={cam_points[:,2].max():.2f}")


        # Transform to ego-vehicle coordinates
        cam_points_homogeneous = torch.cat([cam_points, torch.ones((B, 1, self.D, H, W), device=device)], dim=1)
        ego_points = extrinsics.view(B, 4, 4) @ cam_points_homogeneous.flatten(2) # B, 4, D*H*W

        return ego_points.view(B, 4, self.D, H, W)[:, :3] # Return only x,y,z

    def forward(self, img_features, intrinsics, extrinsics):
        B, C, H, W = img_features.shape
        if self.debug:
            print(f"\n--- [ViewTransformerLSS Debug] ---")
            print(f"Input img_features shape: {img_features.shape}")
            print(f"Input intrinsics shape: {intrinsics.shape}")
            print(f"Input extrinsics shape: {extrinsics.shape}")
            print(f"  Grid Config: xbound={self.grid_config['xbound']}, ybound={self.grid_config['ybound']}, dbound={self.grid_config['dbound']}")
            print(f"  Intrinsics (sample 0):\n{intrinsics[0]}")
            print(f"  Extrinsics (sample 0):\n{extrinsics[0]}")

        # 1. Lift: Create context and depth distribution
        context_depth = self.depth_net(img_features)
        context = context_depth[:, :self.in_channels]
        depth_dist = context_depth[:, self.in_channels:].softmax(dim=1) # Softmax over depth dimension
        if self.debug:
            print(f"  Lift -> context shape: {context.shape}")
            print(f"  Lift -> depth_dist shape: {depth_dist.shape}")
            print(f"  Lift -> depth_dist stats: min={depth_dist.min():.4f}, max={depth_dist.max():.4f}, mean={depth_dist.mean():.4f}, sum_per_pixel={depth_dist.sum(dim=1).mean():.4f}")

        # 2. Get Geometry: Create frustum point cloud
        geometry = self.get_geometry(intrinsics, extrinsics, H, W) # B, 3, D, H, W
        if self.debug:
            print(f"  Geometry -> output shape: {geometry.shape}")
            print(f"  Geometry -> ego_points (x): min={geometry[:,0].min():.2f}, max={geometry[:,0].max():.2f}")
            print(f"  Geometry -> ego_points (y): min={geometry[:,1].min():.2f}, max={geometry[:,1].max():.2f}")
            print(f"  Geometry -> ego_points (z): min={geometry[:,2].min():.2f}, max={geometry[:,2].max():.2f}")

        # 3. Splat: Voxel pooling
        # Combine context and depth distribution
        feat_with_depth = depth_dist.unsqueeze(1) * context.unsqueeze(2) # B, C, D, H, W

        # Flatten for voxel pooling
        geom_flat = geometry.view(B, 3, -1) # B, 3, D*H*W
        feat_flat = feat_with_depth.view(B, C, -1) # B, C, D*H*W

        # Create BEV feature map
        bev_h, bev_w = self.y_grid.shape[0], self.x_grid.shape[0]
        bev_feat_map = torch.zeros((B, C, bev_h, bev_w), device=img_features.device)
        # This is a simplified splatting. A more efficient implementation would use scatter_add.
        for b in range(B):
            # Reverting to the original, correct axis mapping for splatting. The BEV grid's X-axis
            # corresponds to the ego's longitudinal (forward) direction.
            x_coords = ((geom_flat[b, 0] - self.grid_config['xbound'][0]) / self.grid_config['xbound'][2]).long()
            y_coords = ((geom_flat[b, 1] - self.grid_config['ybound'][0]) / self.grid_config['ybound'][2]).long()
            if self.debug:
                print(f"  Splat (batch {b}) -> x_coords range: [{x_coords.min()},{x_coords.max()}] (Grid W: {bev_w})")
                print(f"  Splat (batch {b}) -> y_coords range: [{y_coords.min()},{y_coords.max()}] (Grid H: {bev_h})")

            valid = (x_coords >= 0) & (x_coords < bev_w) & (y_coords >= 0) & (y_coords < bev_h)
            num_valid_points = valid.sum()
            if self.debug:
                total_points = valid.numel()
                print(f"  Splat (batch {b}) -> Valid points: {num_valid_points}/{total_points} ({num_valid_points/total_points:.2%})")

            # Convert 2D indices to 1D for the flattened BEV grid and add features
            if num_valid_points > 0:
                indices_1d = y_coords[valid] * bev_w + x_coords[valid]
                bev_feat_map[b].view(C, -1).index_add_(1, indices_1d, feat_flat[b, :, valid])
            elif self.debug:
                print(f"  Splat (batch {b}) -> WARNING: No valid points to splat.")

        final_bev_feat = self.bev_conv(bev_feat_map)
        if self.debug:
            print(f"Output bev_feat_map shape: {final_bev_feat.shape}")
            print(f"--- [ViewTransformerLSS Debug End] ---\n")
        return final_bev_feat