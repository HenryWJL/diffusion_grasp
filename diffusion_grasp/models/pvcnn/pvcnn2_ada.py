import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Optional, Union, Tuple
from einops import rearrange
import diffusion_grasp.models.pvcnn.functional as F
from diffusion_grasp.models.common import Attention


class SE3d(nn.Module):
    
    def __init__(
        self,
        channel: int,
        reduction: Optional[int] = 8
    ) -> None:
        """
        Args:
            channel: input channels.
            
            reduction: hidden dimension = channel / reduction
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
        
    def __repr__(self):
        return f"SE({self.channel}, {self.channel})" 
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)


class BallQuery(nn.Module):
    
    def __init__(
        self,
        radius: float,
        num_neighbors: int,
        include_coordinates: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    @custom_bwd
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)

    @custom_fwd(cast_inputs=torch.float32) 
    def forward(
        self,
        points_coords: torch.Tensor,
        centers_coords: torch.Tensor,
        points_features: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Args:
            points_coords: (B, 3, N)
            
            centers_coords: (B, 3, M)
            
            points_features: (B, D, N)
            
        Returns:
            neighbor_features: (B, D, M, K) or (B, 3+D, M, K)
        """
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)
        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else ''
        )


class AdaGN(nn.Module):
    """Adaptive group normalization"""
    def __init__(
        self,
        feat_dim: int,
        c_dim: int,
        num_groups: Optional[int] = 8
    ) -> None:
        """
        Args:
            feat_dim: the dimension of features.
            
            c_dim: the dimension of conditioning.
            
            num_groups: the number of groups to normalize in GroupNorm.
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, feat_dim)
        self.adaGN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * feat_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: point features (B, D, N) or voxel features (B, D, V, V, V)
            
            c: conditioning (B, C)
            
        Returns:
            (B, D, N) or (B, D, V, V, V)
        """
        assert c is not None, "Conditioning not found!"
        x = self.norm(x)
        c = self.adaGN_modulation(c)
        if len(x.shape) == 5:
            c = rearrange(c, 'b d -> b d 1 1 1')
        elif len(x.shape) == 4:
            c = rearrange(c, 'b d -> b d 1 1')
        elif len(x.shape) == 3:
            c = rearrange(c, 'b d -> b d 1')
        else:
            raise NotImplementedError
        scale, shift = c.chunk(2, dim=1)
        x = x * scale + shift
        return x
    
    
class SharedMLP(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Union[int, list, tuple],
        c_dim: Optional[int] = -1,
        data_dim: Optional[int] = 1
    ) -> None:
        """
        Args:
            in_channels: the dimension of input features.
            
            out_channels: int, list, or tuple of:
                the dimension of output features of intermediate layers.
            
            c_dim: the dimension of conditioning. If c_dim == -1, no conditioning.
            
            data_dim: 1 or 2. If 1, use Conv1d. Otherwise, use Conv2d.
        """
        super().__init__()
        self.with_adaGN = c_dim > 0
        assert data_dim == 1 or data_dim == 2, "data_dim must be 1 or 2!"
        conv_layer = nn.Conv1d if data_dim == 1 else nn.Conv2d
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        self.layers = nn.ModuleList()
        for oc in out_channels:
            self.layers.extend([
                conv_layer(in_channels, oc, 1),
                AdaGN(oc, c_dim) if self.with_adaGN else nn.GroupNorm(8, oc),
                nn.SiLU(),
            ])
            in_channels = oc

    def forward(self, *inputs) -> torch.Tensor:
        """
        Args:
            inputs: tuple of:
                features (B, D, N)
                conditioning (B, C) (optional)
            
        Returns:
            new features
        """
        if len(inputs) == 2:
            x, c = inputs
            if self.with_adaGN:
                assert c is not None, "No conditioning found!"
            for l in self.layers:
                if isinstance(l, AdaGN): 
                    x = l(x, c)
                else:
                    x = l(x)
            return x
        elif len(inputs) == 3:
            x, y, c = inputs
            if self.with_adaGN:
                assert c is not None, "No conditioning found!"
            for l in self.layers:
                if isinstance(l, AdaGN): 
                    x = l(x, c)
                else:
                    x = l(x)
            return x, y, c
        else:
            raise NotImplementedError
         
         
class Voxelization(nn.Module):
    
    def __init__(
        self,
        resolution: Optional[float] = 1.,
        normalize: Optional[bool] = True,
        eps: Optional[float] = 0.
    ) -> None:
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, D, N)
            
            coords: (B, 3, N)
            
        Returns:
            voxel features, voxel coordinates
        """
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


class PVConv(nn.Module):
    """Point-voxel convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        kernel_size: int,
        resolution: Optional[float] = 1., 
        add_point_feat: Optional[bool] = True,
        with_se: Optional[bool] = False, 
        with_attention: Optional[bool] = False,
        c_dim: Optional[int] = -1
    ) -> None:
        """
        Args:
            in_channels: the dimension of input features.
            
            out_channels: the dimension of output features.
            
            kernel_size: the kernel size of Conv3d in voxel layers.
            
            resolution: the resolution of voxels.
            
            add_point_feat: if True, return fused features. Otherwise, only return voxel features.
            
            with_se: if True, add SE3d layer to the end of voxel layers. 
            
            with_attention: if True, add self-attention to the end of PVConv.
            
            c_dim: the dimension of conditioning. If c_dim == -1, no conditioning.
        """
        super().__init__()
        self.resolution = resolution
        self.with_adaGN = c_dim > 0
        self.voxelization = Voxelization(resolution)
        self.voxel_layers = nn.ModuleList([
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            AdaGN(out_channels, c_dim) if self.with_adaGN else nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            AdaGN(out_channels, c_dim) if self.with_adaGN else nn.GroupNorm(8, out_channels),
            SE3d(out_channels) if with_se else nn.Identity()
        ])
        self.attn = Attention(out_channels) if with_attention else None
        self.point_conv = SharedMLP(
            in_channels,
            out_channels,
            c_dim
        ) if add_point_feat else None
        
    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  
        """
        Args: 
            inputs: tuple of: 
                features (B, D, N)
                coordinates (B, 3, N)
                conditioning (B, D)
                
        Returns:
            fused_features (B, D', N)
            coordinates (same as input coordinates)
            conditioning (same as input conditioning)
        """
        features, coords, condition = inputs 
        voxel_features, voxel_coords = self.voxelization(features, coords)
        for l in self.voxel_layers:
            if isinstance(l, AdaGN):
                voxel_features = l(voxel_features, condition)
            else:
                voxel_features = l(voxel_features)  
        voxel_features = F.trilinear_devoxelize(
            voxel_features,
            voxel_coords,
            self.resolution
        )
        fused_features = voxel_features 
        if self.point_conv is not None:
            fused_features += self.point_conv(features, condition)   
        if self.attn is not None:
            fused_features = self.attn(fused_features.transpose(2, 1)).transpose(2, 1) 
        return fused_features, coords, condition
    
    
    