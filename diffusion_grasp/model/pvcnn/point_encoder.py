import torch
import torch.nn as nn
from typing import Optional, Union
from diffusion_grasp.diffusion_grasp.model.dit import Attention
from diffusion_grasp.model.pvcnn.pvcnn2_ada import SharedMLP
from diffusion_grasp.model.pvcnn.pointnet import (
    create_mlp_components,
    create_pointnet2_sa_components,
    create_pointnet2_fp_modules
)


class PVCNN2Unet(nn.Module):

    def __init__(
        self, 
        sa_blocks: list,
        fp_blocks: list,
        final_dim: int,
        input_dim: Optional[int] = 3,
        extra_feature_channels: Optional[int] = 0, 
        use_att: Optional[bool] = True,
        c_dim: Optional[int] = -1 
    ) -> None:
        super().__init__()
        self.input_dim = input_dim 
        self.sa_blocks = sa_blocks 
        self.fp_blocks = fp_blocks
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            input_dim=input_dim,
            sa_blocks=self.sa_blocks, 
            extra_feature_channels=extra_feature_channels, 
            with_se=True, 
            use_att=use_att,
            c_dim=c_dim
        )
        self.sa_layers = nn.ModuleList(sa_layers)
        self.global_att = None if not use_att else Attention(channels_sa_features)
        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels + input_dim - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features, 
            sa_in_channels=sa_in_channels, 
            with_se=True,
            use_att=use_att,
            c_dim=c_dim 
        )
        self.fp_layers = nn.ModuleList(fp_layers)
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features, 
            out_channels=[128, 0.1, final_dim],
            classifier=True, 
            c_dim=c_dim
        )
        self.classifier = nn.ModuleList(layers)

    def forward(self, *inputs) -> torch.Tensor:
        input, condition = inputs
        coords = input[:, :self.input_dim, :].contiguous() 
        features = input 
        coords_list, in_features_list = [], []
        for _, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, _ = sa_blocks((features, coords, condition)) 
        in_features_list[0] = input[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features.transpose(2, 1)).transpose(2, 1)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, _ = fp_blocks((
                coords_list[-1-fp_idx],
                coords, 
                features, 
                in_features_list[-1-fp_idx],
                condition
            ))
        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features, condition)
            else:
                features = l(features)
        return features 


class PointTransPVC(nn.Module):
    """Point cloud encoder"""
    def __init__(
        self,
        final_dim: int,
        extra_feature_dim: Optional[int] = 0,
        c_dim = -1
    ) -> None:
        """
        Args:
            final_dim: the dimension of the final point-wise features.
            
            extra_feature_dim: the dimension of extra features (in addition to coordinates).
            
            c_dim: the dimension of conditioning.
        """
        super().__init__()
        self.sa_blocks = [
            ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
            ((64, 3, 16), (256, 0.2, 32, (64, 128))),
            ((128, 3, 8), (64, 0.4, 32, (128, 256))),
            (None, (16, 0.8, 32, (128, 128, 128))), 
        ]
        self.fp_blocks = [
            ((128, 128), (128, 3, 8)),
            ((128, 128), (128, 3, 8)),
            ((128, 128), (128, 2, 16)),
            ((128, 128, 64), (64, 2, 32)),
        ]
        self.layers = PVCNN2Unet(
            sa_blocks=self.sa_blocks,
            fp_blocks=self.fp_blocks,
            final_dim=final_dim,
            extra_feature_channels=extra_feature_dim,
            c_dim=c_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, (3 + extra_feature_dim))
            
            condition: (B, D)
            
        Returns:
            (B, final_dim)
        """
        output = self.layers(x.transpose(2, 1), condition).transpose(2, 1).max(-2)[0]
        return output