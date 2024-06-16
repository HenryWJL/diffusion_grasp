import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from functools import partial
import diffusion_grasp.models.pvcnn.functional as F
from diffusion_grasp.models.pvcnn.pvcnn2_ada import (
    BallQuery,
    SharedMLP,
    PVConv
)


class PointNetAModule(nn.Module):
    """PointNet abstraction layer"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_coordinates: Optional[bool] = True,
        c_dim: Optional[bool] = -1
        ):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]
        mlps = []
        total_out_channels = 0
        for _out_channels in out_channels:
            mlps.append(
                SharedMLP(
                    in_channels=in_channels + (3 if include_coordinates else 0),
                    out_channels=_out_channels,
                    c_dim=c_dim
                )
            )
            total_out_channels += _out_channels[-1]
        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            inputs: a tuple of: 
                features (B, D, N)
                coordinates (B, 3, N)
                conditioning (B, D)
                
        Returns:
            features
            coordinates
        """
        features, coords, condition = inputs
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(
                    mlp(features, condition).max(dim=-1, keepdim=True).values
                )
            return torch.cat(features_list, dim=1), coords
        else:
            return self.mlps[0](features, condition).max(dim=-1, keepdim=True).values, coords

    def extra_repr(self):
        return f'out_channels={self.out_channels}, include_coordinates={self.include_coordinates}'
    
    
class PointNetSAModule(nn.Module):
    """PointNet set abstraction layer"""
    def __init__(
        self,
        num_centers: Union[float, list, tuple],
        radius: Union[float, list, tuple],
        num_neighbors: Union[float, list, tuple],
        in_channels: int,
        out_channels: Union[int, list, tuple],
        include_coordinates: Optional[bool] = True,
        c_dim: Optional[int] = -1
    ) -> None:
        super().__init__()
        self.with_adaGN = c_dim > 0
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(
                    radius=_radius,
                    num_neighbors=_num_neighbors, 
                    include_coordinates=include_coordinates
                )
            )
            mlps.append(
                SharedMLP(
                    in_channels=in_channels + (3 if include_coordinates else 0),
                    out_channels=_out_channels,
                    c_dim=c_dim,
                    data_dim=2
                )
            )
            total_out_channels += _out_channels[-1]
        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            inputs: a tuple of: 
                features (B, D, N)
                coordinates (B, 3, N)
                conditioning (B, D)
                
        Returns:
            features
            sampled center coordinates
        """
        features, coords, condition = inputs 
        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        features_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouper_output = grouper(coords, centers_coords, features)
            features_list.append(
                mlp(
                    grouper_output,
                    condition
                ).max(dim=-1).values
            )
        if len(features_list) > 1:
            return torch.cat(features_list, dim=1), centers_coords, condition
        else:
            return features_list[0], centers_coords, condition

    def extra_repr(self):
        return f'num_centers={self.num_centers}, out_channels={self.out_channels}'


class PointNetFPModule(nn.Module):
    """PointNet feature propagation layer"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Union[int, list, tuple],
        c_dim: Optional[int] = -1
        ):
        super().__init__()
        self.with_adaGN = c_dim > 0
        self.mlp = SharedMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            c_dim=c_dim
        )

    def forward(self, inputs: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(inputs) == 4:
            points_coords, centers_coords, centers_features, condition = inputs 
            points_features = None
        elif len(inputs) == 5:
            points_coords, centers_coords, centers_features, points_features, condition = inputs 
        else:
            raise NotImplementedError
        interpolated_features = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        if points_features is not None:
            interpolated_features = torch.cat([interpolated_features, points_features], dim=1)
        interpolated_features = self.mlp(interpolated_features, condition)
        return interpolated_features, points_coords, condition
    
    
def _linear_gn_relu(in_channels: int, out_channels: int) -> nn.Module:
    module = nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8,out_channels),
        nn.SiLU()
    )
    return module


def create_mlp_components(
    in_channels: int,
    out_channels: int,
    classifier: Optional[bool] = False,
    dim: Optional[int] = 2,
    width_multiplier: Optional[int] = 1,
    c_dim: Optional[int] = -1
) -> Tuple[list, int]:
    r = width_multiplier
    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc, c_dim=c_dim))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet2_sa_components(
    sa_blocks: Union[tuple, list],
    extra_feature_channels: int,   
    input_dim: Optional[int] = 3, 
    use_att: Optional[bool] = False,
    force_att: Optional[int] = 0,
    with_se: Optional[bool] = False,
    width_multiplier: Optional[int] = 1,
    voxel_resolution_multiplier: Optional[int] = 1,
    c_dim: Optional[int] = -1
) -> Tuple[list, int, int, int]:
    """
    Returns: 
        in_channels: the last output channels of the sa blocks.
    """
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + input_dim 
    sa_layers, sa_in_channels = [], []
    c = 0
    num_centers = None
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = ((c+1) % 2 == 0 and use_att and p == 0) or (force_att and c > 0)
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = partial(
                        PVConv,
                        kernel_size=3, 
                        resolution=int(vr * voxel_resolution),
                        with_attention=attention,
                        with_se=with_se,
                        c_dim=c_dim
                    )
                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels, c_dim=c_dim))
                elif k ==0:
                    sa_blocks.append(block(in_channels, out_channels, c_dim=c_dim))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        if sa_configs is not None:
            num_centers, radius, num_neighbors, out_channels = sa_configs
            _out_channels = []
            for oc in out_channels:
                if isinstance(oc, (list, tuple)):
                    _out_channels.append([int(r * _oc) for _oc in oc])
                else:
                    _out_channels.append(int(r * oc))
            out_channels = _out_channels
            if num_centers is None:
                block = PointNetAModule
            else:
                block = partial(
                    PointNetSAModule,
                    num_centers=num_centers,
                    radius=radius,
                    num_neighbors=num_neighbors
                )
            sa_blocks.append(
                block(
                    c_dim=c_dim,
                    in_channels=extra_feature_channels, 
                    out_channels=out_channels,
                    include_coordinates=True
                    )
                )
            in_channels = extra_feature_channels = sa_blocks[-1].out_channels 
        c += 1
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(
    fp_blocks: Union[tuple, list],
    in_channels: int,
    sa_in_channels: int,
    use_att: Optional[bool] = False,
    with_se: Optional[bool] = False,
    width_multiplier: Optional[int] = 1,
    voxel_resolution_multiplier: Optional[int] = 1,
    c_dim: Optional[int] = -1
) -> Tuple[list, int]:
    r, vr = width_multiplier, voxel_resolution_multiplier
    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(
                in_channels=in_channels + sa_in_channels[-1 - fp_idx], 
                out_channels=out_channels,
                c_dim=c_dim
            )
        )
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c+1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = partial(SharedMLP, c_dim=c_dim)
                else:
                    block = partial(
                        PVConv,
                        kernel_size=3, 
                        resolution=int(vr * voxel_resolution),
                        with_attention=attention,
                        with_se=with_se, 
                        c_dim=c_dim
                    )
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))
        c += 1
    return fp_layers, in_channels
    