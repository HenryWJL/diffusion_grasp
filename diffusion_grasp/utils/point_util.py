import torch
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat


def gather(
    x: Optional[torch.Tensor],
    idx: Optional[torch.Tensor]
    ):
    """gather elemtents using indices
    Args:
        x: (B, N, C)
        
        idx: (B, N, K)
        
    Returns:
        (B, N, K)
    """
    idx = torch.where(idx==-1, idx.shape[1], idx)
    idx = idx[..., None].repeat_interleave(x.shape[-1], -1)
    y = x[..., None, :].repeat_interleave(idx.shape[-2], -2).gather(1, idx)
    return y


def square_distance(
    p1: Optional[torch.Tensor],
    p2: Optional[torch.Tensor]
    ):
    """
    Args:
        p1: the xyz coordinates of a point cloud (B, N, 3)
        
        p2: the xyz coordinates of another point cloud (B, M, 3)
        
    Returns:
        p1 per point square distance w.r.t p2 (B, N, M)
    """
    B, N, _ = p1.shape
    M = p2.shape[1]
    p1 = p1.unsqueeze(-2).repeat(1, 1, M, 1)  # (B, N, M, 3)
    p2 = p2.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, M, 3)
    square_dist = torch.sum((p1 - p2) ** 2, dim=-1)
    return square_dist


def knn_points(
    p1: Optional[torch.Tensor],
    p2: Optional[torch.Tensor],
    K: Optional[int]
    ):
    """
    Args:
        p1: the xyz coordinates of a point cloud (B, N, 3)
        
        p2: the xyz coordinates of another point cloud (B, M, 3)
        
        K: the number of neighbors
        
    Returns:
        square distance between p1 and its K nearest neighbors in p2 (B, N, K)
        
        indices of p1's K nearest neighbors in p2 (B, N, K)
        
        p1's K nearest neighbors in p2 (B, N, K, 3)
    """
    dist = square_distance(p1, p2)
    neighbor_dist, neighbor_idx = torch.topk(dist, K, dim=-1, largest=False, sorted=False)
    k_neighbor = gather(p2, neighbor_idx)
    return neighbor_dist, neighbor_idx, k_neighbor


def set_ground_truth(
    grasp: Optional[torch.Tensor],
    grasp_gt: Optional[torch.Tensor]
    ):
    """Set ground-truth grasps using the nearest neighbor.
    Args:
        grasp: the predicted grasps (B, M, 7)
        
        grasp_gt: the ground-truth grasps (B, M, 7)

    Returns:
        grasp_gt: the assigned ground-truth grasp (B, M, 7)
    """
    center_xyz = grasp[..., :3]
    gt_center_xyz = grasp_gt[..., :3]
    _, neighbor_idx, _ = knn_points(
        p1=center_xyz, 
        p2=gt_center_xyz, 
        K=1
    )
    grasp_gt = gather(grasp_gt, neighbor_idx).squeeze()
    return grasp_gt


def pc_normalize(xyz: Optional[torch.Tensor]):
    """Normalize point cloud"""
    B, N, _ = xyz.shape  
    center_xyz = torch.mean(xyz, dim=-2).detach()
    center_xyz = repeat(center_xyz, 'b c -> b n c', n=N)
    dist = F.pairwise_distance(xyz.reshape(-1, 3), center_xyz.reshape(-1, 3))
    dist = rearrange(dist, '(b n) -> b n', b=B)
    dist_max = dist.max(-1)[0].detach()
    dist_max = repeat(dist_max, 'b -> b n c', n=N, c=3)
    xyz = (xyz - center_xyz) / dist_max
    return xyz