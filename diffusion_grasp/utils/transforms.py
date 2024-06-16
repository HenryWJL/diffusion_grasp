import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation as R


def create_mat(trans: Optional[np.ndarray], rot_mat: Optional[np.ndarray]):
    """Create transformation matrices.
    Args:
        trans: translation vectors (N, 3).
    
        rot_mat: rotation matrices (N, 3, 3)
    
    Returns:
        transformation matrices (N, 4, 4)
    """
    trans_mat = np.concatenate([rot_mat, trans[..., np.newaxis]], axis=-1)
    pad = np.array([0, 0, 0, 1])[np.newaxis, np.newaxis, ...].repeat(len(trans_mat), 0)
    trans_mat = np.concatenate([trans_mat, pad], axis=-2)
    return trans_mat

# def rot2quat(rot_mat: Optional[np.ndarray]):
#     """Transform rotation matrices to quaternions.
#     Args:
#         rot_mat: (..., 3, 3)
    
#     Returns:
#         unit quaternions (..., 4)
#     """
#     trace = rot_mat[..., 0, 0] + rot_mat[..., 1, 1] + rot_mat[..., 2, 2]
#     w = np.sqrt(trace + 1) / 2
#     x = (rot_mat[..., 2, 1] - rot_mat[..., 1, 2]) / (4 * w)
#     y = (rot_mat[..., 0, 2] - rot_mat[..., 2, 0]) / (4 * w)
#     z = (rot_mat[..., 1, 0] - rot_mat[..., 0, 1]) / (4 * w)
#     quat = np.stack([w, x, y, z], axis=-1)
#     norm = np.linalg.norm(quat, axis=-1)[..., np.newaxis]
#     quat = quat / norm
#     return quat

def rot2mrp(rot_mat: Optional[np.ndarray]):
    """Transform rotation matrices to Modified Rodrigues Parameters (MRP).
    Args:
        rot_mat: (..., 3, 3)
    
    Returns:
        Modified Rodrigues Parameters (..., 3)
    """
    if len(rot_mat.shape) == 4:
        B, N, _, _ = rot_mat.shape
        rot_mat = rot_mat.reshape(-1, 3, 3)
        mrp = R.from_matrix(rot_mat).as_mrp().reshape(B, N, 3)
    elif len(rot_mat.shape) == 3:
        mrp = R.from_matrix(rot_mat).as_mrp()
    else:
        raise NotImplementedError
    return mrp


def mat2grasp(trans_mat: Optional[np.ndarray]):
    """Transform transformation matrices to 6-DoF grasps."""
    trans = trans_mat[..., :3, 3]
    rot_mat = trans_mat[..., :3, :3]
    mrp = rot2mrp(rot_mat)
    grasp = np.concatenate([trans, mrp], axis=-1)
    return grasp