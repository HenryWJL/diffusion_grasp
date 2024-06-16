import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from graspnetAPI import GraspNet
from diffusion_grasp.utils.transforms import create_mat, mat2grasp

class GraspDataset(Dataset):
    
    def __init__(
        self,
        root_dir: Optional[str],
        num_points: Optional[int],
        num_grasps: Optional[int],
        split: Optional[str] = "train",
        with_width: Optional[bool] = True,
        with_score: Optional[bool] = False
        ) -> None:
        """
        Args:
            root_dir: the root directory of dataset.
            
            num_points: the number of points.
            
            num_grasps: the number of grasps.
            
            split: five options:
                "train": 100 scenes.
                "test": 90 scenes.
                "test_seen": 30 scenes.
                "test_similar": 30 scenes.
                "test_novel": 30 scenes.
            
            with_width: if True, also return gripper widths.
            
            with_score: if True, also return grasp quality scores.
        """
        super().__init__()
        # prepare
        self.with_score = with_score
        ROOT_DIR = Path(os.path.expanduser(root_dir)).absolute()
        load_path = ROOT_DIR.joinpath("simplified_data").joinpath(split).joinpath("data.npz")
        assert load_path.is_file(), f"\"{split}\" is not implemented!"
        # load raw data
        loader = np.load(str(load_path), allow_pickle=True)
        self.points = loader["points"]
        self.grasps = loader["grasps"]
        self.widths = loader["widths"]
        self.scores = loader["scores"]
        loader.close()
        self.grasps = mat2grasp(self.grasps)
        self.widths = self.widths[..., np.newaxis]
        if with_width:
            self.grasps = np.concatenate([self.grasps, self.widths], axis=-1)
        # sample points and grasps
        point_idx = np.arange(self.points.shape[0])
        np.random.shuffle(point_idx)
        self.points = self.points[point_idx[: num_points]]
        grasp_idx = np.arange(self.grasps.shape[0])
        np.random.shuffle(grasp_idx)
        self.grasps = self.grasps[grasp_idx[: num_grasps]]
        self.scores = self.scores[grasp_idx[: num_grasps]]
        # convert to torch.Tensor
        self.points = torch.from_numpy(self.points).float()
        self.grasps = torch.from_numpy(self.grasps).float()
        self.scores = torch.from_numpy(self.scores).float()
        
    def __len__(self):
        return len(self.grasps)
    
    def __getitem__(self, index):
        if self.with_score:
            return self.points[index], self.grasps[index], self.scores[index]
        else:
            return self.points[index], self.grasps[index]

def create_dataloader(cfg: Optional[OmegaConf]) -> DataLoader:
    dataset = GraspDataset(
        cfg.dataset.root_dir,
        cfg.pvcnn.num_points,
        cfg.vae.num_grasps,
        cfg.dataset.split
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=1
    )
    return dataloader



# class GraspDataset(Dataset):
    
#     def __init__(
#         self,
#         root_dir: Optional[str],
#         num_points: Optional[int],
#         num_grasps: Optional[int],
#         split: Optional[str] = "train",
#         camera: Optional[str] = "realsense",
#         fric_threshold: Optional[float] = 0.1,
#         with_augment: Optional[bool] = True,
#         with_width: Optional[bool] = True,
#         with_score: Optional[bool] = False
#         ) -> None:
#         """
#         Args:
#             root_dir: the root directory of dataset.
            
#             num_points: the number of points.
            
#             num_grasps: the number of grasps.
            
#             split: five options:
#                 "train": 100 scenes.
#                 "test": 90 scenes.
#                 "test_seen": 30 scenes.
#                 "test_similar": 30 scenes.
#                 "test_novel": 30 scenes.
            
#             camera: "kinect" or "realsense".
            
#             fric_threshold: only grasps with friction coefficients < fric_threshold will be loaded.
            
#             with_augment: if True, add data augmentation.
            
#             with_width: if True, also return gripper widths.
            
#             with_score: if True, also return grasp quality scores.
#         """
#         super().__init__()
#         self.ROOT_DIR = Path(os.path.expanduser(root_dir)).absolute()
#         self.loader = GraspNet(str(self.ROOT_DIR), camera=camera, split=split)
#         if split == "train":
#             self.sceneIds = list(range(100))
#         elif split == "test":
#             self.sceneIds = list(range(100, 190))
#         elif split == "test_seen":
#             self.sceneIds = list(range(100, 130))
#         elif split == "test_similar":
#             self.sceneIds = list(range(130, 160))
#         elif split == "test_novel":
#             self.sceneIds = list(range(160, 190))
#         else:
#             raise NotImplementedError(f"\"{split}\" is not implemented!")
#         # self.sceneIds = [f"scene_{str(id).zfill(4)}" for id in self.sceneIds]
#         self.camera = camera
#         self.num_points = num_points
#         self.num_grasps = num_grasps
#         self.fric_threshold = fric_threshold
#         self.with_width = with_width
#         self.with_score = with_score
#         self.with_augment = with_augment
#         self.points = None
#         self.grasps = None
#         self.widths = None
#         self.scores = None
#         self.load()
        
#     def load(self):
#         points = []
#         grasps = []  
#         widths = []
#         scores = []  
#         for sid in tqdm(self.sceneIds, desc="Loading data..."):
#             for aid in range(256):
#                 # get point clouds
#                 point, _ = self.loader.loadScenePointCloud(
#                     sceneId=sid,
#                     annId=aid,
#                     camera=self.camera,
#                     format="numpy"
#                 )
#                 # sample point clouds
#                 point_idx = np.arange(point.shape[0])
#                 np.random.shuffle(point_idx)
#                 point = point[point_idx[: self.num_points]]  # (num_points, 3)
#                 # get grasps and scores
#                 grasp = self.loader.loadGrasp(
#                     sceneId=sid,
#                     annId=aid,
#                     format="6d",
#                     camera=self.camera,
#                     fric_coef_thresh=self.fric_threshold
#                 )
#                 # sample grasps
#                 if self.num_grasps >= len(grasp):
#                     print("Skip!")
#                     continue
#                 grasp_idx = np.arange(len(grasp))
#                 np.random.shuffle(grasp_idx)
#                 trans = grasp.translations[grasp_idx[: self.num_grasps]]
#                 rot_mat = grasp.rotation_matrices[grasp_idx[: self.num_grasps]]
#                 width = grasp.widths[grasp_idx[: self.num_grasps]]
#                 score = grasp.scores[grasp_idx[: self.num_grasps]]
#                 grasp = create_mat(trans, rot_mat)  # (num_grasps, 4, 4)
#                 if self.with_augment:
#                     point, grasp = self.augment(point, grasp)  
#                 points.append(point)
#                 grasps.append(grasp)
#                 widths.append(width)
#                 scores.append(score)
#                 print(f"{(sid + 1) * (aid + 1)} data are processed.")
        
#         self.points = np.array(points)
#         self.grasps = np.array(grasps)
#         self.widths = np.array(widths)
#         self.scores = np.array(scores)
#         np.savez_compressed(
#             str(self.ROOT_DIR.joinpath("simplified_data/train/data.npz")),
#             points=self.points,
#             grasps=self.grasps,
#             widths = self.widths,
#             scores=self.scores
#         )        
#         # self.points = np.array(points)
#         # self.grasps = mat2grasp(np.array(grasps))
#         # self.widths = np.array(widths)
#         # self.scores = torch.from_numpy(np.array(scores)).float()
#         # if self.with_width:
#         #     self.grasps = np.concatenate([self.grasps, self.widths[..., np.newaxis]], axis=-1)
#         # self.points = torch.from_numpy(self.points).float()
#         # self.grasps = torch.from_numpy(self.grasps).float()

#     def augment(
#         self,
#         points: Optional[np.ndarray],
#         grasps: Optional[np.ndarray]
#         ):
#         # Flipping along Y-Z plane
#         if np.random.random() > 0.5:
#             flip_mat = np.array([
#                 [-1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1]
#             ])
#             flip_mat = flip_mat[np.newaxis, ...]
#             points = np.matmul(flip_mat, points[..., np.newaxis]).squeeze()
#             grasps = np.matmul(create_mat(np.zeros((1, 3)), flip_mat), grasps)
#         # Rotating along Z-axis
#         rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
#         c, s = np.cos(rot_angle), np.sin(rot_angle)
#         rot_mat = np.array([
#             [1, 0, 0],
#             [0, c, -s],
#             [0, s, c]
#         ])
#         rot_mat = rot_mat[np.newaxis, ...]
#         points = np.matmul(rot_mat, points[..., np.newaxis]).squeeze()
#         grasps = np.matmul(create_mat(np.zeros((1, 3)), rot_mat), grasps)
#         return points, grasps
        
#     def __len__(self):
#         return len(self.grasps)
    
#     def __getitem__(self, index):
#         if self.with_score:
#             return self.points[index], self.grasps[index], self.scores[index]
#         else:
#             return self.points[index], self.grasps[index]

if __name__ == "__main__":
    # d = GraspDataset(
    #     root_dir="/home/wangjunlin/project/graspnet-1billion",
    #     num_points=15000,
    #     num_grasps=2048,
    #     with_augment=False,
    #     with_score=True
    # )
    d = GraspDataset(
        root_dir="/home/wangjunlin/project/graspnet-1billion",
        num_points=2048,
        num_grasps=1024,
        with_score=True
    )
    # np.savez_compressed(
    #     str("/home/wangjunlin/project/graspnet-1billion/simplified_data/train.npz"),
    #     points=np.random.random((10, 2048, 3)),
    #     grasps=np.random.random((10, 1024, 4, 4)),
    #     widths = np.random.random((10, 1024)),
    #     scores=np.random.random((10, 1024))
    # )
    # loader = np.load("/home/wangjunlin/project/graspnet-1billion/simplified_data/train/data.npz", allow_pickle=True)
    # points = loader["points"]
    # grasps = loader["grasps"]
    # widths = loader["widths"]
    # print(points.shape)
    # print(grasps.shape)
    # print(widths.shape)
    # loader.close()
    
    
    
    
    