import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose


class RGBPose3DDatasetGlobalNorm(Dataset):
    """
    Dataset for multi-frame 2D images with global normalization across all frames.
    
    关键改进：使用全局最小值/最大值归一化所有样本，而不是per-sample归一化。
    这样可以保证不同帧之间的坐标一致性，有助于网络学习统一的3D表示。
    """

    def __init__(self, images_dir: str, pose_file: str, copy_to_gpu: bool = False, grayscale: bool = True,
                 crop_size: int = None, angles_in_degrees: bool = False, mode: str = 'train'):
        super().__init__()
        self.images_dir = images_dir
        self.pose_file = pose_file
        self.copy_to_gpu = copy_to_gpu
        self.grayscale = grayscale
        self.crop_size = crop_size
        self.angles_in_degrees = angles_in_degrees
        self.mode = mode.lower()
        assert self.mode in ['train', 'test'], "mode should be 'train' or 'test'"

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*')))
        assert len(self.image_paths) > 0, f"No images found in {self.images_dir}"

        # load poses
        assert os.path.isfile(self.pose_file), f"Pose file not found: {self.pose_file}"
        self.pose_df = pd.read_excel(self.pose_file)
        assert len(self.pose_df) == len(self.image_paths), "Number of images and poses must match"

        # split indices: every 5th image to test set (i % 5 == 4)
        self.indices = []
        for i in range(len(self.image_paths)):
            if self.mode == 'train' and i % 5 != 4:
                self.indices.append(i)
            elif self.mode == 'test' and i % 5 == 4:
                self.indices.append(i)

        # build transform
        self.transform = Compose([ToTensor()])

        # 预计算全局归一化参数
        print(f"[{self.mode}] 计算全局归一化参数...")
        self._compute_global_normalization()
        print(f"[{self.mode}] 全局坐标范围: min={self.global_min}, max={self.global_max}")
        
        self.float_dtype = torch.float32

    def _compute_global_normalization(self):
        """预计算所有样本的全局最小值和最大值用于归一化"""
        all_mins = []
        all_maxs = []
        
        # 只需要检查训练集/测试集中的样本
        sample_indices = self.indices[::max(1, len(self.indices)//20)]  # 采样20个样本
        
        for idx in sample_indices:
            pose_row = self.pose_df.iloc[idx].values
            trans = pose_row[1:4].astype(np.float32)
            rot = pose_row[4:7].astype(np.float32)
            
            # 获取一个样本图像的尺寸
            if len(all_mins) == 0:
                img = Image.open(self.image_paths[idx])
                W, H = img.size
                if self.crop_size is not None:
                    # crop_size 可能是 int 或 [H, W] 列表
                    if isinstance(self.crop_size, (list, tuple)):
                        H, W = self.crop_size
                    else:
                        H = W = self.crop_size
            
            coords3d = self._build_coords3d_from_pose(H, W, trans, rot)
            all_mins.append(coords3d.min())
            all_maxs.append(coords3d.max())
        
        self.global_min = min(all_mins)
        self.global_max = max(all_maxs)

    def __len__(self):
        return len(self.indices)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img_t = self.transform(img)  # [C,H,W] in [0,1]
        return img_t

    def _euler_to_matrix(self, angles: np.ndarray) -> np.ndarray:
        rx, ry, rz = angles.tolist()
        if self.angles_in_degrees:
            rx = np.deg2rad(rx)
            ry = np.deg2rad(ry)
            rz = np.deg2rad(rz)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        return R

    def _build_coords3d_from_pose(self, H: int, W: int, trans: np.ndarray, rot: np.ndarray) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
        zz = np.zeros_like(xx)
        grid = np.stack([xx, yy, zz], axis=-1)  # (H,W,3)
        R = self._euler_to_matrix(rot)
        T = trans.astype(np.float32)
        coords = grid.reshape(-1, 3) @ R.T + T[None, :]
        return coords.reshape(H, W, 3)

    def _normalize_coords_global(self, coords_hw3: np.ndarray) -> np.ndarray:
        """使用全局最小值/最大值归一化到[-1,1]"""
        c = coords_hw3.reshape(-1, 3)
        span = np.clip(self.global_max - self.global_min, 1e-8, None)
        c_norm = 2.0 * (c - self.global_min) / span - 1.0
        return c_norm.reshape(coords_hw3.shape)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        gt_img = self._load_image(img_path)  # [C,H,W]
        _, H, W = gt_img.shape

        # optional center crop
        if self.crop_size is not None:
            # crop_size 可能是 int 或 [H, W] 列表
            if isinstance(self.crop_size, (list, tuple)):
                ch, cw = self.crop_size
            else:
                ch = cw = self.crop_size
            assert H >= ch and W >= cw, "crop_size larger than images"
            top = (H - ch) // 2
            left = (W - cw) // 2
            gt_img = gt_img[:, top:top+ch, left:left+cw]
            H, W = ch, cw

        # load pose
        pose_row = self.pose_df.iloc[real_idx].values
        trans = pose_row[1:4].astype(np.float32)
        rot = pose_row[4:7].astype(np.float32)
        coords3d_hw3 = self._build_coords3d_from_pose(H, W, trans, rot)
        
        # 使用全局归一化
        coords3d_hw3 = self._normalize_coords_global(coords3d_hw3)

        # flatten to (1, H*W, 3)
        coords_flat = torch.from_numpy(coords3d_hw3.reshape(-1, 3)).to(self.float_dtype).unsqueeze(0)

        segments = torch.zeros(H * W, dtype=torch.long)
        dino = torch.zeros(1, dtype=self.float_dtype)

        sample = {
            'coords': coords_flat,
            'gt_img': gt_img,
            'segments': segments,
            'dino': dino,
        }
        return sample

