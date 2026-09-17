import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


class RGBPose3DDatasetImproved(Dataset):
    """
    改进的RGBPose3D数据集，修复坐标归一化和数据预处理问题
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

        # build transform (normalize to [-1,1])
        if self.copy_to_gpu:
            self.transform = Compose([
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])

        self.float_dtype = torch.float32

    def __len__(self):
        return len(self.indices)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img_t = self.transform(img)  # [C,H,W] in [-1,1]
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
        # 创建标准化的2D网格坐标，确保覆盖完整范围
        yy, xx = np.meshgrid(np.linspace(-1, 1, H, dtype=np.float32), 
                            np.linspace(-1, 1, W, dtype=np.float32), 
                            indexing='ij')
        zz = np.zeros_like(xx)
        grid = np.stack([xx, yy, zz], axis=-1)  # (H,W,3)
        
        # 应用旋转和平移
        R = self._euler_to_matrix(rot)
        T = trans.astype(np.float32)
        
        # 将网格坐标变换到3D空间
        coords = grid.reshape(-1, 3) @ R.T + T[None, :]
        
        # 确保坐标在合理范围内
        coords = coords.reshape(H, W, 3)
        
        # 检查坐标范围
        coords_flat = coords.reshape(-1, 3)
        print(f"    变换前坐标范围: X[{coords_flat[:, 0].min():.4f}, {coords_flat[:, 0].max():.4f}], "
              f"Y[{coords_flat[:, 1].min():.4f}, {coords_flat[:, 1].max():.4f}], "
              f"Z[{coords_flat[:, 2].min():.4f}, {coords_flat[:, 2].max():.4f}]")
        
        return coords

    def _normalize_coords_improved(self, coords_hw3: np.ndarray) -> np.ndarray:
        """
        改进的坐标归一化：
        1. 确保坐标在合理范围内
        2. 保持坐标的相对关系
        3. 避免除零错误
        4. 确保覆盖完整范围
        """
        c = coords_hw3.reshape(-1, 3)
        
        # 计算每个维度的统计信息
        mins = c.min(axis=0)
        maxs = c.max(axis=0)
        means = c.mean(axis=0)
        stds = c.std(axis=0)
        
        print(f"    归一化前统计: 均值={means}, 标准差={stds}, 范围=[{mins}, {maxs}]")
        
        # 使用min-max归一化确保覆盖完整范围
        ranges = maxs - mins
        ranges = np.clip(ranges, 1e-6, None)  # 避免除零
        
        # 归一化到[-1, 1]
        c_norm = 2.0 * (c - mins) / ranges - 1.0
        
        # 确保坐标在[-1,1]范围内
        c_norm = np.clip(c_norm, -1.0, 1.0)
        
        print(f"    归一化后范围: X[{c_norm[:, 0].min():.4f}, {c_norm[:, 0].max():.4f}], "
              f"Y[{c_norm[:, 1].min():.4f}, {c_norm[:, 1].max():.4f}], "
              f"Z[{c_norm[:, 2].min():.4f}, {c_norm[:, 2].max():.4f}]")
        
        return c_norm.reshape(coords_hw3.shape)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        gt_img = self._load_image(img_path)  # [C,H,W]
        _, H, W = gt_img.shape

        # optional center crop
        if self.crop_size is not None:
            ch, cw = self.crop_size, self.crop_size
            assert H >= ch and W >= cw, "crop_size larger than images"
            top = (H - ch) // 2
            left = (W - cw) // 2
            gt_img = gt_img[:, top:top+ch, left:left+cw]
            H, W = ch, cw

        # load pose row
        pose_row = self.pose_df.iloc[real_idx].values
        trans = pose_row[1:4].astype(np.float32)
        rot = pose_row[4:7].astype(np.float32)
        
        # 生成3D坐标
        coords3d_hw3 = self._build_coords3d_from_pose(H, W, trans, rot)
        coords3d_hw3 = self._normalize_coords_improved(coords3d_hw3)

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
