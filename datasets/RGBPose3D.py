import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose


class RGBPose3DDataset(Dataset):
    """
    Dataset for multi-frame 2D images where each pixel has a corresponding 3D coordinate.

    Expected directory layout (example, adjustable via args):
      - images_dir: directory with images (png/jpg). For images 'frame_0001.png'
      - coords3d_dir: directory with per-images 3D coords saved as .npy shaped (H, W, 3)
                       using the same stem name, e.g. 'frame_0001.npy'.

    Returns per-sample dict keys compatible with Neural-Experts training loop:
      - 'coords': torch.float32 tensor with shape (1, H*W, 3)
      - 'gt_img': torch.float32 tensor with shape (C, H, W) in [0,1]
      - 'segments': torch.long tensor with shape (H*W,) (placeholder if not used)
      - 'dino': torch.float32 placeholder tensor (not used by default)
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

        # load poses (expects columns: [id, tx, ty, tz, rx, ry, rz]) per MyMoE/newdataset.py usage
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

        # build transform (normalize to [0,1])
        if self.copy_to_gpu:
            # caller is responsible to move to device later; keep CPU tensors here
            self.transform = Compose([
                ToTensor()
            ])
        else:
            self.transform = Compose([
                ToTensor()
            ])

        # pre-compute dtypes
        self.float_dtype = torch.float32

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
        # angles: (3,) in radians by default; set angles_in_degrees=True if needed
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

    def _normalize_coords(self, coords_hw3: np.ndarray) -> np.ndarray:
        # Global min-max normalization to [-1,1] to match MyMoE approach
        c = coords_hw3.reshape(-1, 3)
        min_val = c.min()  # Global minimum (scalar)
        max_val = c.max()  # Global maximum (scalar)
        # avoid division by zero
        span = np.clip(max_val - min_val, 1e-8, None)
        c_norm = 2.0 * (c - min_val) / span - 1.0
        return c_norm.reshape(coords_hw3.shape)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        gt_img = self._load_image(img_path)  # [C,H,W]
        _, H, W = gt_img.shape

        # optional center crop to crop_size to match your preprocessing
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

        # load pose row (match by index order)
        pose_row = self.pose_df.iloc[real_idx].values
        trans = pose_row[1:4].astype(np.float32)
        rot = pose_row[4:7].astype(np.float32)
        coords3d_hw3 = self._build_coords3d_from_pose(H, W, trans, rot)
        coords3d_hw3 = self._normalize_coords(coords3d_hw3)

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


