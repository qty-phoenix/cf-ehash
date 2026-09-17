"""
文本引导的RGBPose3D数据集
为每个图像/数据集添加文本描述，用于引导路由机制
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose


class RGBPose3DTextGuidedDataset(Dataset):
    """
    支持文本描述的RGBPose3D数据集
    
    文本描述可以是：
    1. 全局描述：整个数据集一个描述
    2. 每个图像一个描述：存储在JSON文件中
    
    目录结构示例：
      - images_dir: 图像目录
      - pose_file: 姿态文件 (Excel)
      - text_descriptions: JSON文件或单个字符串
          JSON格式: {"image_0001.png": "description1", "image_0002.png": "description2", ...}
          或全局: "This dataset contains brain MRI scans..."
    """
    
    def __init__(self, images_dir: str, pose_file: str, 
                 text_descriptions=None,
                 text_description_file: str = None,
                 copy_to_gpu: bool = False, 
                 grayscale: bool = True,
                 crop_size: int = None, 
                 angles_in_degrees: bool = False, 
                 mode: str = 'train',
                 use_tokenizer: bool = False,
                 tokenizer=None,
                 max_text_length: int = 77):
        super().__init__()
        self.images_dir = images_dir
        self.pose_file = pose_file
        self.copy_to_gpu = copy_to_gpu
        self.grayscale = grayscale
        self.crop_size = crop_size
        self.angles_in_degrees = angles_in_degrees
        self.mode = mode.lower()
        self.use_tokenizer = use_tokenizer
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        assert self.mode in ['train', 'test'], "mode should be 'train' or 'test'"
        
        # 加载图像路径
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, '*')))
        assert len(self.image_paths) > 0, f"No images found in {self.images_dir}"
        
        # 加载姿态数据
        assert os.path.isfile(self.pose_file), f"Pose file not found: {self.pose_file}"
        self.pose_df = pd.read_excel(self.pose_file)
        assert len(self.pose_df) == len(self.image_paths), "Number of images and poses must match"
        
        # 加载文本描述
        self.text_descriptions_dict = {}
        self.global_text_description = None
        
        if text_descriptions is not None:
            # 直接提供的文本描述（字符串）
            if isinstance(text_descriptions, str):
                self.global_text_description = text_descriptions
            elif isinstance(text_descriptions, dict):
                self.text_descriptions_dict = text_descriptions
        elif text_description_file is not None:
            # 从文件加载文本描述
            if text_description_file.endswith('.json'):
                with open(text_description_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if 'global' in data:
                            self.global_text_description = data['global']
                        else:
                            self.text_descriptions_dict = data
                    elif isinstance(data, str):
                        self.global_text_description = data
            elif text_description_file.endswith('.txt'):
                # 单个文本文件作为全局描述
                with open(text_description_file, 'r', encoding='utf-8') as f:
                    self.global_text_description = f.read().strip()
        
        # 如果既没有全局描述也没有per-image描述，使用默认描述
        if self.global_text_description is None and len(self.text_descriptions_dict) == 0:
            self.global_text_description = "A multi-view image dataset with 3D pose information."
            print(f"Warning: No text descriptions provided. Using default: {self.global_text_description}")
        
        # 数据集划分
        self.indices = []
        for i in range(len(self.image_paths)):
            if self.mode == 'train' and i % 5 != 4:
                self.indices.append(i)
            elif self.mode == 'test' and i % 5 == 4:
                self.indices.append(i)
        
        # 图像变换
        self.transform = Compose([ToTensor()])
        self.float_dtype = torch.float32
        
        print(f"Loaded {len(self)} {mode} samples")
        if self.global_text_description:
            print(f"Global text description: {self.global_text_description[:100]}...")
        else:
            print(f"Per-image text descriptions: {len(self.text_descriptions_dict)} descriptions")

    def __len__(self):
        return len(self.indices)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img_t = self.transform(img)
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
        grid = np.stack([xx, yy, zz], axis=-1)
        R = self._euler_to_matrix(rot)
        T = trans.astype(np.float32)
        coords = grid.reshape(-1, 3) @ R.T + T[None, :]
        return coords.reshape(H, W, 3)

    def _normalize_coords(self, coords_hw3: np.ndarray) -> np.ndarray:
        c = coords_hw3.reshape(-1, 3)
        min_val = c.min()
        max_val = c.max()
        span = np.clip(max_val - min_val, 1e-8, None)
        c_norm = 2.0 * (c - min_val) / span - 1.0
        return c_norm.reshape(coords_hw3.shape)
    
    def _get_text_description(self, img_path: str) -> str:
        """获取图像对应的文本描述"""
        if self.global_text_description is not None:
            return self.global_text_description
        
        # 尝试从字典中查找
        img_name = os.path.basename(img_path)
        if img_name in self.text_descriptions_dict:
            return self.text_descriptions_dict[img_name]
        
        # 尝试不带扩展名的名称
        img_stem = os.path.splitext(img_name)[0]
        if img_stem in self.text_descriptions_dict:
            return self.text_descriptions_dict[img_stem]
        
        # 默认描述
        return f"Image {img_name}"
    
    def _tokenize_text(self, text: str):
        """将文本转换为token IDs"""
        if self.tokenizer is not None:
            # 使用提供的tokenizer
            return self.tokenizer(text, max_length=self.max_text_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
        else:
            # 简单的字符级tokenization（用于SimpleTextEncoder）
            # 这里返回原始文本，在collate_fn中统一处理
            return text

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        
        # 加载图像
        gt_img = self._load_image(img_path)
        _, H, W = gt_img.shape
        
        # 可选裁剪
        if self.crop_size is not None:
            if isinstance(self.crop_size, (list, tuple)):
                ch, cw = self.crop_size
            else:
                ch = cw = self.crop_size
            assert H >= ch and W >= cw, "crop_size larger than images"
            top = (H - ch) // 2
            left = (W - cw) // 2
            gt_img = gt_img[:, top:top+ch, left:left+cw]
            H, W = ch, cw
        
        # 加载姿态
        pose_row = self.pose_df.iloc[real_idx].values
        trans = pose_row[1:4].astype(np.float32)
        rot = pose_row[4:7].astype(np.float32)
        coords3d_hw3 = self._build_coords3d_from_pose(H, W, trans, rot)
        coords3d_hw3 = self._normalize_coords(coords3d_hw3)
        
        # 扁平化坐标
        coords_flat = torch.from_numpy(coords3d_hw3.reshape(-1, 3)).to(self.float_dtype).unsqueeze(0)
        
        # 获取文本描述
        text_description = self._get_text_description(img_path)
        
        # 如果使用tokenizer，进行tokenization
        if self.use_tokenizer:
            text_input = self._tokenize_text(text_description)
        else:
            text_input = text_description  # 保留原始文本
        
        segments = torch.zeros(H * W, dtype=torch.long)
        dino = torch.zeros(1, dtype=self.float_dtype)
        
        sample = {
            'coords': coords_flat,
            'gt_img': gt_img,
            'segments': segments,
            'dino': dino,
            'text': text_input,  # 添加文本字段
            'img_path': img_path,  # 用于调试
        }
        return sample


def text_guided_collate_fn(batch):
    """
    自定义collate函数，处理文本数据
    """
    coords = torch.stack([item['coords'] for item in batch])
    gt_img = torch.stack([item['gt_img'] for item in batch])
    segments = torch.stack([item['segments'] for item in batch])
    dino = torch.stack([item['dino'] for item in batch])
    
    # 收集文本
    texts = [item['text'] for item in batch]
    
    return {
        'coords': coords.squeeze(1),  # (B, N, 3)
        'gt_img': gt_img,  # (B, C, H, W)
        'segments': segments,  # (B, N)
        'dino': dino,  # (B, 1)
        'text': texts,  # List[str] or tokenized
    }


