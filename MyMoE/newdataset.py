import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from PIL import Image

class CustomDatasetCompatible(data.Dataset):
    def __init__(self, image_folder, pose_file, mode='train', crop_size=480, split_ratio=0.8):
        self.image_folder = image_folder
        self.pose_file = pose_file
        self.crop_size = crop_size
        self.mode = mode.lower()
        assert self.mode in ['train', 'test'], "mode should be 'train' or 'test'"

        # Load pose info
        self.pose_data = pd.read_excel(pose_file)
        self.image_files = sorted(os.listdir(image_folder))
        assert len(self.image_files) == len(self.pose_data), "图像数量和姿态信息不一致"

        # Split into train/test 每五个的第五张作为测试集
        total = len(self.image_files)
        self.indices = []
        for i in range(len(self.image_files)):
            if self.mode == 'train' and i % 5 != 4:  # 0, 1, 2, 3
                self.indices.append(i)
            elif self.mode == 'test' and i % 5 == 4:  # 4
                self.indices.append(i)
        # Pose & images info storage
        self.overall_store = {}
        self.list_IDs = list(range(len(self.indices)))
        for new_id, idx in enumerate(self.indices):
            pose = self.pose_data.iloc[idx].values
            trans_ground = pose[1:4].astype(np.float32)
            rot_ground = pose[4:7].astype(np.float32)
            store = {
                'img_path': os.path.join(image_folder, self.image_files[idx]),
                'rot_ground': rot_ground,
                'trans_ground': trans_ground,
                # 'rot_pred': rot_ground,
                # 'trans_pred': trans_ground
            }
            self.overall_store[new_id] = store

        # Calculate the global maximum pixel value这个真的有用吗，其实还是/255.0吧
        self.global_max_pixel_value = self._calculate_global_max_pixel_value()

    def __len__(self):
        return len(self.indices)
    # 获取单个索引的方式
    def __getitem__(self, index):
        real_index = self.indices[index]
        image_path = os.path.join(self.image_folder, self.image_files[real_index])

        # Load and crop images
        image = Image.open(image_path).convert('L')
        width, height = image.size
        left = (width - self.crop_size) // 2
        top = (height - self.crop_size) // 2
        right = (width + self.crop_size) // 2
        bottom = (height + self.crop_size) // 2
        image = image.crop((left, top, right, bottom))
        image = np.array(image, dtype=np.float32) / 255.0
        # self.global_max_pixel_value Normalize using global max
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # Pose
        pose = self.pose_data.iloc[real_index].values
        trans_ground = torch.tensor(pose[1:4], dtype=torch.float32)
        rot_ground = torch.tensor(pose[4:7], dtype=torch.float32)
        # rot_pred = rot_ground.clone()
        # trans_pred = trans_ground.clone()

        return index, image, rot_ground, trans_ground#, rot_pred, trans_pred

    def _overall_store(self):
        return self.overall_store

    def _calculate_global_max_pixel_value(self):
        """
        Calculate the global maximum pixel value across all images in the dataset.
        """
        max_pixel_value = 0
        for image_file in self.image_files:
            image_path = os.path.join(self.image_folder, image_file)
            image = Image.open(image_path).convert('L')
            image_array = np.array(image, dtype=np.float32)
            max_pixel_value = max(max_pixel_value, np.max(image_array))
        return max_pixel_value