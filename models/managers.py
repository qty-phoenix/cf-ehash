# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This file contains the code for different manager architecture implementations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from models.pointnet import PointNetfeat
from models.modules import ParallelFullyConnectedNN
from models.modules import InputEncoder, FullyConnectedNN


class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return args[0]

class DummyManager(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.n_experts = n_experts
    def forward(self, points):
        return torch.zeros(points.shape[0], self.n_experts, device=points.device), (None, None)

class DummyBias(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.n_experts = n_experts
    def forward(self, points, q):
        return q

class ManagerConditioner(nn.Module):
    def __init__(self, manager_conditioning, laster_layer_dim=128, expert_decoder=None):
        super().__init__()
        self.manager_conditioning = manager_conditioning

    def forward(self, x, manager_input, **kwargs):
        if self.manager_conditioning == 'max':
            point_rep = torch.max(x, dim=1)[0].unsqueeze(1).expand(-1, manager_input.shape[1], -1)  # global information for the manager
        elif self.manager_conditioning == 'mean':
            point_rep = torch.mean(x, dim=1).unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == 'cat':
            point_rep = x
        else:
            point_rep = None

        if point_rep is not None:
            manager_input = torch.cat([manager_input, point_rep], dim=-1)

        return manager_input


class TextGuidedManagerConditioner(nn.Module):
    """
    文本引导的Manager条件调节器
    将文本特征融入到Manager的决策中
    """
    def __init__(self, manager_conditioning, layer_dim=128, text_dim=256, 
                 fusion_type='add', expert_decoder=None):
        super().__init__()
        self.manager_conditioning = manager_conditioning
        self.fusion_type = fusion_type  # 'add', 'concat', 'cross_attention'
        self.text_dim = text_dim
        self.layer_dim = layer_dim
        
        # 文本特征投影层
        if fusion_type == 'add':
            # 加法融合：将文本特征投影到与manager_input相同的维度
            self.text_projection = nn.Linear(text_dim, layer_dim)
        elif fusion_type == 'concat':
            # 拼接融合：不需要额外投影
            self.text_projection = nn.Identity()
        elif fusion_type == 'cross_attention':
            # 交叉注意力融合
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=layer_dim,
                num_heads=8,
                batch_first=True
            )
            self.text_projection = nn.Linear(text_dim, layer_dim)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
    
    def forward(self, x, manager_input, text_features=None, **kwargs):
        """
        Args:
            x: expert编码特征 (B, n_points, layer_dim)
            manager_input: manager输入特征 (B, n_points, input_dim)
            text_features: 文本特征 (B, text_dim)
        Returns:
            conditioned_input: 条件调节后的输入
        """
        # 首先应用原始的条件调节逻辑
        if self.manager_conditioning == 'max':
            point_rep = torch.max(x, dim=1)[0].unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == 'mean':
            point_rep = torch.mean(x, dim=1).unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == 'cat':
            point_rep = x
        else:
            point_rep = None

        if point_rep is not None:
            manager_input = torch.cat([manager_input, point_rep], dim=-1)
        
        # 如果提供了文本特征，进行文本引导的调节
        if text_features is not None:
            B, n_points, input_dim = manager_input.shape
            
            if self.fusion_type == 'add':
                # 加法融合
                text_proj = self.text_projection(text_features)  # (B, layer_dim)
                text_proj = text_proj.unsqueeze(1).expand(-1, n_points, -1)  # (B, n_points, layer_dim)
                # 这里假设manager_input的最后layer_dim维度用于融合
                if input_dim >= self.layer_dim:
                    manager_input[..., :self.layer_dim] = manager_input[..., :self.layer_dim] + text_proj
                else:
                    # 如果维度不够，进行拼接
                    manager_input = torch.cat([manager_input, text_proj], dim=-1)
            
            elif self.fusion_type == 'concat':
                # 拼接融合
                text_expanded = text_features.unsqueeze(1).expand(-1, n_points, -1)  # (B, n_points, text_dim)
                manager_input = torch.cat([manager_input, text_expanded], dim=-1)
            
            elif self.fusion_type == 'cross_attention':
                # 交叉注意力融合
                text_proj = self.text_projection(text_features).unsqueeze(1)  # (B, 1, layer_dim)
                # manager_input作为query, text作为key和value
                if input_dim == self.layer_dim:
                    attn_out, _ = self.cross_attn(manager_input, text_proj, text_proj)
                    manager_input = manager_input + attn_out  # 残差连接
                else:
                    # 如果维度不匹配，只对前layer_dim维度做attention
                    query = manager_input[..., :self.layer_dim]
                    attn_out, _ = self.cross_attn(query, text_proj, text_proj)
                    manager_input = torch.cat([query + attn_out, manager_input[..., self.layer_dim:]], dim=-1)

        return manager_input
