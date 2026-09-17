#!/usr/bin/env python3
"""
KAN (Kolmogorov-Arnold Networks) 模块实现
用于替换专家网络中的MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SplineBasis(nn.Module):
    """样条基函数实现"""
    
    def __init__(self, in_dim: int, out_dim: int, num_grids: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grids = num_grids
        self.spline_order = spline_order
        
        # 样条系数 - 修正维度
        self.coeff = nn.Parameter(torch.randn(in_dim, out_dim, num_grids + 1))
        
        # 网格点
        self.grid = nn.Parameter(torch.linspace(-1, 1, num_grids + 1).unsqueeze(0).repeat(in_dim, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
        batch_shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)
        
        # 计算样条插值
        output = torch.zeros(x.shape[0], self.out_dim, device=x.device)
        
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                # 找到x[i]在网格中的位置
                grid_points = self.grid[i]
                x_i = x[:, i]
                
                # 计算B样条基函数值
                spline_values = self._compute_spline_basis(x_i, grid_points)
                
                # 加权求和
                output[:, j] += torch.sum(spline_values * self.coeff[i, j], dim=-1)
        
        return output.view(*batch_shape, self.out_dim)
    
    def _compute_spline_basis(self, x: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """计算B样条基函数值"""
        # 简化的B样条实现
        num_points = len(grid_points)
        basis = torch.zeros(x.shape[0], num_points, device=x.device)
        
        # 处理边界情况
        x_clamped = torch.clamp(x, grid_points[0], grid_points[-1])
        
        for i in range(num_points - 1):
            mask = (x_clamped >= grid_points[i]) & (x_clamped < grid_points[i + 1])
            if mask.any():
                # 线性插值
                t = (x_clamped[mask] - grid_points[i]) / (grid_points[i + 1] - grid_points[i])
                basis[mask, i] = 1 - t
                basis[mask, i + 1] = t
        
        # 处理边界点
        mask_left = x_clamped <= grid_points[0]
        mask_right = x_clamped >= grid_points[-1]
        if mask_left.any():
            basis[mask_left, 0] = 1.0
        if mask_right.any():
            basis[mask_right, -1] = 1.0
        
        return basis


class KANLayer(nn.Module):
    """KAN层实现"""
    
    def __init__(self, in_dim: int, out_dim: int, num_grids: int = 5, 
                 spline_order: int = 3, scale_noise: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 样条基函数
        self.spline_basis = SplineBasis(in_dim, out_dim, num_grids, spline_order)
        
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(out_dim))
        
        # 噪声参数（用于正则化）
        self.scale_noise = scale_noise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
        # 样条变换
        spline_out = self.spline_basis(x)
        
        # 应用缩放
        output = spline_out * self.scale
        
        # 添加噪声（训练时）
        if self.training and self.scale_noise > 0:
            noise = torch.randn_like(output) * self.scale_noise
            output = output + noise
            
        return output


class KANNetwork(nn.Module):
    """完整的KAN网络实现"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], 
                 num_grids: int = 5, spline_order: int = 3, 
                 activation: str = 'silu', dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 构建网络层
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], num_grids, spline_order))
            
            # 添加激活函数（除了最后一层）
            if i < len(dims) - 2:
                if activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.SiLU())  # 默认使用SiLU
                
                # 添加dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
        return self.network(x)


class ParallelKANNetwork(nn.Module):
    """并行KAN网络实现（用于专家网络）"""
    
    def __init__(self, k: int, in_dim: int, out_dim: int, hidden_dims: List[int],
                 num_grids: int = 5, spline_order: int = 3, 
                 activation: str = 'silu', dropout: float = 0.0,
                 module_name: str = ''):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module_name = module_name
        
        # 创建k个并行的KAN网络
        self.kan_networks = nn.ModuleList([
            KANNetwork(in_dim, out_dim, hidden_dims, num_grids, spline_order, activation, dropout)
            for _ in range(k)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., n, in_dim) 输入坐标
        Returns:
            (..., k, n, out_dim) k个专家的输出
        """
        batch_shape = x.shape[:-2]  # (...,)
        n_points = x.shape[-2]      # n
        in_dim = x.shape[-1]        # in_dim
        
        # 重塑输入
        x_flat = x.view(-1, in_dim)  # (..., n, in_dim) -> (..., n, in_dim)
        
        # 并行计算所有专家的输出
        outputs = []
        for i in range(self.k):
            expert_out = self.kan_networks[i](x_flat)  # (..., n, out_dim)
            outputs.append(expert_out)
        
        # 堆叠输出
        stacked_output = torch.stack(outputs, dim=-3)  # (..., k, n, out_dim)
        
        return stacked_output


class KANInputEncoder(nn.Module):
    """KAN输入编码器"""
    
    def __init__(self, cfg: dict, input_encoding: str, hidden_dim: int, module_name: str = ''):
        super().__init__()
        self.cfg = cfg
        self.input_encoding = input_encoding
        self.hidden_dim = hidden_dim
        self.module_name = module_name
        
        # 解析输入编码类型
        if 'PE' in input_encoding:
            # 位置编码
            self.freq = cfg.get('decoder_freqs', 30.0)
            self.trainable_freqs = cfg.get('decoder_trainable_freqs', False)
            self.first_layer_dim = cfg['in_dim'] * (2 * int(self.freq) + 1)
            
            if self.trainable_freqs:
                self.freq_params = nn.Parameter(torch.randn(int(self.freq)))
            else:
                self.freq_params = None
                
        elif 'learned' in input_encoding:
            # 学习编码
            self.first_layer_dim = hidden_dim
            self.encoding_net = KANNetwork(
                cfg['in_dim'], hidden_dim, [hidden_dim], 
                num_grids=5, spline_order=3
            )
        else:
            # 无编码
            self.first_layer_dim = cfg['in_dim']
            
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., first_layer_dim)
        """
        if 'PE' in self.input_encoding:
            # 位置编码
            if self.trainable_freqs:
                freqs = self.freq_params
            else:
                freqs = torch.arange(1, int(self.freq) + 1, device=x.device)
            
            # 计算位置编码
            encoded = [x]
            for freq in freqs:
                encoded.append(torch.sin(2 * np.pi * freq * x))
                encoded.append(torch.cos(2 * np.pi * freq * x))
            
            return torch.cat(encoded, dim=-1)
            
        elif 'learned' in self.input_encoding:
            # 学习编码
            return self.encoding_net(x)
        else:
            # 无编码
            return x


# 兼容性函数，用于替换原有的FullyConnectedNN
def create_kan_expert(in_dim: int, out_dim: int, hidden_dims: List[int], 
                     num_grids: int = 5, spline_order: int = 3,
                     activation: str = 'silu', dropout: float = 0.0) -> KANNetwork:
    """创建KAN专家网络"""
    return KANNetwork(in_dim, out_dim, hidden_dims, num_grids, spline_order, activation, dropout)


def create_parallel_kan_experts(k: int, in_dim: int, out_dim: int, hidden_dims: List[int],
                               num_grids: int = 5, spline_order: int = 3,
                               activation: str = 'silu', dropout: float = 0.0,
                               module_name: str = '') -> ParallelKANNetwork:
    """创建并行KAN专家网络"""
    return ParallelKANNetwork(k, in_dim, out_dim, hidden_dims, num_grids, spline_order, activation, dropout, module_name)
