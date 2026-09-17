#!/usr/bin/env python3
"""
简化版KAN (Kolmogorov-Arnold Networks) 模块实现
更稳定的实现，避免复杂的样条计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class SimpleKANLayer(nn.Module):
    """简化的KAN层实现"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64, 
                 num_basis: int = 8, activation: str = 'silu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_basis = num_basis
        
        # 为每个输入维度创建独立的基函数网络
        self.basis_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU() if activation == 'silu' else nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU() if activation == 'silu' else nn.ReLU(),
                nn.Linear(hidden_dim, num_basis)
            ) for _ in range(in_dim)
        ])
        
        # 输出组合层
        self.output_layer = nn.Linear(in_dim * num_basis, out_dim)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for basis_net in self.basis_networks:
            for layer in basis_net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
        original_shape = x.shape
        batch_shape = original_shape[:-1]
        
        # 重塑输入为 (N, in_dim) 其中 N 是所有batch维度的乘积
        x_flat = x.reshape(-1, self.in_dim)
        batch_size = x_flat.shape[0]
        
        # 为每个输入维度计算基函数
        basis_outputs = []
        for i in range(self.in_dim):
            x_i = x_flat[:, i:i+1]  # (batch_size, 1)
            basis_out = self.basis_networks[i](x_i)  # (batch_size, num_basis)
            basis_outputs.append(basis_out)
        
        # 拼接所有基函数输出
        combined_basis = torch.cat(basis_outputs, dim=-1)  # (batch_size, in_dim * num_basis)
        
        # 计算最终输出
        output = self.output_layer(combined_basis)  # (batch_size, out_dim)
        
        # 重塑回原始batch形状
        output = output.reshape(*batch_shape, self.out_dim)
        
        return output


class SimpleKANNetwork(nn.Module):
    """简化的KAN网络实现"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], 
                 num_basis: int = 8, activation: str = 'silu', dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 构建网络层
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        
        for i in range(len(dims) - 1):
            layers.append(SimpleKANLayer(dims[i], dims[i + 1], 
                                       hidden_dim=hidden_dims[0] if i < len(hidden_dims) else 64,
                                       num_basis=num_basis, activation=activation))
            
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
                    layers.append(nn.SiLU())
                
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


class ParallelSimpleKANNetwork(nn.Module):
    """并行简化KAN网络实现（用于专家网络）"""
    
    def __init__(self, k: int, in_dim: int, out_dim: int, hidden_dims: List[int],
                 num_basis: int = 8, activation: str = 'silu', dropout: float = 0.0,
                 module_name: str = ''):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.module_name = module_name
        
        # 创建k个并行的KAN网络
        self.kan_networks = nn.ModuleList([
            SimpleKANNetwork(in_dim, out_dim, hidden_dims, num_basis, activation, dropout)
            for _ in range(k)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., n, in_dim) 输入坐标
        Returns:
            (1, k, n, out_dim) k个专家的输出
        """
        original_shape = x.shape
        batch_shape = original_shape[:-2]  # (...,)
        n_points = original_shape[-2]      # n
        in_dim_actual = original_shape[-1] # in_dim
        
        # 并行计算所有专家的输出
        outputs = []
        for i in range(self.k):
            # 直接传入x，让SimpleKANNetwork处理形状
            expert_out = self.kan_networks[i](x)  # (..., n, out_dim)
            outputs.append(expert_out)
        
        # 堆叠输出：在倒数第三个维度插入专家维度
        # outputs: list of (..., n, out_dim)
        # stack along dim=1 for batch_shape=(), or appropriate dim
        stacked_output = torch.stack(outputs, dim=-2)  # (..., n, k, out_dim)
        
        # 调整维度顺序为 (..., k, n, out_dim)
        # 如果batch_shape=()，则 stacked_output是 (n, k, out_dim)，需要变成 (k, n, out_dim)
        # 如果batch_shape=(1,)，则 stacked_output是 (1, n, k, out_dim)，需要变成 (1, k, n, out_dim)
        
        ndim = len(original_shape)
        # stacked_output: (*batch_shape, n, k, out_dim)
        # target: (*batch_shape, k, n, out_dim)
        # 交换倒数第2和第3个维度
        perm = list(range(ndim))  # 生成索引列表
        perm[-2], perm[-3] = perm[-3], perm[-2]  # 交换 k 和 n 的位置
        stacked_output = stacked_output.permute(*perm)  # (*batch_shape, k, n, out_dim)
        
        # 确保输出形状与原始MLP一致: (..., k, n, out_dim)
        # 对于RGB重建，out_dim应该是1或3
        
        return stacked_output


class SimpleKANInputEncoder(nn.Module):
    """简化的KAN输入编码器"""
    
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
            self.encoding_net = SimpleKANNetwork(
                cfg['in_dim'], hidden_dim, [hidden_dim], 
                num_basis=8, activation='silu'
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
def create_simple_kan_expert(in_dim: int, out_dim: int, hidden_dims: List[int], 
                            num_basis: int = 8, activation: str = 'silu', dropout: float = 0.0) -> SimpleKANNetwork:
    """创建简化KAN专家网络"""
    return SimpleKANNetwork(in_dim, out_dim, hidden_dims, num_basis, activation, dropout)


def create_parallel_simple_kan_experts(k: int, in_dim: int, out_dim: int, hidden_dims: List[int],
                                      num_basis: int = 8, activation: str = 'silu', dropout: float = 0.0,
                                      module_name: str = '') -> ParallelSimpleKANNetwork:
    """创建并行简化KAN专家网络"""
    return ParallelSimpleKANNetwork(k, in_dim, out_dim, hidden_dims, num_basis, activation, dropout, module_name)
