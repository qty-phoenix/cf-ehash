#!/usr/bin/env python3
"""
简单的KAN（Kolmogorov-Arnold Network）实现
基于B样条函数的可学习激活函数网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BSplineActivation(nn.Module):
    """
    B样条激活函数
    使用三次B样条作为可学习的激活函数
    """
    def __init__(self, grid_size=8, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 初始化样条系数（可学习参数）
        self.coefficients = nn.Parameter(torch.randn(grid_size + spline_order))
        
        # 创建B样条的节点向量 (knot vector)
        # 在[-1, 1]范围内均匀分布
        knots = torch.linspace(-1, 1, grid_size)
        # 添加边界节点以支持三次样条
        knots = torch.cat([
            torch.ones(spline_order) * (-1),
            knots,
            torch.ones(spline_order) * 1
        ])
        self.register_buffer('knots', knots)
    
    def forward(self, x):
        """
        使用B样条插值计算激活函数值
        x: 输入张量，形状任意
        返回: 相同形状的输出张量
        """
        # 将输入限制在[-1, 1]范围内
        x = torch.tanh(x)
        
        # 计算B样条基函数
        # 简化实现：使用线性插值近似B样条
        x_shape = x.shape
        x_flat = x.reshape(-1)
        
        # 将x映射到网格索引
        grid_idx = (x_flat + 1) * (self.grid_size - 1) / 2.0
        grid_idx = torch.clamp(grid_idx, 0, self.grid_size - 1)
        
        # 获取相邻的网格点
        idx_low = grid_idx.long()
        idx_high = torch.clamp(idx_low + 1, 0, self.grid_size - 1)
        
        # 线性插值权重
        weight_high = grid_idx - idx_low.float()
        weight_low = 1.0 - weight_high
        
        # 使用样条系数进行插值
        output = (weight_low * self.coefficients[idx_low] + 
                 weight_high * self.coefficients[idx_high])
        
        return output.reshape(x_shape)


class SimpleKANLayer(nn.Module):
    """
    简单的KAN层
    每个连接都有自己的可学习激活函数（B样条）
    """
    def __init__(self, in_features, out_features, grid_size=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 基础线性变换
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # 每个输出节点都有一个可学习的激活函数
        self.activations = nn.ModuleList([
            BSplineActivation(grid_size=grid_size) 
            for _ in range(out_features)
        ])
        
        # 初始化
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """
        前向传播
        x: (batch, ..., in_features)
        返回: (batch, ..., out_features)
        """
        # 线性变换
        h = self.linear(x)
        
        # 对每个输出特征应用独立的可学习激活
        outputs = []
        for i in range(self.out_features):
            activated = self.activations[i](h[..., i])
            outputs.append(activated)
        
        return torch.stack(outputs, dim=-1)


class SimpleKAN(nn.Module):
    """
    简单的KAN网络
    使用可学习的激活函数替代传统MLP
    """
    def __init__(self, in_features, out_features, hidden_features=128, 
                 num_hidden_layers=2, grid_size=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers
        
        # 构建网络层
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(SimpleKANLayer(in_features, hidden_features, grid_size))
        
        # 隐藏层
        for _ in range(num_hidden_layers):
            self.layers.append(SimpleKANLayer(hidden_features, hidden_features, grid_size))
        
        # 输出层（使用标准线性层，不使用激活函数）
        self.output_layer = nn.Linear(hidden_features, out_features)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        前向传播
        x: (batch, ..., in_features)
        返回: (batch, ..., out_features)
        """
        # 通过KAN层
        for layer in self.layers:
            x = layer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x


class ParallelSimpleKAN(nn.Module):
    """
    并行的简单KAN网络（用于MoE架构）
    k个独立的KAN网络并行执行
    """
    def __init__(self, k, in_features, out_features, hidden_features=128,
                 num_hidden_layers=2, grid_size=8):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_hidden_layers = num_hidden_layers
        
        # 创建k个独立的KAN网络
        self.experts = nn.ModuleList([
            SimpleKAN(in_features, out_features, hidden_features, 
                     num_hidden_layers, grid_size)
            for _ in range(k)
        ])
    
    def forward(self, x):
        """
        并行前向传播
        x: (..., n, d) - 输入坐标
        返回: (..., k, n, out_features) - k个专家的输出
        """
        outputs = []
        for expert in self.experts:
            out = expert(x)  # (..., n, out_features)
            outputs.append(out)
        
        # 堆叠所有专家的输出
        # 从 list of (..., n, out_features) 到 (..., k, n, out_features)
        stacked = torch.stack(outputs, dim=-3)
        
        return stacked


class SimpleKANActivation(nn.Module):
    """
    将SimpleKAN包装成激活函数接口
    用于替代传统的Sine、ReLU等激活函数
    """
    def __init__(self, features, grid_size=8):
        super().__init__()
        self.features = features
        # 为每个特征维度创建一个B样条激活
        self.activations = nn.ModuleList([
            BSplineActivation(grid_size=grid_size) 
            for _ in range(features)
        ])
    
    def forward(self, x):
        """
        x: (..., features)
        返回: (..., features)
        """
        outputs = []
        for i in range(self.features):
            activated = self.activations[i](x[..., i])
            outputs.append(activated)
        return torch.stack(outputs, dim=-1)


