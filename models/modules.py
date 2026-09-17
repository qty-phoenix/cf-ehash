import torch.nn as nn
import torch.nn.functional as F
import models.initializations as inits
import torch
from collections import OrderedDict
import numpy as np
import math

class HashEncoding(nn.Module):
    """
    多分辨率哈希编码 (Multi-Resolution Hash Encoding)
    基于Instant-NGP论文实现的纯PyTorch版本
    
    参数:
        n_levels: 哈希层级数量
        n_features_per_level: 每层的特征维度
        log2_hashmap_size: 哈希表大小的log2值 (实际大小为2^log2_hashmap_size)
        base_resolution: 最粗糙层级的分辨率
        finest_resolution: 最精细层级的分辨率
        input_dim: 输入坐标维度 (2D或3D)
    """
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19,
                 base_resolution=16, finest_resolution=512, input_dim=3):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.input_dim = input_dim
        self.output_dim = n_levels * n_features_per_level
        
        # 计算每层的分辨率增长因子
        if n_levels > 1:
            self.per_level_scale = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1))
        else:
            self.per_level_scale = 1.0
        
        # 哈希表大小
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # 为每一层创建哈希表 (可学习的embedding)
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level) 
            for _ in range(n_levels)
        ])
        
        # 初始化哈希表 - 使用接近SIREN标准的初始化范围
        # SIREN的第一层期望输入在[-1, 1]附近，所以哈希表需要较大的初始值
        with torch.no_grad():
            for hash_table in self.hash_tables:
                # 使用更大的范围：[-0.1, 0.1]
                hash_table.weight.uniform_(-0.1, 0.1)
        
        # 预计算每层的分辨率
        self.resolutions = [int(base_resolution * (self.per_level_scale ** i)) for i in range(n_levels)]
        
        # 【修复】哈希函数的大质数（用于3D坐标的哈希）
        # 使用 Instant-NGP 标准质数：所有维度都用大质数，避免第一维未混洗导致碰撞
        # 参考: https://github.com/NVlabs/instant-ngp
        if input_dim == 3:
            self.primes = [2654435761, 805459861, 1958374283]
        elif input_dim == 2:
            self.primes = [2654435761, 805459861]
        else:
            # 对于其他维度，使用通用大质数序列
            self.primes = [2654435761, 805459861, 1958374283, 3267000013][:input_dim]
        
        # 【优化】预计算顶点偏移模板，避免每次 forward 重复计算
        # 2D: 4个顶点 (00, 01, 10, 11)，3D: 8个顶点 (000, 001, 010, ..., 111)
        n_vertices = 2 ** input_dim
        vertex_offsets = torch.zeros(n_vertices, input_dim, dtype=torch.long)
        for i in range(n_vertices):
            for d in range(input_dim):
                if (i >> d) & 1:
                    vertex_offsets[i, d] = 1
        # 注册为 buffer（自动跟随模型移动到 GPU/CPU，但不参与训练）
        self.register_buffer('vertex_offsets', vertex_offsets)
        
        # 【修复】可学习的输出缩放因子，初始化为10.0以匹配SIREN期望的输入范围
        # 训练过程中会自动调整以适应实际的哈希表权重分布
        self.output_scale = nn.Parameter(torch.tensor(10.0))
        
    def hash_function(self, coords, resolution):
        """
        将坐标映射到哈希表索引
        coords: (B, N, input_dim) 归一化坐标 [0, 1]
        resolution: 当前层级的分辨率
        返回: (B, N, 2^input_dim) 顶点索引和插值权重
        """
        # 将坐标缩放到网格空间 [0, resolution-1]
        coords_scaled = coords * (resolution - 1)
        
        # 计算网格坐标（下界）- 先进行clamp确保在有效范围内
        coords_grid_float = torch.floor(coords_scaled)
        coords_grid = torch.clamp(coords_grid_float, 0, resolution - 2).long()
        
        # 【修复】基于clamp后的coords_grid计算插值权重，确保权重与实际使用的网格一致
        interp_weights = coords_scaled - coords_grid.float()
        # 将插值权重限制在[0, 1]范围内（处理边界情况）
        interp_weights = torch.clamp(interp_weights, 0.0, 1.0)
        
        # 生成网格顶点（2D有4个顶点，3D有8个顶点）
        B, N = coords_grid.shape[0], coords_grid.shape[1]
        
        # 【优化】直接使用预计算的顶点偏移模板（已在 __init__ 中计算并注册为 buffer）
        # self.vertex_offsets: (n_vertices, input_dim)，已自动移动到正确的设备
        # 广播计算所有顶点坐标 (B, N, n_vertices, input_dim)
        vertex_coords = coords_grid.unsqueeze(2) + self.vertex_offsets.unsqueeze(0).unsqueeze(0)
        
        # 【优化】向量化计算哈希值
        # 计算哈希: hash = (x*prime[0] XOR y*prime[1] XOR z*prime[2]) % hashmap_size
        n_vertices = self.vertex_offsets.shape[0]  # 使用预计算的顶点数
        hash_val = torch.zeros(B, N, n_vertices, dtype=torch.long, device=coords.device)
        for d in range(self.input_dim):
            hash_val = hash_val ^ (vertex_coords[..., d] * self.primes[d])
        hash_val = hash_val % self.hashmap_size
        
        return hash_val, interp_weights
    
    def interpolate_features(self, hash_indices, interpolation_weights, hash_table):
        """
        使用多线性插值从哈希表中获取特征
        hash_indices: (B, N, n_vertices) 顶点哈希索引
        interpolation_weights: (B, N, input_dim) 插值权重
        hash_table: 当前层的哈希表
        """
        B, N, n_vertices = hash_indices.shape
        
        # 查找所有顶点的特征
        vertex_features = hash_table(hash_indices.reshape(-1))  # (B*N*n_vertices, F)
        vertex_features = vertex_features.reshape(B, N, n_vertices, self.n_features_per_level)
        
        # 计算插值权重（多线性插值）
        weights = []
        for i in range(n_vertices):
            weight = torch.ones(B, N, 1, device=interpolation_weights.device)
            for d in range(self.input_dim):
                if (i >> d) & 1:
                    weight = weight * interpolation_weights[..., d:d+1]
                else:
                    weight = weight * (1 - interpolation_weights[..., d:d+1])
            weights.append(weight)
        
        weights = torch.cat(weights, dim=-1)  # (B, N, n_vertices)
        
        # 加权求和
        features = (vertex_features * weights.unsqueeze(-1)).sum(dim=2)  # (B, N, F)
        
        return features
    
    def forward(self, coords):
        """
        coords: (B, N, input_dim) 输入坐标，假设已归一化到 [-1, 1]
        返回: (B, N, n_levels * n_features_per_level) 编码后的特征
        """
        # 将坐标从 [-1, 1] 归一化到 [0, 1]
        coords_normalized = (coords + 1.0) / 2.0
        coords_normalized = torch.clamp(coords_normalized, 0.0, 1.0)
        
        # 对每一层进行编码
        encoded_features = []
        for level in range(self.n_levels):
            resolution = self.resolutions[level]
            hash_table = self.hash_tables[level]
            
            # 计算哈希索引和插值权重
            hash_indices, interp_weights = self.hash_function(coords_normalized, resolution)
            
            # 插值获取特征
            features = self.interpolate_features(hash_indices, interp_weights, hash_table)
            encoded_features.append(features)
        
        # 拼接所有层的特征
        encoded = torch.cat(encoded_features, dim=-1)  # (B, N, n_levels * n_features_per_level)
        
        # 【修复】使用可学习的缩放因子，自动适应训练过程中的权重变化
        # 初始值10.0配合0.1的初始化，输出范围约为[-1, 1]，适配SIREN
        encoded = encoded * self.output_scale
        
        return encoded
    
    def extra_repr(self):
        return (f'n_levels={self.n_levels}, n_features_per_level={self.n_features_per_level}, '
                f'hashmap_size=2^{self.log2_hashmap_size}={self.hashmap_size}, '
                f'resolution=[{self.base_resolution}, {self.finest_resolution}], '
                f'output_dim={self.output_dim}, output_scale={self.output_scale.item():.4f}')


class MultiExpertHashEncoding(nn.Module):
    """
    多专家版本哈希编码
    前shared_levels层共用（单专家），后n_levels-shared_levels层使用多专家路由
    
    参数:
        n_levels: 哈希层级数量（例如12层）
        n_experts: 专家数量（例如4个）
        n_features_per_level: 每层的特征维度
        log2_hashmap_size: 哈希表大小的log2值
        base_resolution: 最粗糙层级的分辨率
        finest_resolution: 最精细层级的分辨率
        input_dim: 输入坐标维度 (2D或3D)
        shared_levels: 前几层共享（例如6层），剩余层为多专家层
    """
    def __init__(self, n_levels=12, n_experts=4, n_features_per_level=2, log2_hashmap_size=14,
                 base_resolution=16, finest_resolution=2048, input_dim=3, shared_levels=6):
        super().__init__()
        self.n_levels = n_levels
        self.n_experts = n_experts
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.input_dim = input_dim
        self.shared_levels = shared_levels
        self.expert_levels = n_levels - shared_levels
        
        # 验证参数
        assert shared_levels > 0, "shared_levels必须大于0"
        assert shared_levels < n_levels, "shared_levels必须小于n_levels"
        assert self.expert_levels > 0, "expert_levels必须大于0"
        
        # 前shared_levels层的输出维度（共用层）
        self.shared_output_dim = shared_levels * n_features_per_level
        # 后expert_levels层每个专家的输出维度
        self.expert_output_dim = self.expert_levels * n_features_per_level
        # 总输出维度（对于单个专家）
        self.output_dim = self.shared_output_dim + self.expert_output_dim
        
        # 计算每层的分辨率增长因子
        if n_levels > 1:
            self.per_level_scale = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1))
        else:
            self.per_level_scale = 1.0
        
        # 哈希表大小
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # 前shared_levels层共用的哈希表
        self.shared_hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level) 
            for _ in range(shared_levels)
        ])
        
        # 后expert_levels层：每个专家独立的哈希表组
        # 每个专家有expert_levels个哈希表
        self.expert_hash_tables = nn.ModuleList([
            nn.ModuleList([
                nn.Embedding(self.hashmap_size, n_features_per_level)
                for _ in range(self.expert_levels)
            ])
            for _ in range(n_experts)
        ])
        
        # 初始化哈希表
        with torch.no_grad():
            for hash_table in self.shared_hash_tables:
                hash_table.weight.uniform_(-0.1, 0.1)
            for expert_table_group in self.expert_hash_tables:
                for hash_table in expert_table_group:
                    hash_table.weight.uniform_(-0.1, 0.1)
        
        # 预计算每层的分辨率
        self.resolutions = [int(base_resolution * (self.per_level_scale ** i)) for i in range(n_levels)]
        
        # 哈希函数的大质数
        if input_dim == 3:
            self.primes = [2654435761, 805459861, 1958374283]
        elif input_dim == 2:
            self.primes = [2654435761, 805459861]
        else:
            self.primes = [2654435761, 805459861, 1958374283, 3267000013][:input_dim]
        
        # 预计算顶点偏移模板
        n_vertices = 2 ** input_dim
        vertex_offsets = torch.zeros(n_vertices, input_dim, dtype=torch.long)
        for i in range(n_vertices):
            for d in range(input_dim):
                if (i >> d) & 1:
                    vertex_offsets[i, d] = 1
        self.register_buffer('vertex_offsets', vertex_offsets)
        
        # 可学习的输出缩放因子
        self.output_scale = nn.Parameter(torch.tensor(10.0))
        
    def hash_function(self, coords, resolution):
        """哈希函数：将坐标映射到哈希表索引"""
        coords_scaled = coords * (resolution - 1)
        coords_grid_float = torch.floor(coords_scaled)
        coords_grid = torch.clamp(coords_grid_float, 0, resolution - 2).long()
        
        interp_weights = coords_scaled - coords_grid.float()
        interp_weights = torch.clamp(interp_weights, 0.0, 1.0)
        
        B, N = coords_grid.shape[0], coords_grid.shape[1]
        vertex_coords = coords_grid.unsqueeze(2) + self.vertex_offsets.unsqueeze(0).unsqueeze(0)
        
        n_vertices = self.vertex_offsets.shape[0]
        hash_val = torch.zeros(B, N, n_vertices, dtype=torch.long, device=coords.device)
        for d in range(self.input_dim):
            hash_val = hash_val ^ (vertex_coords[..., d] * self.primes[d])
        hash_val = hash_val % self.hashmap_size
        
        return hash_val, interp_weights
    
    def interpolate_features(self, hash_indices, interpolation_weights, hash_table):
        """多线性插值获取特征"""
        B, N, n_vertices = hash_indices.shape
        
        vertex_features = hash_table(hash_indices.reshape(-1))
        vertex_features = vertex_features.reshape(B, N, n_vertices, self.n_features_per_level)
        
        weights = []
        for i in range(n_vertices):
            weight = torch.ones(B, N, 1, device=interpolation_weights.device)
            for d in range(self.input_dim):
                if (i >> d) & 1:
                    weight = weight * interpolation_weights[..., d:d+1]
                else:
                    weight = weight * (1 - interpolation_weights[..., d:d+1])
            weights.append(weight)
        
        weights = torch.cat(weights, dim=-1)
        features = (vertex_features * weights.unsqueeze(-1)).sum(dim=2)
        
        return features
    
    def forward(self, coords):
        """
        coords: (B, N, input_dim) 输入坐标，假设已归一化到 [-1, 1]
        返回: 
            shared_features: (B, N, shared_output_dim) 前shared_levels层的共用特征
            expert_features: (B, n_experts, N, expert_output_dim) 后expert_levels层每个专家的特征
        """
        # 将坐标从 [-1, 1] 归一化到 [0, 1]
        coords_normalized = (coords + 1.0) / 2.0
        coords_normalized = torch.clamp(coords_normalized, 0.0, 1.0)
        
        # 编码前shared_levels层（共用）
        shared_features = []
        for level in range(self.shared_levels):
            resolution = self.resolutions[level]
            hash_table = self.shared_hash_tables[level]
            
            hash_indices, interp_weights = self.hash_function(coords_normalized, resolution)
            features = self.interpolate_features(hash_indices, interp_weights, hash_table)
            shared_features.append(features)
        
        shared_encoded = torch.cat(shared_features, dim=-1)  # (B, N, shared_output_dim)
        shared_encoded = shared_encoded * self.output_scale
        
        # 编码后expert_levels层（多专家）
        expert_features = []
        
        for expert_idx in range(self.n_experts):
            expert_level_features = []
            for level_offset in range(self.expert_levels):
                level = self.shared_levels + level_offset
                resolution = self.resolutions[level]
                hash_table = self.expert_hash_tables[expert_idx][level_offset]
                
                hash_indices, interp_weights = self.hash_function(coords_normalized, resolution)
                features = self.interpolate_features(hash_indices, interp_weights, hash_table)
                expert_level_features.append(features)
            
            # 拼接该专家所有层的特征
            expert_encoded = torch.cat(expert_level_features, dim=-1)  # (B, N, expert_output_dim)
            expert_features.append(expert_encoded)
        
        # expert_features: list of (B, N, expert_output_dim)
        expert_encoded = torch.stack(expert_features, dim=1)  # (B, n_experts, N, expert_output_dim)
        expert_encoded = expert_encoded * self.output_scale
        
        return shared_encoded, expert_encoded
    
    def extra_repr(self):
        return (f'n_levels={self.n_levels}, shared_levels={self.shared_levels}, expert_levels={self.expert_levels}, '
                f'n_experts={self.n_experts}, n_features_per_level={self.n_features_per_level}, '
                f'hashmap_size=2^{self.log2_hashmap_size}={self.hashmap_size}, '
                f'resolution=[{self.base_resolution}, {self.finest_resolution}], '
                f'shared_output_dim={self.shared_output_dim}, expert_output_dim={self.expert_output_dim}')


class HeterogeneousMultiExpertHashEncoding(nn.Module):
    """
    Switch NeRF++的异构哈希专家混合编码 (Heterogeneous Mixture of Hash Experts, HMoHE)
    每个专家使用不同分辨率范围的哈希网格，实现场景的异构表示
    
    参数:
        n_experts: 专家数量（例如4个）
        n_levels_per_expert: 每个专家的哈希层级数量（例如6层）
        n_features_per_level: 每层的特征维度
        log2_hashmap_size: 哈希表大小的log2值
        base_resolutions: 每个专家的最粗糙分辨率列表 (list，长度=n_experts)
        finest_resolutions: 每个专家的最精细分辨率列表 (list，长度=n_experts)
        input_dim: 输入坐标维度 (2D或3D)
    """
    def __init__(self, n_experts=4, n_levels_per_expert=6, n_features_per_level=2, 
                 log2_hashmap_size=14, base_resolutions=None, finest_resolutions=None, input_dim=3):
        super().__init__()
        self.n_experts = n_experts
        self.n_levels_per_expert = n_levels_per_expert
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.input_dim = input_dim
        
        # 如果未提供分辨率范围，自动生成（不同专家使用不同范围）
        if base_resolutions is None:
            # 默认: 专家0使用较低分辨率，专家3使用较高分辨率
            base_resolutions = [16, 32, 64, 128][:n_experts]
        if finest_resolutions is None:
            finest_resolutions = [512, 1024, 2048, 4096][:n_experts]
        
        assert len(base_resolutions) == n_experts, f"base_resolutions长度必须等于n_experts ({n_experts})"
        assert len(finest_resolutions) == n_experts, f"finest_resolutions长度必须等于n_experts ({n_experts})"
        
        self.base_resolutions = base_resolutions
        self.finest_resolutions = finest_resolutions
        
        # 每个专家的输出维度
        self.expert_output_dim = n_levels_per_expert * n_features_per_level
        
        # 哈希表大小
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # 为每个专家创建独立的哈希编码器
        self.expert_encoders = nn.ModuleList([
            HashEncoding(
                n_levels=n_levels_per_expert,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolutions[e],
                finest_resolution=finest_resolutions[e],
                input_dim=input_dim
            )
            for e in range(n_experts)
        ])
        
        print(f"[HeterogeneousMultiExpertHashEncoding] 异构哈希专家混合初始化:")
        for e in range(n_experts):
            print(f"  - Expert {e}: {n_levels_per_expert} levels, resolution range [{base_resolutions[e]}, {finest_resolutions[e]}]")
        print(f"  - Features per level: {n_features_per_level}")
        print(f"  - Hashmap size: 2^{log2_hashmap_size} = {self.hashmap_size}")
        print(f"  - Expert output dim: {self.expert_output_dim}")
    
    def forward(self, coords):
        """
        coords: (B, N, input_dim) 输入坐标，假设已归一化到 [-1, 1]
        返回: 
            expert_features: (B, n_experts, N, expert_output_dim) 每个专家的编码特征
        """
        # HashEncoding的forward方法期望输入在[-1, 1]范围，内部会处理归一化
        # 所以这里直接传入coords即可
        
        # 为每个专家编码
        expert_features = []
        for expert_idx in range(self.n_experts):
            # 每个专家的哈希编码器独立编码
            expert_encoded = self.expert_encoders[expert_idx](coords)
            # expert_encoded: (B, N, expert_output_dim)
            expert_features.append(expert_encoded)
        
        # 堆叠: (B, n_experts, N, expert_output_dim)
        expert_encoded = torch.stack(expert_features, dim=1)
        
        return expert_encoded
    
    def extra_repr(self):
        return (f'n_experts={self.n_experts}, n_levels_per_expert={self.n_levels_per_expert}, '
                f'n_features_per_level={self.n_features_per_level}, '
                f'hashmap_size=2^{self.log2_hashmap_size}={self.hashmap_size}, '
                f'resolution_ranges={list(zip(self.base_resolutions, self.finest_resolutions))}, '
                f'expert_output_dim={self.expert_output_dim}')


class Sine(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = freq
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.freq * input)

class FINER(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = freq
    def forward(self, input):
        with torch.no_grad():
            scale = torch.abs(input) + 1
        return torch.sin(self.freq * scale * input)


class FullyConnectedNN(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 input_encoding=None,
                 sphere_init_params=[1.6,1.0], verbose=True, init_r=0.5, freq=30, trainable_freqs=False, module_name=''):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.init_type = init_type
        self.input_encoding = input_encoding
        self.sphere_init_params = sphere_init_params
        first_layer_dim = in_features

        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.c_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(first_layer_dim, hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(hidden_features)))
        for i in range(num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(hidden_features, hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(hidden_features, out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(out_features)))


        self.module_name = module_name

        nl_dict = {'sine': Sine(freq, trainable_freqs), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'finer': FINER()}
        assert nonlinearity in nl_dict.keys()
        self.nl = nl_dict[nonlinearity]

        init_dict = {'siren': lambda: inits.sirenWeightInit(self), 
                     'finer': lambda: inits.finerWeightInit(self),
                     'geometric_sine': lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
                     'geometric_relu': lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
                     'normal': lambda: inits.kaimingNormalWeightInit(self),
                     'kaiminguniform': lambda: inits.kaimingUniformWeightInit(self),
                        }
        init_dict[init_type]()
    
    def forward(self, input):
        # coords: (1,n,d)

        # Run through network
        x = torch.einsum('...nd,dh->...nh', input, self.weights_list[0]) + self.biases_list[0] # (1,n,h)
        x = self.nl(x)
        for i in range(self.num_hidden_layers):
            x = torch.einsum('...nd,dh->...nh', x, self.weights_list[i+1]) + self.biases_list[i+1] # (1,n,h)
            x = self.nl(x)

        x = torch.einsum('...nh,ho->...no', x, self.weights_list[-1]) + self.biases_list[-1] # (1,n,o)

        if not self.outermost_linear:
            x = self.nl(x)

        # Apply output scaling if any
        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            x = torch.sign(x)*torch.sqrt(x.abs()+1e-8)
            x -= radius # 1.6
            x *= scaling # 1.0

        return x


class ParallelFullyConnectedNN(nn.Module):
    '''K parallel connected neural networks.
    '''

    def __init__(self, k, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 input_encoding=None,
                 sphere_init_params=[1.6,1.0], init_r=0.5, freq=30,
                 module_name='', cfg=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.init_type = init_type
        self.input_encoding = input_encoding
        self.sphere_init_params = sphere_init_params

        first_layer_dim = in_features


        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(k, first_layer_dim, hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(k, hidden_features)))
        for _ in range(num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(k, hidden_features, hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(k, hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(k, hidden_features, out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(k, out_features)))

        self.module_name = module_name

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(),'finer': FINER()}
        assert nonlinearity in nl_dict.keys(), nonlinearity
        self.nl = nl_dict[nonlinearity]

        init_dict = {'siren': lambda: inits.sirenWeightInit(self),
                     'finer': lambda: inits.finerWeightInit(self),
                     'sirensame': lambda: inits.sirenSameWeightInit(self),
                     'geometric_sine': lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
                     'geometric_relu': lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
                     'normal': lambda: inits.kaimingNormalWeightInit(self),
                     'kaiminguniform': lambda: inits.kaimingUniformWeightInit(self),
                    }
        with torch.no_grad():
            init_dict[init_type]()

    def forward(self, x):
        # coords: (...,n,d)

        # Run through network
        x = torch.einsum('...nd,kdh->...knh', x, self.weights_list[0]) + self.biases_list[0].unsqueeze(-2) # (1,k,n,h)
        x = self.nl(x)
        for i in range(self.num_hidden_layers):
            x = torch.einsum('...knd,kdh->...knh', x, self.weights_list[i+1]) + self.biases_list[i+1].unsqueeze(-2) # (1,k,n,h)
            x = self.nl(x)
        x = torch.einsum('...knh,kho->...kno', x, self.weights_list[-1]) + self.biases_list[-1].unsqueeze(-2) # (1,k,n,o)
        if not self.outermost_linear:
            x = self.nl(x)

        # Apply output scaling if any
        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            x = torch.sign(x)*torch.sqrt(x.abs()+1e-8)
            x -= radius # 1.6
            x *= scaling # 1.0

        # return x[...,0,:,:] # (...,n,1), taking the 1st expert's output
        return x # (...,k,n,1)


class InputEncoder(nn.Module):
    def __init__(self, cfg, input_encoding, hidden_dim, module_name=''):
        super().__init__()
        self.input_encoding = input_encoding
        hidden_features = hidden_dim
        in_features = cfg['in_dim']
        self.first_layer_dim = in_features

        # concatenated input features
        if 'FF' in self.input_encoding:
            # Fourier Features
            assert hidden_features % 2 == 0
            self.bvals_size = hidden_features // 2
            bvals = torch.randn(size=[self.bvals_size, cfg['in_dim']], dtype=torch.float32) * 1
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = hidden_features + cfg['in_dim']
        elif 'PE' in self.input_encoding:
            # Positional Encoding - 支持指数频率（NeRF风格）和线性频率
            # 优先使用pe_num_freqs，如果没有则使用decoder_freqs或manager_freqs的整数值，最后默认6
            if 'pe_num_freqs' in cfg:
                num_freqs = int(cfg['pe_num_freqs'])
            elif 'decoder_freqs' in cfg:
                num_freqs = int(cfg['decoder_freqs'])
            elif 'manager_freqs' in cfg:
                num_freqs = int(cfg['manager_freqs'])
            else:
                num_freqs = 6  # 默认6个频率
            
            # 检查是否使用线性频率（通过配置参数 pe_linear_freqs 控制）
            use_linear_freqs = cfg.get('pe_linear_freqs', False)
            if use_linear_freqs:
                # 线性频率：从1到num_freqs的均匀分布
                bvals = torch.linspace(1.0, num_freqs, num_freqs)
            else:
                # 指数频率（NeRF风格）：2的幂次
                bvals = 2 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
            
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = cfg['in_dim'] * num_freqs * 2 + cfg['in_dim']
        elif 'dino' in self.input_encoding:
            self.first_layer_dim = cfg['in_dim'] + cfg['dino_dim']
        elif 'HE' in self.input_encoding or 'learned' in self.input_encoding:
            # Hash Encoding - 多分辨率哈希编码
            if 'HE' in self.input_encoding:
                n_levels = cfg.get('hash_n_levels', 12)
                n_features_per_level = cfg.get('hash_n_features_per_level', 2)
                log2_hashmap_size = cfg.get('hash_log2_hashmap_size', 14)
                base_resolution = cfg.get('hash_base_resolution', 16)
                finest_resolution = cfg.get('hash_finest_resolution', 2048)
                
                self.hash_encoder = HashEncoding(
                    n_levels=n_levels,
                    n_features_per_level=n_features_per_level,
                    log2_hashmap_size=log2_hashmap_size,
                    base_resolution=base_resolution,
                    finest_resolution=finest_resolution,
                    input_dim=cfg['in_dim']
                )
                hash_dim = self.hash_encoder.output_dim
                self.first_layer_dim = hash_dim + cfg['in_dim']  # 拼接原始坐标
            else:
                self.hash_encoder = None
                hash_dim = 0

            # learned input encoding
            if 'learned' in self.input_encoding:
                parsed_str = self.input_encoding.split('_')
                enc_hidden_features = int(parsed_str[1])
                enc_n_layers = int(parsed_str[2])
                nl = parsed_str[3]
                init = parsed_str[4]
                self.encoder = FullyConnectedNN(self.first_layer_dim, enc_hidden_features, num_hidden_layers=enc_n_layers,
                                        hidden_features=enc_hidden_features, outermost_linear=False,
                                        nonlinearity=nl, init_type=init,
                                        module_name=module_name + '.encoder')
                self.first_layer_dim = enc_hidden_features + self.first_layer_dim if 'cat' in self.input_encoding else enc_hidden_features
        
        # 打印输入编码维度信息
        encoding_type = self.input_encoding
        if 'PE' in encoding_type:
            use_linear_freqs = cfg.get('pe_linear_freqs', False)
            if use_linear_freqs:
                encoding_name = "Linear Positional Encoding"
            else:
                encoding_name = "NeRF-style Positional Encoding (Exponential)"
            if hasattr(self, 'bvals'):
                freq_list = [f"{f:.1f}" for f in self.bvals.cpu().numpy().tolist()]
                bvals_str = f"频率: [{', '.join(freq_list)}]"
            else:
                bvals_str = ""
        elif 'HE' in encoding_type:
            encoding_name = "Hash Encoding (多分辨率哈希编码)"
            if hasattr(self, 'hash_encoder'):
                bvals_str = (f"层级数: {self.hash_encoder.n_levels}, "
                           f"每层特征: {self.hash_encoder.n_features_per_level}, "
                           f"哈希编码输出维度: {self.hash_encoder.output_dim}")
            else:
                bvals_str = ""
        elif 'FF' in encoding_type:
            encoding_name = "Fourier Features"
            bvals_str = f"特征数: {self.bvals_size}" if hasattr(self, 'bvals_size') else ""
        elif 'dino' in encoding_type:
            encoding_name = "DINO特征编码"
            bvals_str = ""
        else:
            encoding_name = "无编码（直接使用原始坐标）"
            bvals_str = ""
        
        print(f"[{module_name}] 输入编码信息:")
        print(f"  - 编码类型: {encoding_name}")
        print(f"  - 原始坐标维度: {in_features}")
        print(f"  - 编码后输入维度: {self.first_layer_dim}")
        if bvals_str:
            print(f"  - {bvals_str}")
        print(f"  - 维度变化: {in_features}维 -> {self.first_layer_dim}维")

    def forward(self, coords, **kwargs):
        # Apply input encoding if any
        if 'FF' in self.input_encoding:
            x = (2*np.pi*coords) @ self.bvals.T # (1, n, bvals_size)
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1) / np.sqrt(self.bvals_size) # (1, n, bvals_size*2)
            x = torch.cat([coords, x], axis=-1) # (1, n, bvals_size*2 + d)
        elif 'PE' in self.input_encoding:
            x = coords[..., None] * self.bvals  # (1,n,d,num_scales)
            x = x.reshape(*x.shape[:-2], -1)  # (1,n,d*num_scales)
            x = torch.sin(torch.cat([x, x + np.pi / 2.0], dim=-1)) # (1,n,2*d*num_scales)
            x = torch.cat([coords, x], axis=-1)  # (1,n,d+2*d*num_scales)
        elif 'dino' in self.input_encoding:
            dino = kwargs['dino']
            x = torch.cat([coords, dino], axis=-1)  # (1,n,d+2*d*num_scales)
        elif 'HE' in self.input_encoding:
            # Hash Encoding
            hash_features = self.hash_encoder(coords)  # (B, N, hash_dim)
            x = torch.cat([coords, hash_features], dim=-1)  # (B, N, in_dim + hash_dim)
        else:
            x = coords

        if 'learned' in self.input_encoding:
            if isinstance(x, (list, tuple)):
                x = [self.encoder(x_i) for x_i in x]
            else:
                x = self.encoder(x)
            if 'cat' in self.input_encoding:
                if isinstance(x, (list, tuple)):
                    x[0] = torch.cat([coords, x[0]], axis=-1)
                else:
                    x = torch.cat([coords, x], axis=-1)
        
        return x

class tSoftMax(nn.Module):
    def __init__(self, temperature, dim=-1, trainable=False):
        super().__init__()
        if trainable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        self.dim = dim
        self.activation = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.activation(x / self.temperature)

class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
