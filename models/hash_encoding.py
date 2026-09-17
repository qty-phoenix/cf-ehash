"""
多分辨率哈希编码 (Multi-resolution Hash Encoding)
基于NVIDIA Instant-NGP论文实现
参考: https://nvlabs.github.io/instant-ngp/
"""

import torch
import torch.nn as nn
import numpy as np


class HashEncoding(nn.Module):
    """
    多分辨率哈希编码
    
    参数:
        n_levels: 哈希表层数（分辨率级别数）
        n_features_per_level: 每个层级的特征维度
        log2_hashmap_size: 哈希表大小的log2值（实际大小为2^log2_hashmap_size）
        base_resolution: 最粗糙层级的网格分辨率
        finest_resolution: 最精细层级的网格分辨率
        input_dim: 输入坐标维度（2D或3D）
    """
    
    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        input_dim=2,
    ):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.input_dim = input_dim
        self.output_dim = n_levels * n_features_per_level
        
        # 哈希表大小
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # 计算每个层级的分辨率增长因子
        # b = exp((ln(N_max) - ln(N_min)) / (L-1))
        self.growth_factor = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )
        
        # 初始化哈希表参数 - 每个层级一个哈希表
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        
        # 初始化哈希表权重 - 使用更合理的范围
        # 参考Instant-NGP和torch-ngp实现，使用标准的均匀初始化
        for embedding in self.embeddings:
            nn.init.uniform_(embedding.weight, -1e-2, 1e-2)  # 增大初始化范围
        
        # 哈希函数的素数（用于空间哈希）
        # 对于2D和3D分别使用不同的素数
        if input_dim == 2:
            self.primes = [1, 2654435761]  # 2D
        elif input_dim == 3:
            self.primes = [1, 2654435761, 805459861]  # 3D
        else:
            raise ValueError(f"Unsupported input_dim: {input_dim}")
    
    def spatial_hash(self, coords, resolution):
        """
        空间哈希函数：将整数坐标映射到哈希表索引
        
        参数:
            coords: 整数坐标 (..., n_points, input_dim)
            resolution: 当前层级的分辨率
        
        返回:
            哈希索引 (..., n_points)
        """
        # 确保坐标在有效范围内
        coords = coords.long()
        
        # 空间哈希: h(x,y,z) = (x*p1 ⊕ y*p2 ⊕ z*p3) mod T
        # 其中 ⊕ 是异或操作, T 是哈希表大小
        hash_val = torch.zeros_like(coords[..., 0])
        
        for i in range(self.input_dim):
            hash_val ^= coords[..., i] * self.primes[i]
        
        return hash_val % self.hashmap_size
    
    def get_vertex_embeddings(self, coords, level):
        """
        获取网格顶点的嵌入向量
        
        参数:
            coords: 输入坐标 (..., n_points, input_dim), 范围[0, 1]
            level: 当前层级索引
        
        返回:
            顶点嵌入 (..., n_points, 2^input_dim, n_features_per_level)
        """
        # 计算当前层级的分辨率
        resolution = int(np.floor(self.base_resolution * (self.growth_factor ** level)))
        
        # 将归一化坐标映射到网格坐标
        grid_coords = coords * resolution
        
        # 获取网格单元的底部顶点
        grid_coords_floor = torch.floor(grid_coords)
        
        # 计算插值权重
        interp_weights = grid_coords - grid_coords_floor
        
        # 获取所有顶点的坐标（2D: 4个顶点, 3D: 8个顶点）
        n_vertices = 2 ** self.input_dim
        vertex_embeddings_list = []
        
        if self.input_dim == 2:
            # 2D情况：4个顶点
            for i in range(2):
                for j in range(2):
                    offset = torch.stack([
                        torch.full_like(grid_coords_floor[..., 0], i),
                        torch.full_like(grid_coords_floor[..., 1], j)
                    ], dim=-1)
                    vertex_coords = grid_coords_floor + offset
                    
                    # 处理边界情况
                    vertex_coords = torch.clamp(vertex_coords, 0, resolution - 1)
                    
                    # 计算哈希索引
                    hash_idx = self.spatial_hash(vertex_coords, resolution)
                    
                    # 查找嵌入
                    embedding = self.embeddings[level](hash_idx)
                    vertex_embeddings_list.append(embedding)
        
        elif self.input_dim == 3:
            # 3D情况：8个顶点
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        offset = torch.stack([
                            torch.full_like(grid_coords_floor[..., 0], i),
                            torch.full_like(grid_coords_floor[..., 1], j),
                            torch.full_like(grid_coords_floor[..., 2], k)
                        ], dim=-1)
                        vertex_coords = grid_coords_floor + offset
                        
                        # 处理边界情况
                        vertex_coords = torch.clamp(vertex_coords, 0, resolution - 1)
                        
                        # 计算哈希索引
                        hash_idx = self.spatial_hash(vertex_coords, resolution)
                        
                        # 查找嵌入
                        embedding = self.embeddings[level](hash_idx)
                        vertex_embeddings_list.append(embedding)
        
        # 堆叠所有顶点的嵌入
        vertex_embeddings = torch.stack(vertex_embeddings_list, dim=-2)  # (..., n_points, n_vertices, F)
        
        return vertex_embeddings, interp_weights
    
    def trilinear_interpolation(self, vertex_embeddings, weights):
        """
        三线性插值（2D为双线性插值）
        
        参数:
            vertex_embeddings: 顶点嵌入 (..., n_points, n_vertices, n_features)
            weights: 插值权重 (..., n_points, input_dim)
        
        返回:
            插值结果 (..., n_points, n_features)
        """
        if self.input_dim == 2:
            # 双线性插值
            # weights: (..., n_points, 2) -> (wx, wy)
            wx = weights[..., 0:1]  # (..., n_points, 1)
            wy = weights[..., 1:2]  # (..., n_points, 1)
            
            # vertex_embeddings顺序: (0,0), (0,1), (1,0), (1,1)
            c00 = vertex_embeddings[..., 0, :]  # (..., n_points, F)
            c01 = vertex_embeddings[..., 1, :]
            c10 = vertex_embeddings[..., 2, :]
            c11 = vertex_embeddings[..., 3, :]
            
            # 双线性插值公式
            result = (
                (1 - wx) * (1 - wy) * c00 +
                (1 - wx) * wy * c01 +
                wx * (1 - wy) * c10 +
                wx * wy * c11
            )
            
        elif self.input_dim == 3:
            # 三线性插值
            wx = weights[..., 0:1]  # (..., n_points, 1)
            wy = weights[..., 1:2]
            wz = weights[..., 2:3]
            
            # vertex_embeddings顺序: (i,j,k) for i,j,k in {0,1}
            c000 = vertex_embeddings[..., 0, :]
            c001 = vertex_embeddings[..., 1, :]
            c010 = vertex_embeddings[..., 2, :]
            c011 = vertex_embeddings[..., 3, :]
            c100 = vertex_embeddings[..., 4, :]
            c101 = vertex_embeddings[..., 5, :]
            c110 = vertex_embeddings[..., 6, :]
            c111 = vertex_embeddings[..., 7, :]
            
            # 三线性插值公式
            result = (
                (1 - wx) * (1 - wy) * (1 - wz) * c000 +
                (1 - wx) * (1 - wy) * wz * c001 +
                (1 - wx) * wy * (1 - wz) * c010 +
                (1 - wx) * wy * wz * c011 +
                wx * (1 - wy) * (1 - wz) * c100 +
                wx * (1 - wy) * wz * c101 +
                wx * wy * (1 - wz) * c110 +
                wx * wy * wz * c111
            )
        
        return result
    
    def forward(self, coords):
        """
        前向传播
        
        参数:
            coords: 输入坐标 (..., n_points, input_dim)
                   可以是[-1, 1]或[0, 1]范围，会自动转换
        
        返回:
            哈希编码特征 (..., n_points, n_levels * n_features_per_level)
        """
        # 自动检测并转换坐标范围（不修改原始coords）
        # 如果坐标在[-1, 1]范围，转换到[0, 1]
        coords_normalized = coords.clone()  # 避免修改原始输入
        
        if coords_normalized.min() < -0.5:  # 假设[-1,1]范围
            coords_normalized = (coords_normalized + 1.0) / 2.0  # 从[-1,1]转换到[0,1]
        
        # 确保坐标在[0, 1]范围内
        coords_normalized = torch.clamp(coords_normalized, 0.0, 1.0)
        
        # 存储所有层级的特征
        level_features = []
        
        # 对每个层级进行哈希编码
        for level in range(self.n_levels):
            # 获取顶点嵌入和插值权重
            vertex_embeddings, interp_weights = self.get_vertex_embeddings(coords_normalized, level)
            
            # 执行插值
            level_feature = self.trilinear_interpolation(vertex_embeddings, interp_weights)
            
            level_features.append(level_feature)
        
        # 拼接所有层级的特征
        output = torch.cat(level_features, dim=-1)  # (..., n_points, L*F)
        
        return output
    
    def get_params_summary(self):
        """返回参数摘要信息"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'n_levels': self.n_levels,
            'n_features_per_level': self.n_features_per_level,
            'hashmap_size': self.hashmap_size,
            'base_resolution': self.base_resolution,
            'finest_resolution': self.finest_resolution,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'total_params': total_params,
            'growth_factor': self.growth_factor
        }


# 测试代码
if __name__ == "__main__":
    print("测试哈希编码模块...")
    
    # 2D测试
    print("\n=== 2D测试 ===")
    hash_enc_2d = HashEncoding(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        input_dim=2
    )
    
    # 创建测试输入
    coords_2d = torch.rand(1, 1000, 2)  # (batch, n_points, 2)
    output_2d = hash_enc_2d(coords_2d)
    
    print(f"输入形状: {coords_2d.shape}")
    print(f"输出形状: {output_2d.shape}")
    print(f"参数信息: {hash_enc_2d.get_params_summary()}")
    
    # 3D测试
    print("\n=== 3D测试 ===")
    hash_enc_3d = HashEncoding(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        input_dim=3
    )
    
    coords_3d = torch.rand(1, 1000, 3)  # (batch, n_points, 3)
    output_3d = hash_enc_3d(coords_3d)
    
    print(f"输入形状: {coords_3d.shape}")
    print(f"输出形状: {output_3d.shape}")
    print(f"参数信息: {hash_enc_3d.get_params_summary()}")
    
    print("\n✓ 哈希编码模块测试通过!")

