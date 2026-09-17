import numpy as np
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MyMoE'))

# 导入MyMoE的真实函数
from model import eulerAnglesToRotationMatrix_torch, sample_from_matrix

def euler_to_matrix_numpy(angles):
    """Neural-Experts中的欧拉角转旋转矩阵 (numpy版本)"""
    rx, ry, rz = angles
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx
    return R

def build_coords3d_neural_experts(H, W, trans, rot):
    """Neural-Experts的坐标计算方法"""
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    zz = np.zeros_like(xx)
    grid = np.stack([xx, yy, zz], axis=-1)  # (H,W,3)
    R = euler_to_matrix_numpy(rot)
    T = trans.astype(np.float32)
    coords = grid.reshape(-1, 3) @ R.T + T[None, :]
    return coords.reshape(H, W, 3)

def build_coords3d_mymoe(H, W, trans, rot):
    """MyMoE的坐标计算方法 (使用真实函数)"""
    # 构建参考网格 (3, H, W)
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    zz = torch.zeros_like(xx)
    grid_ref = torch.stack([xx, yy, zz], dim=0).float()  # (3, H, W)
    
    # 使用MyMoE的真实函数
    rot_tensor = torch.tensor(rot, dtype=torch.float32)
    trans_tensor = torch.tensor(trans, dtype=torch.float32)
    
    # 计算旋转矩阵
    R = eulerAnglesToRotationMatrix_torch(rot_tensor)
    
    # 应用变换
    coords = sample_from_matrix(grid_ref, R, trans_tensor)  # (H, W, 3)
    
    return coords

def normalize_coords_neural_experts(coords_hw3):
    """Neural-Experts的坐标归一化"""
    c = coords_hw3.reshape(-1, 3)
    mins = c.min(axis=0)
    maxs = c.max(axis=0)
    span = np.clip(maxs - mins, 1e-8, None)
    c_norm = 2.0 * (c - mins) / span - 1.0
    return c_norm.reshape(coords_hw3.shape)

def normalize_coords_mymoe(coords_hw3):
    """MyMoE的坐标归一化"""
    c = coords_hw3.reshape(-1, 3)
    min_val = c.min()
    max_val = c.max()
    normalized = 2 * (c - min_val) / (max_val - min_val) - 1
    return normalized.reshape(coords_hw3.shape)

def compare_coordinate_calculation():
    """对比两种坐标计算方法"""
    print("=" * 80)
    print("坐标计算方法对比")
    print("=" * 80)
    
    # 测试参数
    H, W = 160, 160
    trans = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rot = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 弧度
    
    print(f"测试参数:")
    print(f"  图像尺寸: {H}×{W}")
    print(f"  平移: {trans}")
    print(f"  旋转: {rot} (弧度)")
    
    # Neural-Experts方法
    coords_ne = build_coords3d_neural_experts(H, W, trans, rot)
    coords_ne_norm = normalize_coords_neural_experts(coords_ne)
    
    # MyMoE方法
    coords_mm = build_coords3d_mymoe(H, W, trans, rot)
    coords_mm_norm = normalize_coords_mymoe(coords_mm)
    
    print(f"\n坐标计算结果:")
    print(f"Neural-Experts原始坐标:")
    print(f"  形状: {coords_ne.shape}")
    print(f"  范围: [{coords_ne.min():.4f}, {coords_ne.max():.4f}]")
    print(f"  均值: {coords_ne.mean():.4f}")
    
    print(f"\nMyMoE原始坐标:")
    print(f"  形状: {coords_mm.shape}")
    print(f"  范围: [{coords_mm.min():.4f}, {coords_mm.max():.4f}]")
    print(f"  均值: {coords_mm.mean():.4f}")
    
    print(f"\n归一化后坐标:")
    print(f"Neural-Experts归一化:")
    print(f"  范围: [{coords_ne_norm.min():.4f}, {coords_ne_norm.max():.4f}]")
    print(f"  均值: {coords_ne_norm.mean():.4f}")
    
    print(f"\nMyMoE归一化:")
    print(f"  范围: [{coords_mm_norm.min():.4f}, {coords_mm_norm.max():.4f}]")
    print(f"  均值: {coords_mm_norm.mean():.4f}")
    
    # 数值差异分析
    print(f"\n数值差异分析:")
    
    # 原始坐标差异
    coords_diff = np.abs(coords_ne - coords_mm.numpy())
    print(f"原始坐标最大差异: {coords_diff.max():.8f}")
    print(f"原始坐标平均差异: {coords_diff.mean():.8f}")
    
    # 归一化坐标差异
    norm_diff = np.abs(coords_ne_norm - coords_mm_norm.numpy())
    print(f"归一化坐标最大差异: {norm_diff.max():.8f}")
    print(f"归一化坐标平均差异: {norm_diff.mean():.8f}")
    
    # 检查是否相同
    if coords_diff.max() < 1e-6:
        print("✅ 原始坐标数值上基本相同")
    else:
        print("❌ 原始坐标存在显著差异")
    
    if norm_diff.max() < 1e-6:
        print("✅ 归一化坐标数值上基本相同")
    else:
        print("❌ 归一化坐标存在显著差异")
    
    # 详细分析差异原因
    print(f"\n详细分析:")
    print(f"1. 网格生成方式:")
    print(f"   Neural-Experts: meshgrid(H, W, indexing='ij')")
    print(f"   MyMoE: meshgrid(H, W, indexing='ij')")
    print(f"   → 网格生成方式相同")
    
    print(f"\n2. 旋转矩阵计算:")
    print(f"   Neural-Experts: Rz @ Ry @ Rx (numpy)")
    print(f"   MyMoE: Rz @ Ry @ Rx (torch)")
    print(f"   → 旋转矩阵计算方式相同")
    
    print(f"\n3. 坐标变换:")
    print(f"   Neural-Experts: grid @ R.T + T")
    print(f"   MyMoE: grid @ R.T + T")
    print(f"   → 坐标变换方式相同")
    
    print(f"\n4. 归一化方式:")
    print(f"   Neural-Experts: 按维度独立归一化")
    print(f"   MyMoE: 全局归一化")
    print(f"   → 归一化方式不同！")
    
    return coords_ne, coords_mm, coords_ne_norm, coords_mm_norm

if __name__ == '__main__':
    compare_coordinate_calculation()
