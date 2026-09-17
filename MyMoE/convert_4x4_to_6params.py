#!/usr/bin/env python3
"""
将4×4齐次变换矩阵转换为6参数格式（3平移+3欧拉角）
输入：Excel文件，每行包含16个数字（4×4矩阵按行展开）
输出：Excel文件，格式为 [id, tx, ty, tz, rx, ry, rz]
"""

import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation


def matrix_to_euler_zyx(R):
    """
    将3×3旋转矩阵转换为欧拉角（弧度）
    对应代码中的 Rz @ Ry @ Rx 顺序（intrinsic XYZ）
    
    代码中的实现：
    R = Rz @ Ry @ Rx
    这是intrinsic XYZ rotations（在旋转后的坐标系中旋转）
    
    关键点：
    - 代码期望的顺序：[rx, ry, rz]
    - 代码构建方式：Rz @ Ry @ Rx（先X，再Y，最后Z）
    - 需要手动实现intrinsic XYZ的提取公式
    
    Args:
        R: 3×3旋转矩阵
    
    Returns:
        [rx, ry, rz]: 欧拉角（弧度），顺序与代码期望一致
    """
    # 首先验证旋转矩阵的有效性
    det = np.linalg.det(R)
    if abs(det - 1.0) > 0.01:
        print(f"警告: 旋转矩阵行列式={det:.6f}，不是有效的旋转矩阵（期望≈1.0）")
    
    # 对于 R = Rz(γ) @ Ry(β) @ Rx(α) (intrinsic XYZ)
    # 需要从矩阵中提取 [α, β, γ] = [rx, ry, rz]
    
    # 从 R = Rz @ Ry @ Rx 提取欧拉角
    # R[2,0] = -sin(β)
    # R[2,1] = cos(β) * sin(α)
    # R[2,2] = cos(β) * cos(α)
    # R[0,0] = cos(β) * cos(γ)
    # R[1,0] = cos(β) * sin(γ)
    
    try:
        # 提取 β (ry)
        sy = -R[2, 0]  # sin(β)
        cy = np.sqrt(max(0, 1 - sy**2))  # cos(β)，使用max避免数值误差
        if abs(cy) < 1e-6:
            # 万向锁情况：β ≈ ±90度
            # 此时只能确定 α + γ 或 α - γ
            print("警告: 检测到万向锁情况（ry ≈ ±90°）")
            ry = np.arcsin(np.clip(sy, -1, 1))
            # 使用备选公式
            rx = np.arctan2(R[1, 2], R[0, 2])
            rz = 0.0  # 任意值，通常设为0
        else:
            # 正常情况
            ry = np.arcsin(np.clip(sy, -1, 1))
            
            # 提取 α (rx)
            rx = np.arctan2(R[2, 1] / cy, R[2, 2] / cy)
            
            # 提取 γ (rz)
            rz = np.arctan2(R[1, 0] / cy, R[0, 0] / cy)
        
        return np.array([rx, ry, rz])
        
    except Exception as e:
        # 如果手动方法失败，尝试使用scipy（可能顺序不对，但作为备选）
        try:
            rot = Rotation.from_matrix(R)
            # 尝试不同的顺序
            euler_xyz = rot.as_euler('xyz', degrees=False)  # 可能返回[rx, ry, rz]
            return euler_xyz
        except:
            # 最后尝试zyx顺序
            try:
                rot = Rotation.from_matrix(R)
                euler_zyx = rot.as_euler('zyx', degrees=False)  # 返回[rz, ry, rx]
                return np.array([euler_zyx[2], euler_zyx[1], euler_zyx[0]])
            except Exception as e2:
                raise ValueError(f"无法从旋转矩阵提取欧拉角: {e}\nscipy方法也失败: {e2}\n矩阵:\n{R}")


def extract_pose_from_4x4(matrix_4x4):
    """
    从4×4齐次变换矩阵提取平移和旋转
    
    Args:
        matrix_4x4: 4×4齐次变换矩阵
    
    Returns:
        trans: 3×1平移向量 [tx, ty, tz]
        euler: 3×1欧拉角 [rx, ry, rz] (弧度)
    """
    # 提取旋转矩阵（左上角3×3）
    R = matrix_4x4[:3, :3]
    
    # 提取平移向量（第4列的前3个元素）
    T = matrix_4x4[:3, 3]
    
    # 将旋转矩阵转换为欧拉角（ZYX顺序）
    euler = matrix_to_euler_zyx(R)
    
    return T, euler


def convert_excel_4x4_to_6params(input_file, output_file=None, matrix_start_col=0):
    """
    将包含4×4矩阵的Excel文件转换为6参数格式
    
    Args:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径（如果为None，则在输入文件名后加_6params）
        matrix_start_col: 4×4矩阵数据开始的列索引（默认0，即从第一列开始）
    """
    print(f"读取文件: {input_file}")
    
    # 读取Excel文件
    df = pd.read_excel(input_file)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查数据格式
    n_cols = len(df.columns)
    available_cols = n_cols - matrix_start_col
    
    if available_cols < 16:
        raise ValueError(
            f"可用列数不足16列，无法构成4×4矩阵。\n"
            f"总列数: {n_cols}, 起始列: {matrix_start_col}, 可用列数: {available_cols}\n"
            f"请检查数据格式，或使用 matrix_start_col 参数指定矩阵数据的起始列"
        )
    
    # 提取4×4矩阵的列（从指定列开始，取16列）
    matrix_cols = df.columns[matrix_start_col:matrix_start_col+16].tolist()
    print(f"使用列 {matrix_start_col} 到 {matrix_start_col+15} 作为4×4矩阵数据")
    print(f"矩阵列名: {matrix_cols[:4]} ... {matrix_cols[-4:]}")
    
    # 存储结果
    results = []
    
    print("\n开始转换...")
    for idx, row in df.iterrows():
        # 提取4×4矩阵
        matrix_flat = row[matrix_cols].values.astype(np.float32)
        
        # 重塑为4×4矩阵（按行展开）
        matrix_4x4 = matrix_flat.reshape(4, 4)
        
        # 提取平移和旋转
        try:
            trans, euler = extract_pose_from_4x4(matrix_4x4)
            
            # 存储结果
            results.append({
                'id': idx,
                'tx': trans[0],
                'ty': trans[1],
                'tz': trans[2],
                'rx': euler[0],  # 弧度
                'ry': euler[1],  # 弧度
                'rz': euler[2]   # 弧度
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  已处理 {idx + 1}/{len(df)} 行")
                
        except Exception as e:
            print(f"警告: 第 {idx} 行处理失败: {e}")
            # 使用零值作为占位符
            results.append({
                'id': idx,
                'tx': 0.0,
                'ty': 0.0,
                'tz': 0.0,
                'rx': 0.0,
                'ry': 0.0,
                'rz': 0.0
            })
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(results)
    
    # 设置输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_6params.xlsx"
    
    # 保存结果
    output_df.to_excel(output_file, index=False)
    print(f"\n转换完成！")
    print(f"输出文件: {output_file}")
    print(f"输出数据形状: {output_df.shape}")
    print(f"\n前5行数据预览:")
    print(output_df.head())
    
    # 验证：检查数据范围
    print(f"\n数据统计:")
    print(f"平移范围: tx=[{output_df['tx'].min():.4f}, {output_df['tx'].max():.4f}]")
    print(f"         ty=[{output_df['ty'].min():.4f}, {output_df['ty'].max():.4f}]")
    print(f"         tz=[{output_df['tz'].min():.4f}, {output_df['tz'].max():.4f}]")
    print(f"旋转范围: rx=[{output_df['rx'].min():.4f}, {output_df['rx'].max():.4f}] (弧度)")
    print(f"         ry=[{output_df['ry'].min():.4f}, {output_df['ry'].max():.4f}] (弧度)")
    print(f"         rz=[{output_df['rz'].min():.4f}, {output_df['rz'].max():.4f}] (弧度)")
    
    # 验证转换正确性（检查前几行）
    print(f"\n验证转换正确性（检查前3行）...")
    for i in range(min(3, len(df))):
        # 原始矩阵
        matrix_flat = df.iloc[i][matrix_cols].values.astype(np.float32)
        original_matrix = matrix_flat.reshape(4, 4)
        original_R = original_matrix[:3, :3]
        original_T = original_matrix[:3, 3]
        
        # 检查原始旋转矩阵的有效性
        det = np.linalg.det(original_R)
        is_orthogonal = np.allclose(original_R @ original_R.T, np.eye(3), atol=1e-3)
        
        # 转换后的参数
        converted_rx = output_df.iloc[i]['rx']
        converted_ry = output_df.iloc[i]['ry']
        converted_rz = output_df.iloc[i]['rz']
        converted_T = np.array([output_df.iloc[i]['tx'], 
                               output_df.iloc[i]['ty'], 
                               output_df.iloc[i]['tz']])
        
        # 重建旋转矩阵（使用代码中的方法）
        cx, sx = np.cos(converted_rx), np.sin(converted_rx)
        cy, sy = np.cos(converted_ry), np.sin(converted_ry)
        cz, sz = np.cos(converted_rz), np.sin(converted_rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        reconstructed_R = Rz @ Ry @ Rx
        
        # 检查误差
        R_error = np.abs(original_R - reconstructed_R).max()
        R_error_mean = np.abs(original_R - reconstructed_R).mean()
        T_error = np.abs(original_T - converted_T).max()
        
        if R_error < 1e-3 and T_error < 1e-4:
            print(f"  ✓ 第{i}行: 旋转矩阵误差={R_error:.2e} (平均={R_error_mean:.2e}), 平移误差={T_error:.2e}")
        else:
            print(f"  ⚠ 第{i}行: 旋转矩阵误差={R_error:.2e} (平均={R_error_mean:.2e}), 平移误差={T_error:.2e}")
            print(f"     原始矩阵行列式={det:.6f}, 是否正交={is_orthogonal}")
            print(f"     欧拉角: rx={converted_rx:.4f}, ry={converted_ry:.4f}, rz={converted_rz:.4f} (弧度)")
            print(f"     原始R[2,0]={original_R[2,0]:.6f}, 重建R[2,0]={reconstructed_R[2,0]:.6f}")
    
    return output_file


if __name__ == '__main__':
    import sys
    
    # 输入文件路径
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'LH_Par_C_DtP.xlsx'
    
    # 可选：指定矩阵数据起始列（如果Excel文件有其他列在前面）
    matrix_start_col = 0
    if len(sys.argv) > 2:
        matrix_start_col = int(sys.argv[2])
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 文件不存在: {input_file}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"\n使用方法:")
        print(f"  python convert_4x4_to_6params.py <输入文件> [矩阵起始列]")
        print(f"\n示例:")
        print(f"  python convert_4x4_to_6params.py LH_Par_C_DtP.xlsx")
        print(f"  python convert_4x4_to_6params.py LH_Par_C_DtP.xlsx 0")
        sys.exit(1)
    else:
        # 执行转换
        try:
            output_file = convert_excel_4x4_to_6params(input_file, matrix_start_col=matrix_start_col)
            print(f"\n✅ 转换成功！输出文件: {output_file}")
            print(f"\n📝 注意:")
            print(f"  - 输出的欧拉角单位为弧度")
            print(f"  - 如果代码中设置了 angles_in_degrees=True，需要将角度转换为度数")
        except Exception as e:
            print(f"\n❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

