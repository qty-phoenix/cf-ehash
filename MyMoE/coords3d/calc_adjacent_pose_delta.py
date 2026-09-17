#!/usr/bin/env python3
"""
计算Excel中相邻两帧4x4齐次变换矩阵之间的位姿差。

输入格式（默认）：
- 第1列：序号/帧ID
- 后16列：4x4矩阵按行展开

输出：
- 一个新的xlsx文件，记录每个相邻帧对的平移与旋转差
"""

import os
import sys
import numpy as np
import pandas as pd


def parse_transform_from_row(row_values):
    """将长度16的一维数组重塑为4x4矩阵。"""
    matrix_flat = np.asarray(row_values, dtype=np.float64)
    if matrix_flat.size != 16:
        raise ValueError(f"矩阵元素数量不是16，实际为 {matrix_flat.size}")
    matrix_4x4 = matrix_flat.reshape(4, 4)
    return matrix_4x4


def rotation_angle_from_matrix(rotation_matrix):
    """
    从3x3旋转矩阵计算旋转角（弧度）。
    angle = arccos((trace(R) - 1) / 2)
    """
    trace_val = np.trace(rotation_matrix)
    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def compute_adjacent_pose_deltas(input_file, output_file=None, matrix_start_col=1):
    """
    计算相邻两帧之间的相对位姿差。

    Args:
        input_file: 输入xlsx路径
        output_file: 输出xlsx路径（None时自动命名）
        matrix_start_col: 16列矩阵数据的起始列索引（默认1，跳过第1列ID）
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    df = pd.read_excel(input_file)
    if df.shape[0] < 2:
        raise ValueError("数据行数不足2行，无法计算相邻帧差值。")

    available_cols = df.shape[1] - matrix_start_col
    if available_cols < 16:
        raise ValueError(
            f"可用列数不足16列，无法构成4x4矩阵。"
            f" 总列数={df.shape[1]}, matrix_start_col={matrix_start_col}, 可用列数={available_cols}"
        )

    id_col = df.columns[0]
    matrix_cols = df.columns[matrix_start_col:matrix_start_col + 16]

    results = []
    num_rows = len(df)
    for i in range(num_rows - 1):
        row_a = df.iloc[i]
        row_b = df.iloc[i + 1]

        matrix_a = parse_transform_from_row(row_a[matrix_cols].values)
        matrix_b = parse_transform_from_row(row_b[matrix_cols].values)

        relative_transform = np.linalg.inv(matrix_a) @ matrix_b
        relative_rotation = relative_transform[:3, :3]
        relative_translation = relative_transform[:3, 3]

        translation_norm = np.linalg.norm(relative_translation)
        rotation_rad = rotation_angle_from_matrix(relative_rotation)
        rotation_deg = np.degrees(rotation_rad)

        results.append(
            {
                "pair_index": i,
                "from_id": row_a[id_col],
                "to_id": row_b[id_col],
                "dx": relative_translation[0],
                "dy": relative_translation[1],
                "dz": relative_translation[2],
                "translation_magnitude": translation_norm,
                "rotation_angle_rad": rotation_rad,
                "rotation_angle_deg": rotation_deg,
            }
        )

    out_df = pd.DataFrame(results)

    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}_adjacent_delta.xlsx"

    out_df.to_excel(output_file, index=False)
    return output_file, out_df


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "LH_Par_C_DtP.xlsx"

    if len(sys.argv) > 2:
        matrix_start_col = int(sys.argv[2])
    else:
        matrix_start_col = 1

    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        output_file = None

    try:
        output_path, out_df = compute_adjacent_pose_deltas(
            input_file=input_file,
            output_file=output_file,
            matrix_start_col=matrix_start_col,
        )
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_path}")
        print(f"共处理帧数: {len(out_df) + 1}")
        print(f"输出相邻帧对数: {len(out_df)}")
        print("\n前5行结果预览:")
        print(out_df.head())
    except Exception as exc:
        print(f"计算失败: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

