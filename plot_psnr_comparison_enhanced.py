#!/usr/bin/env python3
"""
绘制两个Excel文件的PSNR对比曲线（增强版 - 凸显差异）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_psnr_comparison_enhanced(aba_file='aba.xlsx', ours_file='ours.xlsx', 
                                 save_path='psnr_comparison_enhanced.png',
                                 aba_label='Ablation', ours_label='Ours',
                                 method='difference'):
    """
    绘制两个Excel文件的PSNR对比曲线（增强版）
    
    Args:
        aba_file: 第一个Excel文件路径
        ours_file: 第二个Excel文件路径
        save_path: 保存图片的路径
        aba_label: 第一条曲线的标签
        ours_label: 第二条曲线的标签
        method: 凸显差异的方法
            - 'difference': 显示差值曲线
            - 'zoom': 放大Y轴范围（只显示差异区间）
            - 'dual_axis': 双Y轴（原始值+差值）
            - 'fill': 填充差值区域
            - 'relative': 显示相对提升百分比
    """
    # 检查文件是否存在
    if not os.path.exists(aba_file):
        print(f"错误: 文件不存在: {aba_file}")
        return
    
    if not os.path.exists(ours_file):
        print(f"错误: 文件不存在: {ours_file}")
        return
    
    # 读取Excel文件
    try:
        df_aba = pd.read_excel(aba_file)
        df_ours = pd.read_excel(ours_file)
    except Exception as e:
        print(f"错误: 读取Excel文件失败: {e}")
        return
    
    # 检查必要的列
    required_cols = ['epoch', 'psnr']
    for col in required_cols:
        if col not in df_aba.columns or col not in df_ours.columns:
            print(f"错误: Excel文件中缺少列: {col}")
            return
    
    # 提取数据
    epochs_aba = df_aba['epoch'].values
    psnr_aba = df_aba['psnr'].values
    
    epochs_ours = df_ours['epoch'].values
    psnr_ours = df_ours['psnr'].values
    
    # 计算差值
    # 对齐epoch（取交集）
    common_epochs = np.intersect1d(epochs_aba, epochs_ours)
    if len(common_epochs) == 0:
        print("错误: 两个文件的epoch没有交集")
        return
    
    # 对齐数据
    aba_dict = dict(zip(epochs_aba, psnr_aba))
    ours_dict = dict(zip(epochs_ours, psnr_ours))
    
    aligned_epochs = sorted(common_epochs)
    aligned_psnr_aba = np.array([aba_dict[e] for e in aligned_epochs])
    aligned_psnr_ours = np.array([ours_dict[e] for e in aligned_epochs])
    
    # 计算差值
    psnr_diff = aligned_psnr_ours - aligned_psnr_aba
    psnr_relative = (aligned_psnr_ours - aligned_psnr_aba) / aligned_psnr_aba * 100
    
    # 根据方法绘制
    if method == 'difference':
        # 方法1: 显示差值曲线
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        
        # 上图：原始曲线
        ax1.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
                linewidth=2, markersize=4, alpha=0.8, color='#FF6B6B')
        ax1.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
                linewidth=2, markersize=4, alpha=0.8, color='#4ECDC4')
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 下图：差值曲线
        ax2.plot(aligned_epochs, psnr_diff, marker='D', label='Difference (Ours - Ablation)', 
                linewidth=2.5, markersize=5, color='#2E86AB', alpha=0.9)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(aligned_epochs, 0, psnr_diff, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('PSNR Difference (dB)', fontsize=12)
        ax2.set_title('Performance Gain', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
    elif method == 'zoom':
        # 方法2: 放大Y轴范围
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
               linewidth=2, markersize=4, alpha=0.8)
        ax.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
               linewidth=2, markersize=4, alpha=0.8)
        
        # 计算Y轴范围（只显示差异区间）
        all_psnr = np.concatenate([psnr_aba, psnr_ours])
        y_min = all_psnr.min() - 0.5
        y_max = all_psnr.max() + 0.5
        
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Comparison (Zoomed)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
    elif method == 'dual_axis':
        # 方法3: 双Y轴
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 左Y轴：原始PSNR值
        ax1.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
               linewidth=2, markersize=4, alpha=0.8, color='#FF6B6B')
        ax1.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
               linewidth=2, markersize=4, alpha=0.8, color='#4ECDC4')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(fontsize=11, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 右Y轴：差值
        ax2 = ax1.twinx()
        ax2.plot(aligned_epochs, psnr_diff, marker='D', label='Difference', 
                linewidth=2.5, markersize=5, color='#2E86AB', alpha=0.9)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('PSNR Difference (dB)', fontsize=12, color='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#2E86AB')
        ax2.legend(fontsize=11, loc='upper right')
        
        plt.title('PSNR Comparison with Difference', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
    elif method == 'fill':
        # 方法4: 填充差值区域
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
               linewidth=2, markersize=4, alpha=0.8, color='#FF6B6B')
        ax.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
               linewidth=2, markersize=4, alpha=0.8, color='#4ECDC4')
        
        # 填充差值区域
        ax.fill_between(aligned_epochs, aligned_psnr_aba, aligned_psnr_ours, 
                       alpha=0.3, color='#2E86AB', label='Performance Gain')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title('PSNR Comparison with Gain Region', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
    elif method == 'relative':
        # 方法5: 显示相对提升百分比
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        
        # 上图：原始曲线
        ax1.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
                linewidth=2, markersize=4, alpha=0.8, color='#FF6B6B')
        ax1.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
                linewidth=2, markersize=4, alpha=0.8, color='#4ECDC4')
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 下图：相对提升百分比
        ax2.plot(aligned_epochs, psnr_relative, marker='D', 
                label='Relative Improvement (%)', 
                linewidth=2.5, markersize=5, color='#2E86AB', alpha=0.9)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(aligned_epochs, 0, psnr_relative, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Relative Improvement (%)', fontsize=12)
        ax2.set_title('Performance Gain (Relative)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已保存到: {save_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print(f"\n📊 统计信息:")
    print(f"{aba_label}:")
    print(f"  最大PSNR: {psnr_aba.max():.2f} dB (Epoch {epochs_aba[psnr_aba.argmax()]})")
    print(f"  最终PSNR: {psnr_aba[-1]:.2f} dB")
    print(f"{ours_label}:")
    print(f"  最大PSNR: {psnr_ours.max():.2f} dB (Epoch {epochs_ours[psnr_ours.argmax()]})")
    print(f"  最终PSNR: {psnr_ours[-1]:.2f} dB")
    print(f"\n🎯 性能提升:")
    print(f"  绝对提升: {psnr_ours[-1] - psnr_aba[-1]:.3f} dB")
    print(f"  相对提升: {(psnr_ours[-1] - psnr_aba[-1]) / psnr_aba[-1] * 100:.2f}%")
    print(f"  平均差值: {psnr_diff.mean():.3f} dB")
    print(f"  最大差值: {psnr_diff.max():.3f} dB (Epoch {aligned_epochs[psnr_diff.argmax()]})")


if __name__ == '__main__':
    import sys
    
    # 解析参数
    if len(sys.argv) >= 3:
        aba_file = sys.argv[1]
        ours_file = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else 'difference'
        save_path = sys.argv[4] if len(sys.argv) > 4 else f'psnr_comparison_{method}.png'
    else:
        aba_file = 'aba.xlsx'
        ours_file = 'ours.xlsx'
        method = 'difference'  # 默认使用差值曲线
        save_path = f'psnr_comparison_{method}.png'
    
    print(f"使用方法: {method}")
    plot_psnr_comparison_enhanced(aba_file, ours_file, save_path, method=method)
