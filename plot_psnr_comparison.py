#!/usr/bin/env python3
"""
绘制两个Excel文件的PSNR对比曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_psnr_comparison(aba_file='aba.xlsx', ours_file='ours.xlsx', 
                        save_path='psnr_comparison.png', 
                        aba_label='Ablation', ours_label='Ours'):
    """
    绘制两个Excel文件的PSNR对比曲线
    
    Args:
        aba_file: 第一个Excel文件路径（默认: aba.xlsx）
        ours_file: 第二个Excel文件路径（默认: ours.xlsx）
        save_path: 保存图片的路径（默认: psnr_comparison.png）
        aba_label: 第一条曲线的标签（默认: Ablation）
        ours_label: 第二条曲线的标签（默认: Ours）
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
    
    # 检查必要的列是否存在
    required_cols = ['epoch', 'psnr']
    for col in required_cols:
        if col not in df_aba.columns:
            print(f"错误: {aba_file} 中缺少列: {col}")
            return
        if col not in df_ours.columns:
            print(f"错误: {ours_file} 中缺少列: {col}")
            return
    
    # 提取数据
    epochs_aba = df_aba['epoch'].values
    psnr_aba = df_aba['psnr'].values
    
    epochs_ours = df_ours['epoch'].values
    psnr_ours = df_ours['psnr'].values
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(epochs_aba, psnr_aba, marker='o', label=aba_label, 
             linewidth=2, markersize=4, alpha=0.8)
    plt.plot(epochs_ours, psnr_ours, marker='s', label=ours_label, 
             linewidth=2, markersize=4, alpha=0.8)
    
    # 设置标签和标题
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('PSNR Comparison: Ablation vs Ours', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置坐标轴
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # 调整布局
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
    print(f"  提升: {psnr_ours[-1] - psnr_aba[-1]:.2f} dB")


if __name__ == '__main__':
    import sys
    
    # 如果提供了命令行参数
    if len(sys.argv) >= 3:
        aba_file = sys.argv[1]
        ours_file = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else 'psnr_comparison.png'
    else:
        # 使用默认值
        aba_file = 'aba.xlsx'
        ours_file = 'ours.xlsx'
        save_path = 'psnr_comparison.png'
    
    plot_psnr_comparison(aba_file, ours_file, save_path)
