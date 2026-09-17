"""
哈希编码演示和测试脚本
展示如何在Neural-Experts框架中使用哈希编码
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from hash_encoding import HashEncoding


def demo_2d_image_fitting():
    """演示：使用哈希编码拟合2D图像"""
    print("=" * 60)
    print("演示：2D图像重建 - 使用哈希编码")
    print("=" * 60)
    
    # 创建简单的测试图像（棋盘格）
    resolution = 128
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (N, 2)
    
    # 创建棋盘格图像
    img = ((torch.floor(xx * 8) + torch.floor(yy * 8)) % 2).float()
    gt_values = img.reshape(-1, 1)  # (N, 1)
    
    # 初始化哈希编码
    hash_enc = HashEncoding(
        n_levels=12,
        n_features_per_level=2,
        log2_hashmap_size=18,
        base_resolution=16,
        finest_resolution=256,
        input_dim=2
    )
    
    # 简单的MLP decoder
    mlp = torch.nn.Sequential(
        torch.nn.Linear(2 + 12 * 2, 64),  # input_dim + hash_features
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    
    # 优化器
    optimizer = torch.optim.Adam(
        list(hash_enc.parameters()) + list(mlp.parameters()),
        lr=1e-3
    )
    
    # 训练
    print("\n开始训练...")
    num_iterations = 1000
    batch_size = 4096
    
    for iteration in range(num_iterations):
        # 随机采样
        indices = torch.randint(0, coords.shape[0], (batch_size,))
        batch_coords = coords[indices].unsqueeze(0)  # (1, B, 2)
        batch_gt = gt_values[indices]  # (B, 1)
        
        # 前向传播
        hash_features = hash_enc(batch_coords)  # (1, B, L*F)
        features = torch.cat([batch_coords, hash_features], dim=-1)  # (1, B, 2+L*F)
        pred = mlp(features.squeeze(0))  # (B, 1)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(pred, batch_gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d} | Loss: {loss.item():.6f}")
    
    # 推理
    print("\n开始推理...")
    with torch.no_grad():
        hash_features = hash_enc(coords.unsqueeze(0))
        features = torch.cat([coords.unsqueeze(0), hash_features], dim=-1)
        pred_img = mlp(features.squeeze(0))
    
    # 计算PSNR
    mse = torch.nn.functional.mse_loss(pred_img, gt_values)
    psnr = -10 * torch.log10(mse)
    print(f"\n最终PSNR: {psnr.item():.2f} dB")
    
    # 可视化（如果matplotlib可用）
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        gt_img_display = gt_values.reshape(resolution, resolution).numpy()
        pred_img_display = pred_img.reshape(resolution, resolution).numpy()
        error = torch.abs(pred_img - gt_values).reshape(resolution, resolution).numpy()
        
        axes[0].imshow(gt_img_display, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(pred_img_display, cmap='gray')
        axes[1].set_title(f'Prediction (PSNR: {psnr.item():.2f} dB)')
        axes[1].axis('off')
        
        axes[2].imshow(error, cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('hash_encoding_demo_2d.png', dpi=150)
        print("\n结果已保存到: hash_encoding_demo_2d.png")
    except Exception as e:
        print(f"\n可视化失败: {e}")
    
    print("\n" + "=" * 60)
    return hash_enc, mlp


def compare_encodings():
    """对比不同编码方式的性能"""
    print("\n" + "=" * 60)
    print("编码方式性能对比")
    print("=" * 60)
    
    n_points = 10000
    coords_2d = torch.rand(1, n_points, 2)  # (1, N, 2)
    coords_3d = torch.rand(1, n_points, 3)  # (1, N, 3)
    
    # 1. 哈希编码
    print("\n1. 哈希编码 (Hash Encoding)")
    hash_enc_2d = HashEncoding(
        n_levels=16, n_features_per_level=2,
        input_dim=2, log2_hashmap_size=19
    )
    import time
    start = time.time()
    output_2d = hash_enc_2d(coords_2d)
    hash_time_2d = time.time() - start
    
    hash_enc_3d = HashEncoding(
        n_levels=16, n_features_per_level=2,
        input_dim=3, log2_hashmap_size=19
    )
    start = time.time()
    output_3d = hash_enc_3d(coords_3d)
    hash_time_3d = time.time() - start
    
    print(f"  2D: 输入 {coords_2d.shape} → 输出 {output_2d.shape}")
    print(f"      推理时间: {hash_time_2d*1000:.2f} ms")
    print(f"      参数量: {sum(p.numel() for p in hash_enc_2d.parameters()):,}")
    print(f"  3D: 输入 {coords_3d.shape} → 输出 {output_3d.shape}")
    print(f"      推理时间: {hash_time_3d*1000:.2f} ms")
    print(f"      参数量: {sum(p.numel() for p in hash_enc_3d.parameters()):,}")
    
    # 2. 傅里叶特征 (用于对比)
    print("\n2. 傅里叶特征 (Fourier Features)")
    freq_2d = torch.randn(16, 2) * 1.0
    start = time.time()
    x_2d = (2 * np.pi * coords_2d) @ freq_2d.T
    ff_output_2d = torch.cat([torch.sin(x_2d), torch.cos(x_2d)], dim=-1)
    ff_time_2d = time.time() - start
    
    freq_3d = torch.randn(16, 3) * 1.0
    start = time.time()
    x_3d = (2 * np.pi * coords_3d) @ freq_3d.T
    ff_output_3d = torch.cat([torch.sin(x_3d), torch.cos(x_3d)], dim=-1)
    ff_time_3d = time.time() - start
    
    print(f"  2D: 输入 {coords_2d.shape} → 输出 {ff_output_2d.shape}")
    print(f"      推理时间: {ff_time_2d*1000:.2f} ms")
    print(f"      参数量: {freq_2d.numel():,} (不可学习)")
    print(f"  3D: 输入 {coords_3d.shape} → 输出 {ff_output_3d.shape}")
    print(f"      推理时间: {ff_time_3d*1000:.2f} ms")
    print(f"      参数量: {freq_3d.numel():,} (不可学习)")
    
    print("\n性能总结:")
    print(f"  哈希编码特点: 可学习参数，自适应多分辨率，内存高效")
    print(f"  傅里叶特征特点: 固定频率，速度快但表达能力有限")
    print("=" * 60)


def visualize_hash_levels():
    """可视化哈希编码的多分辨率层级"""
    print("\n" + "=" * 60)
    print("哈希编码多分辨率层级可视化")
    print("=" * 60)
    
    hash_enc = HashEncoding(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
        input_dim=2
    )
    
    info = hash_enc.get_params_summary()
    
    print(f"\n配置参数:")
    print(f"  层数: {info['n_levels']}")
    print(f"  每层特征维度: {info['n_features_per_level']}")
    print(f"  哈希表大小: {info['hashmap_size']:,}")
    print(f"  基础分辨率: {info['base_resolution']}")
    print(f"  最高分辨率: {info['finest_resolution']}")
    print(f"  增长因子: {info['growth_factor']:.4f}")
    print(f"  总参数量: {info['total_params']:,}")
    
    print(f"\n各层分辨率分布:")
    for level in range(info['n_levels']):
        resolution = int(np.floor(
            info['base_resolution'] * (info['growth_factor'] ** level)
        ))
        print(f"  Level {level:2d}: {resolution:4d} x {resolution:4d}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n🚀 哈希编码演示程序")
    print("=" * 60)
    
    # 1. 可视化层级信息
    visualize_hash_levels()
    
    # 2. 性能对比
    compare_encodings()
    
    # 3. 实际拟合演示（2D图像）
    print("\n是否运行2D图像拟合演示？（需要约1-2分钟）")
    print("注释掉下面这行可以跳过演示")
    # demo_2d_image_fitting()  # 取消注释以运行
    
    print("\n✅ 所有测试完成！")
    print("\n使用建议:")
    print("1. 在配置文件中设置 decoder_input_encoding: 'HE'")
    print("2. 根据数据分辨率调整 hash_finest_resolution")
    print("3. 高细节场景可增大 hash_n_levels 和 hash_log2_hashmap_size")
    print("4. 参考 configs/README_hash_encoding.md 了解详细配置")

