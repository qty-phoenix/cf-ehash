#!/usr/bin/env python3
"""
数据集性能对比测试脚本
对比原始版本vs缓存版本的加载速度
"""

import time
import torch
from torch.utils.data import DataLoader


def benchmark_dataset(dataset_class, name, num_iterations=100, num_workers=0):
    """测试数据集加载性能"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    # 创建数据集
    start_init = time.time()
    try:
        dataset = dataset_class(
            images_dir='./MyMoE/images',
            pose_file='./MyMoE/coords3d/mri.xlsx',
            grayscale=True,
            crop_size=160,
            angles_in_degrees=True,
            mode='train'
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    init_time = time.time() - start_init
    print(f"📊 数据集初始化: {init_time:.2f}秒")
    print(f"📊 数据集大小: {len(dataset)} 张图像")
    
    # 创建加载器
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # 不打乱，确保测试一致性
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 预热（避免冷启动影响）
    print(f"\n🔥 预热中...")
    for i, data in enumerate(loader):
        if i >= 5:
            break
    
    # 正式测试
    print(f"🏃 开始测试 {num_iterations} 次迭代...")
    start_time = time.time()
    
    for i, data in enumerate(loader):
        if i >= num_iterations:
            break
        
        # 模拟实际使用
        coords = data['coords']  # (1, H*W, 3)
        gt_img = data['gt_img']  # (1, H, W)
        
        # 确保数据已经加载
        _ = coords.shape
        _ = gt_img.shape
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 统计
    avg_time = elapsed / num_iterations * 1000  # 转换为毫秒
    throughput = num_iterations / elapsed
    
    print(f"\n📈 性能统计:")
    print(f"   总耗时: {elapsed:.2f} 秒")
    print(f"   平均每次: {avg_time:.2f} ms")
    print(f"   吞吐量: {throughput:.2f} samples/sec")
    
    return {
        'init_time': init_time,
        'total_time': elapsed,
        'avg_time_ms': avg_time,
        'throughput': throughput
    }


def main():
    print("🚀 数据集性能对比测试")
    print("=" * 60)
    
    # 测试参数
    num_iterations = 50  # 测试50次迭代
    num_workers = 0  # 单线程，确保公平对比
    
    results = {}
    
    # 测试1: 原始版本
    print("\n\n1️⃣ 测试原始版本（每次重新计算坐标）")
    try:
        from datasets.RGBPose3D_GlobalNorm import RGBPose3DDatasetGlobalNorm as OriginalDataset
        results['original'] = benchmark_dataset(
            OriginalDataset,
            "原始版本 (GlobalNorm)",
            num_iterations,
            num_workers
        )
    except Exception as e:
        print(f"❌ 原始版本测试失败: {e}")
        results['original'] = None
    
    # 测试2: 缓存版本
    print("\n\n2️⃣ 测试缓存版本（预计算坐标）")
    try:
        from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm as CachedDataset
        results['cached'] = benchmark_dataset(
            CachedDataset,
            "缓存版本 (GlobalNorm + Cache)",
            num_iterations,
            num_workers
        )
    except Exception as e:
        print(f"❌ 缓存版本测试失败: {e}")
        results['cached'] = None
    
    # 对比结果
    print("\n\n" + "="*60)
    print("📊 性能对比总结")
    print("="*60)
    
    if results['original'] and results['cached']:
        orig = results['original']
        cached = results['cached']
        
        print(f"\n⏱️  数据加载时间对比:")
        print(f"   原始版本: {orig['avg_time_ms']:.2f} ms/sample")
        print(f"   缓存版本: {cached['avg_time_ms']:.2f} ms/sample")
        speedup_load = orig['avg_time_ms'] / cached['avg_time_ms']
        print(f"   加速比: {speedup_load:.2f}x ⚡")
        
        print(f"\n🚀 吞吐量对比:")
        print(f"   原始版本: {orig['throughput']:.2f} samples/sec")
        print(f"   缓存版本: {cached['throughput']:.2f} samples/sec")
        speedup_throughput = cached['throughput'] / orig['throughput']
        print(f"   提升: {speedup_throughput:.2f}x ⚡")
        
        print(f"\n💾 初始化时间:")
        print(f"   原始版本: {orig['init_time']:.2f} 秒")
        print(f"   缓存版本: {cached['init_time']:.2f} 秒")
        if cached['init_time'] > orig['init_time']:
            print(f"   注意: 首次运行需要预计算，多出 {cached['init_time'] - orig['init_time']:.2f} 秒")
            print(f"         第二次运行将快速加载缓存")
        
        # 计算训练时间节省
        print(f"\n🎯 实际训练节省估算（500张图像 × 2000 epochs）:")
        time_per_epoch_orig = orig['avg_time_ms'] * 500 / 1000  # 秒
        time_per_epoch_cached = cached['avg_time_ms'] * 500 / 1000  # 秒
        saved_per_epoch = time_per_epoch_orig - time_per_epoch_cached
        total_saved = saved_per_epoch * 2000 / 3600  # 转换为小时
        
        print(f"   每个epoch节省: {saved_per_epoch:.1f} 秒")
        print(f"   总节省时间: {total_saved:.1f} 小时")
        print(f"   总训练加速: {speedup_load:.2f}x ⚡⚡⚡")
        
    else:
        print("\n⚠️  无法完成对比（某个版本测试失败）")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)


if __name__ == '__main__':
    main()


