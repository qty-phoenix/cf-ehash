#!/usr/bin/env python3
"""
测试单专家训练代码的正确性
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np

from models import build_model
from datasets.RGBPose3D import RGBPose3DDataset


def test_single_expert_model():
    """测试单专家模型的构建和前向传播"""
    print("测试单专家模型...")
    
    # 加载配置
    with open('configs/config_RGB_pose3d_single_expert.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 构建模型
    SINR, loss_fn = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")
    
    # 检查模型类型
    print(f"模型类型: {type(SINR).__name__}")
    print(f"是否为单专家模型: {type(SINR).__name__ == 'INR'}")
    
    SINR.to(device)
    SINR.eval()
    
    # 创建测试数据
    batch_size = 1
    n_points = 1000
    coords = torch.randn(batch_size, n_points, 3, device=device)
    gt_img = torch.randn(batch_size, 1, 32, 32, device=device)  # 灰度图像
    dino = torch.randn(batch_size, 768, device=device)
    
    print(f"输入坐标形状: {coords.shape}")
    print(f"真实图像形状: {gt_img.shape}")
    print(f"DINO特征形状: {dino.shape}")
    
    # 前向传播
    with torch.no_grad():
        output_pred = SINR(coords, dino=dino, img=gt_img)
    
    print(f"输出键: {list(output_pred.keys())}")
    
    # 检查输出形状
    nonmanifold_pnts_pred = output_pred.get('nonmanifold_pnts_pred', None)
    if nonmanifold_pnts_pred is not None:
        print(f"非流形点预测形状: {nonmanifold_pnts_pred.shape}")
        print(f"预测值范围: [{nonmanifold_pnts_pred.min().item():.6f}, {nonmanifold_pnts_pred.max().item():.6f}]")
    else:
        print("❌ 未找到nonmanifold_pnts_pred输出")
        return False
    
    manifold_pnts_pred = output_pred.get('manifold_pnts_pred', None)
    if manifold_pnts_pred is not None:
        print(f"流形点预测形状: {manifold_pnts_pred.shape}")
    else:
        print("ℹ️  未找到manifold_pnts_pred输出（这是正常的，因为没有提供流形点）")
    
    print("✅ 单专家模型测试通过!")
    return True


def test_loss_function():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    # 模拟输出和真实数据
    batch_size = 1
    n_points = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟模型输出
    output_pred = {
        'nonmanifold_pnts_pred': torch.randn(batch_size, 1, n_points, device=device)
    }
    
    # 模拟真实图像
    gt_img_flat = torch.randn(batch_size, n_points, 1, device=device)
    
    # 导入损失函数
    from train_rgb_pose3d_single_expert import compute_single_expert_loss
    
    # 计算损失
    loss_dict = compute_single_expert_loss(output_pred, gt_img_flat, recon_weight=1000.0)
    
    print(f"总损失: {loss_dict['loss'].item():.6f}")
    print(f"重建损失: {loss_dict['recon_loss'].item():.6f}")
    print(f"平衡损失: {loss_dict['balance_loss'].item():.6f}")
    print(f"专家使用率: {loss_dict['expert_usage'].item():.6f}")
    print(f"预测形状: {loss_dict['weighted_pred'].shape}")
    
    print("✅ 损失函数测试通过!")
    return True


def test_dataset_loading():
    """测试数据集加载"""
    print("\n测试数据集加载...")
    
    try:
        # 检查数据路径是否存在
        images_dir = './MyMoE/images'
        pose_file = './MyMoE/mri.xlsx'
        
        if not os.path.exists(images_dir):
            print(f"❌ 图像目录不存在: {images_dir}")
            return False
        
        if not os.path.exists(pose_file):
            print(f"❌ 姿态文件不存在: {pose_file}")
            return False
        
        # 尝试加载数据集
        train_set = RGBPose3DDataset(images_dir, pose_file, copy_to_gpu=False, grayscale=True, mode='train')
        print(f"✅ 数据集加载成功，样本数量: {len(train_set)}")
        
        # 测试获取一个样本
        sample = train_set[0]
        print(f"样本键: {list(sample.keys())}")
        print(f"坐标形状: {sample['coords'].shape}")
        print(f"图像形状: {sample['gt_img'].shape}")
        print(f"DINO形状: {sample['dino'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("单专家训练代码测试")
    print("=" * 60)
    
    tests = [
        test_single_expert_model,
        test_loss_function,
        test_dataset_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("🎉 所有测试通过! 单专家训练代码可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查相关代码。")


if __name__ == '__main__':
    main()




