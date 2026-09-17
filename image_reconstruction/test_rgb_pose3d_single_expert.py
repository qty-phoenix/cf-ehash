#!/usr/bin/env python3
"""
单专家模型的RGB Pose3D测试脚本
基于训练脚本，用于在测试集上评估已训练的模型
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import time
from skimage.metrics import structural_similarity as ssim

from models import build_model
from datasets.RGBPose3D import RGBPose3DDataset
from datasets.RGBPose3D_GlobalNorm import RGBPose3DDatasetGlobalNorm


def convert_to_uint8(img):
    """将张量转换为uint8格式用于保存图像"""
    img = np.clip(img, 0.0, 1.0)
    
    if img.shape[-1] == 1:
        img = img.squeeze()
        img = (img * 255).astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    return img


def save_test_results(model, dataloader, device, output_dir, save_images=True):
    """在测试集上评估模型并保存结果"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    all_mse = []
    all_psnr = []
    all_mae = []
    all_ssim = []
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            coords = data['coords'].to(device)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)
            dino = data['dino'].to(device)
            
            # 前向传播
            output_pred = model(coords, dino=dino, img=gt_img)
            
            # 获取预测结果
            nonmanifold_pnts_pred = output_pred.get('nonmanifold_pnts_pred', None)
            if nonmanifold_pnts_pred is None:
                raise ValueError("无法获取模型预测结果")
            
            # 转换形状: (B, C, H*W) -> (B, H*W, C)
            if nonmanifold_pnts_pred.dim() == 3:
                nonmanifold_pnts_pred = nonmanifold_pnts_pred.permute(0, 2, 1)
            
            # 限制预测值到[0,1]范围
            nonmanifold_pnts_pred = torch.clamp(nonmanifold_pnts_pred, 0.0, 1.0)
            
            # 重塑为图像格式
            B, C, H, W = gt_img.shape
            pred_img = nonmanifold_pnts_pred.reshape(B, H, W, C)
            gt_img_flat = gt_img.permute(0, 2, 3, 1)  # (B, H, W, C)
            
            # 计算指标
            mse = torch.mean((pred_img - gt_img_flat) ** 2)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
            mae = torch.mean(torch.abs(pred_img - gt_img_flat))
            
            # 计算SSIM
            pred_np = pred_img.detach().cpu().numpy()
            gt_np = gt_img_flat.detach().cpu().numpy()
            
            ssim_values = []
            for b in range(B):
                if C == 1:  # 灰度图
                    pred_slice = pred_np[b, :, :, 0]
                    gt_slice = gt_np[b, :, :, 0]
                else:  # RGB图
                    pred_slice = pred_np[b]
                    gt_slice = gt_np[b]
                
                ssim_val = ssim(gt_slice, pred_slice, 
                               data_range=1.0,
                               channel_axis=2 if C > 1 else None)
                ssim_values.append(ssim_val)
            
            avg_ssim_batch = np.mean(ssim_values)
            
            all_mse.append(mse.item())
            all_psnr.append(psnr.item())
            all_mae.append(mae.item())
            all_ssim.append(avg_ssim_batch)
            
            # 保存图像
            if save_images:
                pred_np_viz = pred_img.squeeze(0).detach().cpu().numpy()
                gt_np_viz = gt_img_flat.squeeze(0).detach().cpu().numpy()
                
                pred_u8 = convert_to_uint8(pred_np_viz)
                gt_u8 = convert_to_uint8(gt_np_viz)
                
                # 拼接图像
                if pred_u8.shape[-1] == 1:
                    pred_u8 = pred_u8.squeeze()
                    gt_u8 = gt_u8.squeeze()
                    concat_img = np.hstack([gt_u8, pred_u8])
                else:
                    concat_img = np.hstack([gt_u8, pred_u8])
                
                img = Image.fromarray(concat_img)
                img.save(os.path.join(output_dir, f'test_{i:04d}.png'))
            
            print(f"样本 {i+1}: MSE={mse.item():.6f}, PSNR={psnr.item():.2f}dB, "
                  f"SSIM={avg_ssim_batch:.4f}, MAE={mae.item():.6f}")
    
    # 计算总体统计
    avg_mse = np.mean(all_mse)
    avg_psnr = np.mean(all_psnr)
    avg_mae = np.mean(all_mae)
    avg_ssim = np.mean(all_ssim)
    
    std_mse = np.std(all_mse)
    std_psnr = np.std(all_psnr)
    std_mae = np.std(all_mae)
    std_ssim = np.std(all_ssim)
    
    print("\n" + "=" * 80)
    print("测试集评估结果:")
    print(f"MSE:   {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"SSIM:  {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"MAE:   {avg_mae:.6f} ± {std_mae:.6f}")
    print("=" * 80)
    
    # 保存结果到文件
    results_file = os.path.join(output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("测试集评估结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"MSE:   {avg_mse:.6f} ± {std_mse:.6f}\n")
        f.write(f"PSNR:  {avg_psnr:.2f} ± {std_psnr:.2f} dB\n")
        f.write(f"SSIM:  {avg_ssim:.4f} ± {std_ssim:.4f}\n")
        f.write(f"MAE:   {avg_mae:.6f} ± {std_mae:.6f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n结果已保存到: {results_file}")
    if save_images:
        print(f"测试图像已保存到: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGBPose3D with Single Expert Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save test results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save_images', action='store_true',
                        help='Save test images')
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("\n加载测试数据集...")
    use_global_norm = cfg['DATA'].get('use_global_norm', False)
    
    if use_global_norm:
        dataset = RGBPose3DDatasetGlobalNorm(
            images_dir=cfg['DATA']['dataset_path'],
            pose_file=cfg['DATA']['pose_file'],
            copy_to_gpu=cfg['DATA'].get('copy_to_gpu', False),
            grayscale=cfg['MODEL']['out_dim'] == 1,
            crop_size=cfg['DATA'].get('crop_size', None),
            angles_in_degrees=cfg['DATA'].get('angles_in_degrees', True),
            mode='test'  # 使用测试集模式
        )
    else:
        dataset = RGBPose3DDataset(
            images_dir=cfg['DATA']['dataset_path'],
            pose_file=cfg['DATA']['pose_file'],
            copy_to_gpu=cfg['DATA'].get('copy_to_gpu', False),
            grayscale=cfg['MODEL']['out_dim'] == 1,
            crop_size=cfg['DATA'].get('crop_size', None),
            angles_in_degrees=cfg['DATA'].get('angles_in_degrees', True),
            mode='test'  # 使用测试集模式
        )
    
    test_loader = DataLoader(
        dataset,
        batch_size=cfg['TESTING']['batch_size'],
        shuffle=False,
        num_workers=cfg['TESTING']['num_workers']
    )
    
    print(f"测试样本数: {len(dataset)}")
    
    # 构建模型
    print("\n构建模型...")
    SINR, _ = build_model(cfg, cfg['LOSS'])
    SINR.to(device)
    
    # 加载训练好的模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    SINR.load_state_dict(checkpoint)
    print("✓ 模型加载成功")
    
    # 测试
    print("\n开始测试...")
    save_test_results(SINR, test_loader, device, args.output_dir, save_images=args.save_images)
    
    print("\n测试完成!")


if __name__ == '__main__':
    main()

