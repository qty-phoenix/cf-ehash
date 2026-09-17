#!/usr/bin/env python3
"""
单专家模型的RGB Pose3D训练脚本
基于train_rgb_pose3d_gradient_fixed.py，但使用单专家模型（INR）而不是多专家模型（INR_MoE）
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import time
from datetime import datetime
import re
from skimage.metrics import structural_similarity as ssim

# 尝试导入LPIPS，如果未安装则设为None
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: lpips库未安装，LPIPS指标将不可用。请运行: pip install lpips")

from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets.RGBPose3D_Cached import RGBPose3DDataset
from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm


def convert_to_uint8(img):
    """将张量转换为uint8格式用于保存图像"""
    # 确保像素值在[0,1]范围内
    img = np.clip(img, 0.0, 1.0)
    
    if img.shape[-1] == 1:
        img = img.squeeze()
        img = (img * 255).astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    return img


def compute_single_expert_loss(output_pred, gt_img_flat, recon_weight=1000.0):
    """
    计算单专家模型的损失函数：只有重建损失，没有专家平衡损失
    
    Args:
        output_pred: 模型输出
        gt_img_flat: 真实图像 (B, H*W, C)，其中C=1表示灰度图
        recon_weight: 重建损失权重
    """
    # 获取单专家的预测结果
    # INR模型返回: (B, C, H*W)，其中C=1表示灰度图，H*W是像素数量
    nonmanifold_pnts_pred = output_pred.get('nonmanifold_pnts_pred', None)
    if nonmanifold_pnts_pred is None:
        raise ValueError("无法找到nonmanifold_pnts_pred")
    
    # 转换形状: (B, C, H*W) -> (B, H*W, C)
    if nonmanifold_pnts_pred.dim() == 3:
        nonmanifold_pnts_pred = nonmanifold_pnts_pred.permute(0, 2, 1)  # (B, H*W, C)
    else:
        raise ValueError(f"意外的nonmanifold_pnts_pred形状: {nonmanifold_pnts_pred.shape}，期望3维")
    
    # 限制预测值到[0,1]范围
    nonmanifold_pnts_pred = torch.clamp(nonmanifold_pnts_pred, 0.0, 1.0)
    
    # 确保gt_img_flat和pred形状一致: (B, H*W, C)
    assert gt_img_flat.shape == nonmanifold_pnts_pred.shape, \
        f"形状不匹配: gt_img_flat {gt_img_flat.shape} vs pred {nonmanifold_pnts_pred.shape}"
    
    # 重建损失 (MSE)
    recon_loss = torch.mean((nonmanifold_pnts_pred - gt_img_flat) ** 2) * recon_weight
    
    # 单专家模型没有平衡损失
    total_loss = recon_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'balance_loss': torch.tensor(0.0, device=recon_loss.device),  # 无平衡损失
        'expert_usage': torch.tensor([1.0], device=recon_loss.device),  # 单专家使用率100%
        'weighted_pred': nonmanifold_pnts_pred  # 返回预测用于可视化
    }


def evaluate_test_set_single_expert(model, dataloader, device, cfg):
    """
    在测试集上评估单专家模型
    返回测试集的平均指标
    """
    model.eval()
    
    all_mse = []
    all_psnr = []
    all_mae = []
    all_ssim = []
    all_lpips = []
    
    # 初始化LPIPS模型（如果可用）
    lpips_model = None
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
    
    with torch.no_grad():
        for data in dataloader:
            coords = data['coords'].to(device)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)
            dino = data['dino'].to(device)
            
            # 前向传播
            output_pred = model(coords, dino=dino, img=gt_img)
            B, C, H, W = gt_img.shape
            gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # 获取单专家预测
            nonmanifold_pnts_pred = output_pred.get('nonmanifold_pnts_pred', None)
            if nonmanifold_pnts_pred is None:
                raise ValueError("无法获取模型预测结果")
            
            # 转换形状: (B, C, H*W) -> (B, H*W, C)
            if nonmanifold_pnts_pred.dim() == 3:
                pred_flat = nonmanifold_pnts_pred.permute(0, 2, 1)  # (B, H*W, C)
            else:
                raise ValueError(f"意外的nonmanifold_pnts_pred形状: {nonmanifold_pnts_pred.shape}")
            
            # 限制预测值到[0,1]范围
            pred_flat = torch.clamp(pred_flat, 0.0, 1.0)
            
            # 计算指标
            mse = torch.mean((pred_flat - gt_img_flat) ** 2)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
            mae = torch.mean(torch.abs(pred_flat - gt_img_flat))
            
            # 计算SSIM
            pred_2d = pred_flat.reshape(B, H, W, C).detach().cpu().numpy()
            gt_2d = gt_img_flat.reshape(B, H, W, C).detach().cpu().numpy()
            
            ssim_values = []
            for i in range(B):
                if C == 1:  # 灰度图
                    pred_slice = pred_2d[i, :, :, 0]
                    gt_slice = gt_2d[i, :, :, 0]
                else:  # RGB图
                    pred_slice = pred_2d[i, :, :, 0]  # 使用第一个通道
                    gt_slice = gt_2d[i, :, :, 0]
                
                ssim_val = ssim(gt_slice, pred_slice, 
                               data_range=1.0,
                               channel_axis=None)
                ssim_values.append(ssim_val)
            
            avg_ssim_batch = np.mean(ssim_values)
            
            # 计算LPIPS (如果可用)
            if lpips_model is not None:
                # 将预测图像重塑为 (B, C, H, W) 格式
                pred_img = pred_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
                
                if C == 1:
                    pred_img = pred_img.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
                    gt_img_rgb = gt_img.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
                else:
                    gt_img_rgb = gt_img  # 已经是(B, C, H, W)格式
                
                pred_img = torch.clamp(pred_img, 0.0, 1.0)
                gt_img_rgb = torch.clamp(gt_img_rgb, 0.0, 1.0)
                
                pred_img_lpips = pred_img * 2.0 - 1.0
                gt_img_lpips = gt_img_rgb * 2.0 - 1.0
                
                lpips_val = lpips_model(pred_img_lpips, gt_img_lpips)
                lpips_val_mean = lpips_val.mean().item()
                all_lpips.append(lpips_val_mean)
            
            all_mse.append(mse.item())
            all_psnr.append(psnr.item())
            all_mae.append(mae.item())
            all_ssim.append(avg_ssim_batch)
    
    # 计算总体统计
    avg_mse = np.mean(all_mse)
    avg_psnr = np.mean(all_psnr)
    avg_mae = np.mean(all_mae)
    avg_ssim = np.mean(all_ssim)
    
    result = {
        'mse': avg_mse,
        'psnr': avg_psnr,
        'mae': avg_mae,
        'ssim': avg_ssim
    }
    
    if all_lpips:
        result['lpips'] = np.mean(all_lpips)
    else:
        result['lpips'] = None
    
    return result


def save_prediction_samples_single_expert(model, dataloader, device, output_dir, epoch, num_samples=3, prefix='train'):
    """保存单专家模型的预测样本"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_samples:
                break
                
            coords = data['coords'].to(device)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)
            dino = data['dino'].to(device)
            
            # 前向传播
            output_pred = model(coords, dino=dino, img=gt_img)
            
            # 获取单专家预测
            # INR输出: (B, C, H*W)
            nonmanifold_pnts_pred = output_pred.get('nonmanifold_pnts_pred', None)
            if nonmanifold_pnts_pred is None:
                raise ValueError("无法获取模型预测结果")
            
            # 重塑为图像格式
            B, C, H, W = gt_img.shape
            # nonmanifold_pnts_pred: (B, C, H*W) -> (B, C, H, W)
            pred_img = nonmanifold_pnts_pred.reshape(B, C, H, W)
            
            # 转换为numpy数组
            pred_np = pred_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            gt_np = gt_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            
            # 转换为uint8格式
            pred_u8 = convert_to_uint8(pred_np)
            gt_u8 = convert_to_uint8(gt_np)
            
            # 拼接图像
            if pred_u8.shape[-1] == 1:
                pred_u8 = pred_u8.squeeze(-1)
                gt_u8 = gt_u8.squeeze(-1)
                combined = np.concatenate([gt_u8, pred_u8], axis=1)
            else:
                combined = np.concatenate([gt_u8, pred_u8], axis=1)
            
            # 保存图像
            img_pil = Image.fromarray(combined)
            img_pil.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_sample_{i}_single_expert.png'))


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    # 创建输出目录
    logdir = args.logdir
    model_outdir = os.path.join(logdir, 'models')
    vis_outdir = os.path.join(logdir, 'visualizations')
    os.makedirs(model_outdir, exist_ok=True)
    os.makedirs(vis_outdir, exist_ok=True)
    
    # 数据集设置
    images_dir = cfg['DATA']['dataset_path']
    pose_file = cfg['DATA']['pose_file']
    is_grayscale = cfg['MODEL']['out_dim'] == 1
    
    print(f"图像目录: {images_dir}")
    print(f"姿态文件: {pose_file}")
    print(f"灰度模式: {is_grayscale}")
    
    # 使用全局归一化版本的数据集（关键修复！）
    use_global_norm = cfg['DATA'].get('use_global_norm', True)
    print(f"使用全局归一化: {use_global_norm}")
    
    # 检查是否使用uvfdata2标定矩阵
    use_uvfdata2_calibration = cfg['DATA'].get('use_uvfdata2_calibration', False)
    print("=" * 80)
    if use_uvfdata2_calibration:
        print("✅ 启用uvfdata2标定矩阵变换:")
        print("   - S矩阵 (scaling_from_pixel_to_mm): 像素单位 → 毫米单位")
        print("   - C矩阵 (spatial_calibration): 图像坐标系 → 工具坐标系")
    else:
        print("❌ 未启用uvfdata2标定矩阵")
    print("=" * 80)
    
    DatasetClass = RGBPose3DDatasetGlobalNorm if use_global_norm else RGBPose3DDataset
    train_set = DatasetClass(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                            crop_size=cfg['DATA'].get('crop_size', None),
                            angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                            mode='train',
                            use_uvfdata2_calibration=use_uvfdata2_calibration)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    
    # 创建测试集
    test_set = DatasetClass(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                           crop_size=cfg['DATA'].get('crop_size', None),
                           angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                           mode='test',
                           use_uvfdata2_calibration=use_uvfdata2_calibration)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")

    # 解析损失权重
    loss_type = cfg['LOSS']['loss_type']
    recon_weight_match = re.match(r'(\d+(?:\.\d+)?)rgbrecon', loss_type)
    recon_weight = float(recon_weight_match.group(1)) if recon_weight_match else 1000.0
    
    print(f"损失类型: {loss_type}")
    print(f"重建损失权重: {recon_weight}")

    # 模型和损失函数
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")
    
    # 检查是否有哈希编码，如果有则给予更高的学习率
    base_lr = cfg['TRAINING']['lr']
    
    # 分离哈希编码参数和其他参数
    hash_encoder_params = []
    other_params = []
    
    if hasattr(SINR, 'decoder_input_encoding_module'):
        if hasattr(SINR.decoder_input_encoding_module, 'hash_encoder') and \
           SINR.decoder_input_encoding_module.hash_encoder is not None:
            hash_encoder_params = list(SINR.decoder_input_encoding_module.hash_encoder.parameters())
            print(f"检测到哈希编码，参数数量: {sum(p.numel() for p in hash_encoder_params):,}")
    
    # 收集其他所有参数（排除哈希编码参数）
    hash_param_ids = {id(p) for p in hash_encoder_params}
    for p in SINR.parameters():
        if id(p) not in hash_param_ids:
            other_params.append(p)
    
    # 构建优化器
    if hash_encoder_params:
        # 哈希编码需要更高的学习率（embedding需要从头学习）
        # 但要避免过高导致训练崩溃
        hash_lr = base_lr * 10.0  # 使用10倍学习率，但base_lr很低(1e-4)所以实际是1e-3
        optimizer = optim.Adam([
            {'params': other_params, 'lr': base_lr},
            {'params': hash_encoder_params, 'lr': hash_lr, 'name': 'hash_encoding'}
        ], betas=(0.9, 0.999))
        print(f"使用差异化学习率: 基础={base_lr:.2e}, 哈希编码={hash_lr:.2e}")
    else:
        # 没有哈希编码，使用统一学习率
        optimizer = optim.Adam(SINR.parameters(), lr=base_lr, betas=(0.9, 0.999))
        print(f"使用统一学习率: {base_lr:.2e}")
    
    # 根据配置选择学习率调度器
    lr_scheduler_type = cfg['TRAINING'].get('lr_scheduler', 'ExponentialLR')
    if lr_scheduler_type == 'StepLR':
        step_size = cfg['TRAINING'].get('lr_step_size', 10)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg['TRAINING']['lr_gamma'])
        print(f"使用StepLR调度器: 每{step_size}个epoch学习率乘以{cfg['TRAINING']['lr_gamma']}")
    else:  # ExponentialLR
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['TRAINING']['lr_gamma'])
        print(f"使用ExponentialLR调度器: 每个epoch学习率乘以{cfg['TRAINING']['lr_gamma']}")

    SINR.to(device)
    
    # 训练统计
    total_epochs = cfg['TRAINING']['num_epochs']
    save_interval = 100
    vis_interval = 200
    test_interval = 1  # 每个epoch都测试（与其他代码统一）
    
    # 早停机制
    best_loss = float('inf')
    patience = 200  # 200个epoch没有改善就停止
    patience_counter = 0
    
    print(f"\n开始训练单专家模型...")
    print(f"总epoch数: {total_epochs}")
    print(f"学习率: {cfg['TRAINING']['lr']}")
    print(f"模型保存间隔: {save_interval}")
    print(f"可视化间隔: {vis_interval}")
    print(f"测试集评估间隔: {test_interval}")
    print(f"早停耐心值: {patience} epochs")
    print("=" * 80)

    start_time = time.time()
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        SINR.train()
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_metrics = {'mse': [], 'psnr': [], 'mae': [], 'ssim': []}
        
        for batch_idx, data in enumerate(train_loader):
            coords = data['coords'].to(device)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)
            segments = data['segments'].to(device)
            dino = data['dino'].to(device)

            coords.requires_grad_()

            output_pred = SINR(coords, dino=dino, img=gt_img)
            B, C, H, W = gt_img.shape
            gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            # 使用单专家损失函数
            loss_dict = compute_single_expert_loss(output_pred, gt_img_flat, 
                                                  recon_weight=recon_weight)

            optimizer.zero_grad(set_to_none=True)
            loss_dict['loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(SINR.parameters(), cfg['TRAINING']['grad_clip_norm'])
            
            optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss_dict['loss'].item())
            epoch_recon_losses.append(loss_dict['recon_loss'].item())
            
            # 计算指标
            if batch_idx % 10 == 0:
                pred_img = loss_dict['weighted_pred']  # (B, H*W, C)
                mse = torch.mean((pred_img - gt_img_flat) ** 2)
                psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
                mae = torch.mean(torch.abs(pred_img - gt_img_flat))
                
                # 计算SSIM
                pred_2d = pred_img.reshape(B, H, W, C).detach().cpu().numpy()
                gt_2d = gt_img_flat.reshape(B, H, W, C).detach().cpu().numpy()
                
                # 对batch中的每张图像计算SSIM并取平均
                ssim_values = []
                for i in range(B):
                    if C == 1:  # 灰度图
                        pred_slice = pred_2d[i, :, :, 0]
                        gt_slice = gt_2d[i, :, :, 0]
                    else:  # RGB图
                        pred_slice = pred_2d[i]
                        gt_slice = gt_2d[i]
                    
                    ssim_val = ssim(gt_slice, pred_slice, 
                                   data_range=1.0,
                                   channel_axis=2 if C > 1 else None)
                    ssim_values.append(ssim_val)
                
                avg_ssim_batch = np.mean(ssim_values)
                
                epoch_metrics['mse'].append(mse.item())
                epoch_metrics['psnr'].append(psnr.item())
                epoch_metrics['mae'].append(mae.item())
                epoch_metrics['ssim'].append(avg_ssim_batch)

        # 计算epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_mse = np.mean(epoch_metrics['mse']) if epoch_metrics['mse'] else 0
        avg_psnr = np.mean(epoch_metrics['psnr']) if epoch_metrics['psnr'] else 0
        avg_mae = np.mean(epoch_metrics['mae']) if epoch_metrics['mae'] else 0
        avg_ssim = np.mean(epoch_metrics['ssim']) if epoch_metrics['ssim'] else 0
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        
        # 打印进度信息
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            progress = (epoch + 1) / total_epochs * 100
            eta = elapsed_time / (epoch + 1) * (total_epochs - epoch - 1)
            
            print(f"Epoch {epoch+1:4d}/{total_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Recon: {avg_recon_loss:.6f} | "
                  f"MSE: {avg_mse:.6f} | "
                  f"PSNR: {avg_psnr:.2f}dB | "
                  f"SSIM: {avg_ssim:.4f} | "
                  f"MAE: {avg_mae:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Progress: {progress:.1f}%")

        # 保存模型
        if epoch % save_interval == 0 or epoch == total_epochs - 1:
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_single_expert_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.6f}")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_prediction_samples_single_expert(SINR, train_loader, device, vis_outdir, epoch, num_samples=3, prefix='train')
                print(f"  可视化保存到: {vis_outdir}")

        scheduler.step()
        
        # 测试集评估（每个epoch）
        if (epoch + 1) % test_interval == 0 or epoch == total_epochs - 1:
            print(f"\n{'='*80}")
            print(f"测试集评估 (Epoch {epoch+1})...")
            test_metrics = evaluate_test_set_single_expert(SINR, test_loader, device, cfg)
            print(f"测试集指标:")
            print(f"  MSE:   {test_metrics['mse']:.6f}")
            print(f"  PSNR:  {test_metrics['psnr']:.2f} dB")
            print(f"  SSIM:  {test_metrics['ssim']:.4f}")
            print(f"  MAE:   {test_metrics['mae']:.6f}")
            if test_metrics['lpips'] is not None:
                print(f"  LPIPS: {test_metrics['lpips']:.6f}")
            else:
                print(f"  LPIPS: N/A (lpips库未安装)")
            print(f"{'='*80}\n")
        
        # 🔥 每个epoch都保存测试集可视化（确保每个epoch都有图像结果）
        print(f"  生成测试集可视化 (Epoch {epoch+1})...")
        save_prediction_samples_single_expert(SINR, test_loader, device, vis_outdir, epoch, num_samples=10, prefix='test')
        print(f"  测试集可视化保存到: {vis_outdir}")
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f"🎯 新的最佳损失: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 早停触发! 连续{patience}个epoch没有改善")
                print(f"   最佳损失: {best_loss:.6f}")
                print(f"   当前损失: {avg_loss:.6f}")
                break
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"单专家模型训练完成!")
    print(f"总训练时间: {total_time/60:.1f}分钟")
    print(f"最终模型保存到: {model_outdir}")
    print(f"训练可视化保存到: {vis_outdir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training RGBPose3D with Single Expert Model')
    parser.add_argument('--config', type=str, default='configs/config_RGB_pose3d_single_expert.yaml',
                        help='Path to config file')
    parser.add_argument('--logdir', type=str, default='./log/pose3d_single_expert',
                        help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


if __name__ == '__main__':
    main()




