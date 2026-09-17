#!/usr/bin/env python3
"""
修复Manager网络的训练脚本
专门解决Manager网络不起作用的问题
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

from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets.RGBPose3D import RGBPose3DDataset


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


def compute_manager_aware_loss(output_pred, gt_img_flat, q, balance_weight=10.0, recon_weight=100.0, diversity_weight=1.0):
    """
    计算Manager感知损失：重建损失 + Manager平衡损失 + 专家多样性损失
    
    Args:
        output_pred: 模型输出
        gt_img_flat: 真实图像 (B, H*W, C)
        q: 专家权重 (B, n_experts, H*W)
        balance_weight: 平衡损失权重
        recon_weight: 重建损失权重
        diversity_weight: 多样性损失权重
    """
    # 获取预测结果
    pred = output_pred.get('selected_nonmanifold_pnts_pred', output_pred['nonmanifold_pnts_pred'])
    if pred.dim() == 4:  # (1, 1, H*W, 1)
        pred = pred.squeeze(0).squeeze(-1)  # (1, H*W)
    elif pred.dim() == 3:  # (1, H*W, 1)
        pred = pred.squeeze(-1)  # (1, H*W)
    
    # 限制预测值到[0,1]范围，防止异常值
    pred = torch.clamp(pred, 0.0, 1.0)
    
    # 1. 重建损失 (MSE)
    recon_loss = torch.mean((pred - gt_img_flat) ** 2) * recon_weight
    
    # 2. Manager平衡损失
    # q的形状是 (B, n_experts, H*W)，需要转置为 (B, H*W, n_experts)
    if q.dim() == 3 and q.shape[1] < q.shape[2]:  # (B, n_experts, H*W)
        q = q.transpose(1, 2)  # 转为 (B, H*W, n_experts)
    
    # 计算每个专家的使用频率
    expert_usage = q.mean(dim=1)  # (B, n_experts)
    
    # 计算专家使用频率的方差（鼓励平衡使用）
    usage_variance = torch.var(expert_usage, dim=1).mean()
    
    # 平衡损失：最小化使用频率的方差
    balance_loss = usage_variance
    
    # 3. 专家多样性损失（鼓励专家输出不同）
    # 获取所有专家的预测
    all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)  # (1, k, n_nm, 1)
    if all_expert_preds is not None and all_expert_preds.dim() == 4:
        all_expert_preds = all_expert_preds.squeeze(-1)  # (1, k, n_nm)
        
        # 计算专家间的差异
        expert_diffs = []
        for i in range(all_expert_preds.shape[1]):
            for j in range(i+1, all_expert_preds.shape[1]):
                diff = torch.mean((all_expert_preds[0, i] - all_expert_preds[0, j]) ** 2)
                expert_diffs.append(diff)
        
        if expert_diffs:
            diversity_loss = -torch.mean(torch.stack(expert_diffs))  # 负号表示最大化差异
        else:
            diversity_loss = torch.tensor(0.0, device=pred.device)
    else:
        diversity_loss = torch.tensor(0.0, device=pred.device)
    
    # 4. 总损失
    total_loss = recon_loss + balance_weight * balance_loss + diversity_weight * diversity_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'balance_loss': balance_loss,
        'diversity_loss': diversity_loss,
        'expert_usage': expert_usage.mean(dim=0)  # 平均使用频率
    }


def save_prediction_samples_with_expert_info(model, dataloader, device, output_dir, epoch, num_samples=3):
    """保存预测样本，包含专家选择信息"""
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
            
            # 获取预测结果
            if 'selected_nonmanifold_pnts_pred' in output_pred:
                pred = output_pred['selected_nonmanifold_pnts_pred']
            else:
                pred = output_pred['nonmanifold_pnts_pred']
                if pred.dim() == 3 and pred.shape[0] != gt_img.shape[0]:
                    pred = pred[0]
            
            # 获取专家选择信息
            q = output_pred.get('nonmnfld_q', None)
            expert_selection = None
            if q is not None:
                expert_selection = torch.argmax(q, dim=1)  # (1, H*W)
            
            # 重塑为图像格式
            B, C, H, W = gt_img.shape
            if pred.dim() == 4:  # (1, 1, H*W, 1)
                pred = pred.squeeze(0).squeeze(-1).reshape(H, W).unsqueeze(0).unsqueeze(0)
            elif pred.dim() == 3:  # (1, H*W, 1)
                pred = pred.squeeze(-1).reshape(H, W).unsqueeze(0).unsqueeze(0)
            elif pred.dim() == 2:  # (H*W, 1)
                pred = pred.squeeze(-1).reshape(H, W).unsqueeze(0).unsqueeze(0)
            
            # 转换为numpy数组
            pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
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
            img_pil.save(os.path.join(output_dir, f'epoch_{epoch}_sample_{i}.png'))
            
            # 保存专家选择可视化
            if expert_selection is not None:
                expert_map = expert_selection.squeeze().reshape(H, W).cpu().numpy()
                # 将专家索引映射到灰度值
                expert_map_normalized = (expert_map / expert_map.max() * 255).astype(np.uint8)
                expert_img = Image.fromarray(expert_map_normalized, mode='L')
                expert_img.save(os.path.join(output_dir, f'epoch_{epoch}_expert_selection_{i}.png'))


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
    
    train_set = RGBPose3DDataset(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                                 crop_size=cfg['DATA'].get('crop_size', None),
                                 angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                                 mode='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    
    print(f"训练集大小: {len(train_set)}")

    # 解析损失权重
    loss_type = cfg['LOSS']['loss_type']
    recon_weight_match = re.match(r'(\d+(?:\.\d+)?)rgbrecon', loss_type)
    recon_weight = float(recon_weight_match.group(1)) if recon_weight_match else 100.0
    
    balance_weight_match = re.search(r'\+(\d+(?:\.\d+)?)balance', loss_type)
    balance_weight = float(balance_weight_match.group(1)) if balance_weight_match else 10.0
    
    # 限制重建损失权重，避免梯度爆炸
    recon_weight = min(recon_weight, 100.0)
    
    print(f"损失类型: {loss_type}")
    print(f"重建损失权重: {recon_weight}")
    print(f"平衡损失权重: {balance_weight}")

    # 模型和损失函数
    cfg['MODEL']['out_dim'] = cfg['MODEL']['out_dim']
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")
    
    # 分别设置Manager和Expert的学习率
    manager_params = list(SINR.manager_net.parameters())
    expert_params = list(SINR.decoder.parameters())
    
    manager_lr = cfg['TRAINING']['lr'] * 2.0  # Manager使用更高的学习率
    expert_lr = cfg['TRAINING']['lr']
    
    optimizer = optim.Adam([
        {'params': manager_params, 'lr': manager_lr, 'name': 'manager'},
        {'params': expert_params, 'lr': expert_lr, 'name': 'experts'}
    ], betas=(0.9, 0.999))
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['TRAINING']['lr_gamma'])

    SINR.to(device)
    
    # 训练统计
    total_epochs = cfg['TRAINING']['num_epochs']
    save_interval = 100
    vis_interval = 200
    
    print(f"\n开始训练...")
    print(f"总epoch数: {total_epochs}")
    print(f"Manager学习率: {manager_lr}")
    print(f"Expert学习率: {expert_lr}")
    print(f"模型保存间隔: {save_interval}")
    print(f"可视化间隔: {vis_interval}")
    print("=" * 80)

    start_time = time.time()
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        SINR.train()
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_balance_losses = []
        epoch_diversity_losses = []
        epoch_metrics = {'mse': [], 'psnr': [], 'mae': []}
        epoch_expert_usage = []
        
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
            
            # 获取专家权重q
            q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
            if q is None:
                n_experts = cfg['MODEL']['n_experts']
                q = torch.ones(B, H * W, n_experts, device=device) / n_experts
            
            # 调试信息（仅第一个epoch的前几个batch）
            if epoch == 0 and batch_idx < 3:
                print(f"  Batch {batch_idx}: q形状={q.shape}, q范围=[{q.min().item():.6f}, {q.max().item():.6f}]")
                expert_selection = torch.argmax(q, dim=1) if q.dim() == 3 else None
                if expert_selection is not None:
                    unique_experts = torch.unique(expert_selection)
                    print(f"    选择的专家: {unique_experts.cpu().numpy()}")
            
            # 使用Manager感知损失函数
            loss_dict = compute_manager_aware_loss(output_pred, gt_img_flat, q, 
                                                 balance_weight=balance_weight, 
                                                 recon_weight=recon_weight)

            optimizer.zero_grad(set_to_none=True)
            loss_dict['loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(SINR.parameters(), cfg['TRAINING']['grad_clip_norm'])
            
            optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss_dict['loss'].item())
            epoch_recon_losses.append(loss_dict['recon_loss'].item())
            epoch_balance_losses.append(loss_dict['balance_loss'].item())
            epoch_diversity_losses.append(loss_dict['diversity_loss'].item())
            epoch_expert_usage.append(loss_dict['expert_usage'].detach().cpu().numpy())
            
            # 计算指标
            if batch_idx % 10 == 0:
                pred = output_pred.get('selected_nonmanifold_pnts_pred', 
                                     output_pred['nonmanifold_pnts_pred'])
                if pred.dim() == 4:
                    pred = pred.squeeze(0).squeeze(-1)
                elif pred.dim() == 3:
                    pred = pred.squeeze(-1)
                
                mse = torch.mean((pred - gt_img_flat) ** 2)
                psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
                mae = torch.mean(torch.abs(pred - gt_img_flat))
                
                epoch_metrics['mse'].append(mse.item())
                epoch_metrics['psnr'].append(psnr.item())
                epoch_metrics['mae'].append(mae.item())

        # 计算epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_balance_loss = np.mean(epoch_balance_losses)
        avg_diversity_loss = np.mean(epoch_diversity_losses)
        avg_mse = np.mean(epoch_metrics['mse']) if epoch_metrics['mse'] else 0
        avg_psnr = np.mean(epoch_metrics['psnr']) if epoch_metrics['psnr'] else 0
        avg_mae = np.mean(epoch_metrics['mae']) if epoch_metrics['mae'] else 0
        
        # 计算专家使用统计
        expert_usage_avg = np.mean(epoch_expert_usage, axis=0) if epoch_expert_usage else np.zeros(cfg['MODEL']['n_experts'])
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        
        # 打印进度信息
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            progress = (epoch + 1) / total_epochs * 100
            eta = elapsed_time / (epoch + 1) * (total_epochs - epoch - 1)
            
            print(f"Epoch {epoch+1:4d}/{total_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"Recon: {avg_recon_loss:.6f} | "
                  f"Balance: {avg_balance_loss:.6f} | "
                  f"Diversity: {avg_diversity_loss:.6f} | "
                  f"MSE: {avg_mse:.6f} | "
                  f"PSNR: {avg_psnr:.2f}dB | "
                  f"MAE: {avg_mae:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Progress: {progress:.1f}%")
            
            # 打印专家使用情况
            expert_usage_str = " | ".join([f"E{i}: {usage:.3f}" for i, usage in enumerate(expert_usage_avg)])
            print(f"  专家使用: {expert_usage_str}")

        # 保存模型
        if epoch % save_interval == 0 or epoch == total_epochs - 1:
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_manager_fixed_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, MAE: {avg_mae:.6f}")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_prediction_samples_with_expert_info(SINR, train_loader, device, vis_outdir, epoch, num_samples=3)
                print(f"  可视化保存到: {vis_outdir}")

        scheduler.step()
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"训练完成!")
    print(f"总训练时间: {total_time/60:.1f}分钟")
    print(f"最终模型保存到: {model_outdir}")
    print(f"训练可视化保存到: {vis_outdir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training RGBPose3D with Fixed Manager Network')
    parser.add_argument('--config', type=str, default='configs/config_RGB_pose3d_manager_fixed.yaml',
                        help='Path to config file')
    parser.add_argument('--logdir', type=str, default='./log/pose3d_manager_fixed',
                        help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


if __name__ == '__main__':
    main()





