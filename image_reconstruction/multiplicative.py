#!/usr/bin/env python3
"""
使用MultiplicativeActivation的训练脚本
专家网络使用新的乘法激活函数，Manager网络仍使用sine激活函数
⚠️ 分阶段训练已禁用 - Manager和Expert网络同时训练
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
# from models.stage_handler import TrainingStageHandler  # 禁用分阶段训练
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


def compute_gradient_aware_loss(output_pred, gt_img_flat, q, balance_weight=10.0, recon_weight=100.0):
    """
    计算梯度感知损失：使用加权平均而不是argmax，确保梯度能够传播到Manager网络
    
    Args:
        output_pred: 模型输出
        gt_img_flat: 真实图像 (B, H*W, C)
        q: 专家权重 (B, n_experts, H*W)
        balance_weight: 平衡损失权重
        recon_weight: 重建损失权重
    """
    # 获取所有专家的预测结果
    all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)  # (1, k, n_nm, 1)
    if all_expert_preds is None:
        raise ValueError("无法找到nonmanifold_pnts_pred")
    
    # 确保形状正确
    if all_expert_preds.dim() == 4:  # (1, k, n_nm, 1)
        all_expert_preds = all_expert_preds.squeeze(-1)  # (1, k, n_nm)
    elif all_expert_preds.dim() == 3:  # (1, k, n_nm)
        pass
    else:
        raise ValueError(f"意外的all_expert_preds形状: {all_expert_preds.shape}")
    
    # 确保q的形状正确: (B, n_experts, H*W)
    if q.dim() == 3 and q.shape[1] == all_expert_preds.shape[1]:  # (B, n_experts, H*W)
        pass
    else:
        raise ValueError(f"q形状不匹配: {q.shape}, 期望: (B, {all_expert_preds.shape[1]}, H*W)")
    
    # 关键修复：使用加权平均而不是argmax选择
    # q: (B, n_experts, H*W), all_expert_preds: (B, n_experts, H*W)
    # 计算加权平均: sum(q * all_expert_preds, dim=1)
    weighted_pred = torch.sum(q * all_expert_preds, dim=1)  # (B, H*W)
    
    # 简化处理：直接使用sigmoid将输出映射到[0,1]范围
    # 对标SIREN的简洁处理方式
    weighted_pred = torch.sigmoid(weighted_pred)
    
    # 1. 重建损失 (MSE) - 现在使用加权平均的预测
    recon_loss = torch.mean((weighted_pred - gt_img_flat.squeeze(-1)) ** 2) * recon_weight
    
    # 2. Manager平衡损失
    # 计算每个专家的使用频率
    expert_usage = q.mean(dim=2)  # (B, n_experts)
    
    # 计算专家使用频率的方差（鼓励平衡使用）
    usage_variance = torch.var(expert_usage, dim=1).mean()
    
    # 平衡损失：最小化使用频率的方差
    balance_loss = usage_variance
    
    # 3. 总损失
    total_loss = recon_loss + balance_weight * balance_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'balance_loss': balance_loss,
        'expert_usage': expert_usage.mean(dim=0),  # 平均使用频率
        'weighted_pred': weighted_pred  # 返回加权预测用于可视化
    }


def save_prediction_samples_with_gradient_info(model, dataloader, device, output_dir, epoch, num_samples=3):
    """保存预测样本，包含梯度信息"""
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
            
            # 获取专家权重q
            q = output_pred.get('nonmnfld_q', None)
            if q is None:
                n_experts = 4
                q = torch.ones(1, n_experts, coords.shape[1], device=device) / n_experts
            
            # 计算加权预测
            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is not None:
                if all_expert_preds.dim() == 4:
                    all_expert_preds = all_expert_preds.squeeze(-1)
                weighted_pred = torch.sum(q * all_expert_preds, dim=1)
            else:
                weighted_pred = output_pred.get('selected_nonmanifold_pnts_pred', None)
                if weighted_pred.dim() == 3:
                    weighted_pred = weighted_pred.squeeze(-1)
            
            # 重塑为图像格式
            B, C, H, W = gt_img.shape
            if weighted_pred.dim() == 2:  # (B, H*W)
                weighted_pred = weighted_pred.reshape(B, H, W).unsqueeze(1)
            elif weighted_pred.dim() == 1:  # (H*W,)
                weighted_pred = weighted_pred.reshape(H, W).unsqueeze(0).unsqueeze(0)
            
            # 转换为numpy数组
            pred_np = weighted_pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
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
            img_pil.save(os.path.join(output_dir, f'epoch_{epoch}_sample_{i}_multiplicative.png'))
            
            # 保存专家权重可视化
            if q is not None:
                expert_selection = torch.argmax(q, dim=1)  # (1, H*W)
                expert_map = expert_selection.squeeze().reshape(H, W).cpu().numpy()
                
                # 改进颜色编码：为每个专家分配不同颜色
                n_experts = expert_map.max() + 1
                if n_experts == 2:
                    # 2个专家：黑色(0) 和 白色(255)
                    expert_map_normalized = (expert_map * 255).astype(np.uint8)
                elif n_experts == 4:
                    # 4个专家：黑色(0), 深灰(85), 浅灰(170), 白色(255)
                    expert_map_normalized = (expert_map * 85).astype(np.uint8)
                elif n_experts == 8:
                    # 8个专家：均匀分布到0-255
                    expert_map_normalized = (expert_map * 36).astype(np.uint8)  # 255/7 ≈ 36
                else:
                    # 通用：均匀分布到0-255
                    expert_map_normalized = (expert_map / expert_map.max() * 255).astype(np.uint8)
                
                expert_img = Image.fromarray(expert_map_normalized, mode='L')
                expert_img.save(os.path.join(output_dir, f'epoch_{epoch}_expert_selection_{i}_multiplicative.png'))
                
                # 创建彩色专家选择图
                if n_experts == 4:
                    # 4个专家：红(0), 绿(1), 蓝(2), 黄(3)
                    colors = np.array([
                        [255, 0, 0],    # 红色 - 专家0
                        [0, 255, 0],    # 绿色 - 专家1  
                        [0, 0, 255],    # 蓝色 - 专家2
                        [255, 255, 0]   # 黄色 - 专家3
                    ])
                    expert_map_color = colors[expert_map]
                    expert_img_color = Image.fromarray(expert_map_color.astype(np.uint8), mode='RGB')
                    expert_img_color.save(os.path.join(output_dir, f'epoch_{epoch}_expert_selection_color_{i}_multiplicative.png'))
                elif n_experts == 8:
                    # 8个专家：8种不同颜色
                    colors = np.array([
                        [255, 0, 0],    # 红色 - 专家0
                        [0, 255, 0],    # 绿色 - 专家1
                        [0, 0, 255],    # 蓝色 - 专家2
                        [255, 255, 0],  # 黄色 - 专家3
                        [255, 0, 255],  # 洋红 - 专家4
                        [0, 255, 255],  # 青色 - 专家5
                        [255, 128, 0],  # 橙色 - 专家6
                        [128, 0, 255]   # 紫色 - 专家7
                    ])
                    expert_map_color = colors[expert_map]
                    expert_img_color = Image.fromarray(expert_map_color.astype(np.uint8), mode='RGB')
                    expert_img_color.save(os.path.join(output_dir, f'epoch_{epoch}_expert_selection_color_{i}_multiplicative.png'))
                
                # 打印专家使用统计
                unique_experts, counts = np.unique(expert_map, return_counts=True)
                print(f"    专家选择统计: {dict(zip(unique_experts, counts))}")


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
    print(f"专家网络激活函数: {cfg['MODEL']['decoder_nl']}")
    print(f"Manager网络激活函数: {cfg['MODEL']['manager_nl']}")
    
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
    
    # 强制使用固定的损失权重
    recon_weight = 100.0
    balance_weight = 10.0
    
    print(f"损失类型: {loss_type}")
    print(f"重建损失权重: {recon_weight} (固定)")
    print(f"平衡损失权重: {balance_weight} (固定)")

    # 模型和损失函数
    cfg['MODEL']['out_dim'] = cfg['MODEL']['out_dim']
    
    # 传递MultiplicativeActivation的特殊参数
    if 'wave_alpha' in cfg['MODEL']:
        cfg['MODEL']['multiplicative_alpha'] = cfg['MODEL']['wave_alpha']
    if 'wave_beta' in cfg['MODEL']:
        cfg['MODEL']['multiplicative_beta'] = cfg['MODEL']['wave_beta']
    
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")
    
    # 分别设置Manager和Expert的学习率
    manager_params = list(SINR.manager_net.parameters())
    expert_params = list(SINR.decoder.parameters())
    
    manager_lr = cfg['TRAINING']['lr'] * 3.0  # Manager使用更高的学习率
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
    
    # 早停机制
    best_loss = float('inf')
    patience = 200  # 200个epoch没有改善就停止
    patience_counter = 0
    
    print(f"\n开始训练...")
    print(f"总epoch数: {total_epochs}")
    print(f"Manager学习率: {manager_lr}")
    print(f"Expert学习率: {expert_lr}")
    print(f"模型保存间隔: {save_interval}")
    print(f"可视化间隔: {vis_interval}")
    print(f"早停耐心值: {patience} epochs")
    print("⚠️  分阶段训练已禁用 - Manager和Expert网络同时训练")
    print("=" * 80)

    start_time = time.time()
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        SINR.train()
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_balance_losses = []
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
                q = torch.ones(B, n_experts, H * W, device=device) / n_experts
            
            # 调试信息（每100个epoch打印一次）
            if epoch % 100 == 0 and batch_idx < 3:
                print(f"  Epoch {epoch}, Batch {batch_idx}:")
                print(f"    q形状={q.shape}, q范围=[{q.min().item():.6f}, {q.max().item():.6f}]")
                print(f"    q均值={q.mean().item():.6f}, q标准差={q.std().item():.6f}")
                expert_selection = torch.argmax(q, dim=1) if q.dim() == 3 else None
                if expert_selection is not None:
                    unique_experts = torch.unique(expert_selection)
                    print(f"    选择的专家: {unique_experts.cpu().numpy()}")
                    
                # 检查预测值范围
                all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
                if all_expert_preds is not None:
                    if all_expert_preds.dim() == 4:
                        all_expert_preds = all_expert_preds.squeeze(-1)
                    print(f"    原始专家预测范围=[{all_expert_preds.min().item():.6f}, {all_expert_preds.max().item():.6f}]")
                    
                    # 计算加权预测并应用sigmoid
                    weighted_pred_raw = torch.sum(q * all_expert_preds, dim=1)
                    weighted_pred_sigmoid = torch.sigmoid(weighted_pred_raw)
                    print(f"    加权预测范围(原始)=[{weighted_pred_raw.min().item():.6f}, {weighted_pred_raw.max().item():.6f}]")
                    print(f"    加权预测范围(sigmoid)=[{weighted_pred_sigmoid.min().item():.6f}, {weighted_pred_sigmoid.max().item():.6f}]")
                    
                    # 检查MultiplicativeActivation参数和网络架构
                    print(f"    专家网络类型: {type(SINR.decoder).__name__}")
                    
                    # 检查网络层数（适配ParallelFullyConnectedNN）
                    if hasattr(SINR.decoder, 'layers'):
                        print(f"    专家网络层数: {len(SINR.decoder.layers)}")
                    elif hasattr(SINR.decoder, 'weights_list'):
                        print(f"    专家网络层数: {len(SINR.decoder.weights_list)}")
                    else:
                        print(f"    专家网络层数: N/A")
                    
                    # 检查最后一层是否应用了激活函数
                    if hasattr(SINR.decoder, 'outermost_linear'):
                        print(f"    outermost_linear: {SINR.decoder.outermost_linear}")
                    
                    # 检查MultiplicativeActivation参数（适配ParallelFullyConnectedNN）
                    if hasattr(SINR.decoder, 'nl') and hasattr(SINR.decoder.nl, 'alpha'):
                        alpha_val = SINR.decoder.nl.alpha
                        beta_val = SINR.decoder.nl.beta
                        print(f"    激活函数参数: alpha={alpha_val:.3f}, beta={beta_val:.3f}")
                    elif hasattr(SINR.decoder, 'layers') and len(SINR.decoder.layers) > 0:
                        for i, layer in enumerate(SINR.decoder.layers):
                            if hasattr(layer, 'activation') and hasattr(layer.activation, 'alpha'):
                                alpha_val = layer.activation.alpha
                                beta_val = layer.activation.beta
                                print(f"    专家{i} alpha={alpha_val:.3f}, beta={beta_val:.3f}")
                    
                    # 检查输入到专家网络的值
                    if hasattr(SINR.decoder, 'input_encoding_module'):
                        encoded_input = SINR.decoder.input_encoding_module(coords)
                        print(f"    编码后输入范围=[{encoded_input.min().item():.6f}, {encoded_input.max().item():.6f}]")
            
            # 使用梯度感知损失函数
            loss_dict = compute_gradient_aware_loss(output_pred, gt_img_flat, q, 
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
            epoch_expert_usage.append(loss_dict['expert_usage'].detach().cpu().numpy())
            
            # 计算指标
            if batch_idx % 10 == 0:
                weighted_pred = loss_dict['weighted_pred']
                mse = torch.mean((weighted_pred - gt_img_flat.squeeze(-1)) ** 2)
                psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
                mae = torch.mean(torch.abs(weighted_pred - gt_img_flat.squeeze(-1)))
                
                epoch_metrics['mse'].append(mse.item())
                epoch_metrics['psnr'].append(psnr.item())
                epoch_metrics['mae'].append(mae.item())

        # 计算epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_balance_loss = np.mean(epoch_balance_losses)
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
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_multiplicative_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, MAE: {avg_mae:.6f}")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_prediction_samples_with_gradient_info(SINR, train_loader, device, vis_outdir, epoch, num_samples=3)
                print(f"  可视化保存到: {vis_outdir}")

        scheduler.step()
        
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
    print(f"训练完成!")
    print(f"总训练时间: {total_time/60:.1f}分钟")
    print(f"最终模型保存到: {model_outdir}")
    print(f"训练可视化保存到: {vis_outdir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training RGBPose3D with MultiplicativeActivation Expert Networks')
    parser.add_argument('--config', type=str, default='configs/manager_fixed_multiplicative.yaml',
                        help='Path to config file')
    parser.add_argument('--logdir', type=str, default='./log/pose3d_multiplicative',
                        help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


if __name__ == '__main__':
    main()
