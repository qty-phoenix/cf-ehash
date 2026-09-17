#!/usr/bin/env python3
"""
修复Manager网络梯度问题的训练脚本
关键修复：使用加权平均而不是argmax选择，确保梯度能够传播到Manager网络
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
import math
from skimage.metrics import structural_similarity as ssim

# 尝试导入LPIPS，如果未安装则设为None
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: lpips库未安装，LPIPS指标将不可用。请运行: pip install lpips")

from models import build_model
from datasets.RGBPose3D_Cached import RGBPose3DDataset
from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm


def parse_loss_weights_from_loss_type(loss_type: str):
    """从类似 '100rgbrecon+0.1balance' 的字符串解析权重。"""
    recon_weight_match = re.match(r'(\d+(?:\.\d+)?)rgbrecon', loss_type)
    recon_weight = float(recon_weight_match.group(1)) if recon_weight_match else 100.0

    balance_weight_match = re.search(r'\+(\d+(?:\.\d+)?)balance', loss_type)
    balance_weight = float(balance_weight_match.group(1)) if balance_weight_match else 10.0
    return recon_weight, balance_weight


def _dedup_params(params):
    """去重参数（避免同一个 parameter 被加入多个 param group）。"""
    seen = set()
    out = []
    for p in params:
        if p is None:
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def collect_named_param_groups(model, cfg):
    """
    收集训练可能涉及到的参数组，并给每个组一个稳定 name，便于阶段训练切换时调 lr / 冻结。
    """
    groups = {}

    # Core nets
    groups['manager'] = list(model.manager_net.parameters()) if hasattr(model, 'manager_net') else []
    groups['experts'] = list(model.decoder.parameters()) if hasattr(model, 'decoder') else []

    # Encoders (可能是 PE/FF 这种无参数模块，也可能是 learned/HE 有参数)
    groups['experts_encoder'] = list(model.decoder_input_encoding_module.parameters()) if hasattr(model, 'decoder_input_encoding_module') else []
    groups['manager_encoder'] = list(model.manager_input_encoding_module.parameters()) if hasattr(model, 'manager_input_encoding_module') else []

    # Hash encoders（单独拿出来便于单独 lr；同时从 encoder 组里剔除，避免重复）
    expert_hash_params = []
    if hasattr(model, 'decoder_input_encoding_module') and hasattr(model.decoder_input_encoding_module, 'hash_encoder'):
        if model.decoder_input_encoding_module.hash_encoder is not None:
            expert_hash_params = list(model.decoder_input_encoding_module.hash_encoder.parameters())
    groups['hash_encoding'] = expert_hash_params

    manager_hash_params = []
    if hasattr(model, 'manager_input_encoding_module') and hasattr(model.manager_input_encoding_module, 'hash_encoder'):
        if model.manager_input_encoding_module.hash_encoder is not None:
            manager_hash_params = list(model.manager_input_encoding_module.hash_encoder.parameters())
    groups['manager_hash'] = manager_hash_params

    # Conditioner（仅当 conditioning 是 CNN/FCN/expert_weights 时可能有可训练参数）
    cond_params = []
    if hasattr(model, 'manager_conditioner') and hasattr(model.manager_conditioner, 'cond_encoding'):
        cond_params = list(model.manager_conditioner.cond_encoding.parameters())
    groups['manager_conditioner'] = cond_params

    # 从 encoder 组里剔除 hash 子模块参数，避免重复
    if groups['hash_encoding'] and groups['experts_encoder']:
        hash_ids = {id(p) for p in groups['hash_encoding']}
        groups['experts_encoder'] = [p for p in groups['experts_encoder'] if id(p) not in hash_ids]
    if groups['manager_hash'] and groups['manager_encoder']:
        hash_ids = {id(p) for p in groups['manager_hash']}
        groups['manager_encoder'] = [p for p in groups['manager_encoder'] if id(p) not in hash_ids]

    # 最终去重 & 过滤空组
    for k in list(groups.keys()):
        groups[k] = _dedup_params(groups[k])
        # PE/FF 这类可能没有任何参数，保留空列表即可
    return groups


def set_requires_grad_for_groups(groups: dict, enabled_group_names: set):
    """按组开关 requires_grad。"""
    for gname, params in groups.items():
        flag = gname in enabled_group_names
        for p in params:
            p.requires_grad = flag


def compute_stage_schedule(cfg):
    """
    把 TRAINING.stages 解析为按 epoch 的阶段边界。
    返回: (stages, enabled) 其中 stages 每项包含 end_epoch(int), params(str), loss_type(str)
    """
    stages = cfg.get('TRAINING', {}).get('stages', None)
    if not stages:
        return [], False

    total_epochs = int(cfg['TRAINING']['num_epochs'])
    parsed = []
    last_end = 0
    for i, s in enumerate(stages):
        end_frac = float(s['end_iteration_frac'])
        # 用 ceil 确保覆盖到最后一个 epoch；并强制单调递增
        end_epoch = int(math.ceil(end_frac * total_epochs))
        end_epoch = max(end_epoch, last_end + 1)
        end_epoch = min(end_epoch, total_epochs)
        last_end = end_epoch
        parsed.append({
            'end_epoch': end_epoch,
            'params': s['params'],
            'loss_type': s.get('loss_type', cfg['LOSS']['loss_type']),
        })
    # 最后一段必须覆盖到 total_epochs
    if parsed and parsed[-1]['end_epoch'] < total_epochs:
        parsed[-1]['end_epoch'] = total_epochs
    return parsed, True


def stage_to_enabled_groups(stage_params: str):
    """
    将 stages 里的 params 字段映射到我们上面定义的 param groups name 集合。
    """
    if stage_params == 'manager':
        return {'manager', 'manager_encoder', 'manager_hash', 'manager_conditioner'}
    if stage_params == 'experts':
        return {'experts', 'experts_encoder', 'hash_encoding'}
    if stage_params == 'experts_encoder':
        # 编码器（含 hash）单独训练
        return {'experts_encoder', 'hash_encoding'}
    if stage_params == 'full_experts':
        return {'experts', 'experts_encoder', 'hash_encoding'}
    if stage_params == 'all':
        return {'manager', 'manager_encoder', 'manager_hash', 'manager_conditioner',
                'experts', 'experts_encoder', 'hash_encoding'}
    # 兜底：全训练
    return {'manager', 'manager_encoder', 'manager_hash', 'manager_conditioner',
            'experts', 'experts_encoder', 'hash_encoding'}


def apply_stage_to_optimizer(optimizer, lr_by_group: dict, enabled_group_names: set):
    """
    通过把非本阶段的 param group lr 设为 0 来“硬关闭”更新（配合 requires_grad 双保险）。
    依赖我们创建 optimizer 时给每个 param_group 加了 name 字段。
    """
    for pg in optimizer.param_groups:
        name = pg.get('name', None)
        if not name:
            continue
        if name in enabled_group_names:
            pg['lr'] = lr_by_group.get(name, pg['lr'])
        else:
            pg['lr'] = 0.0


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


def analyze_expert_clamping_stats(output_pred, q, device):
    """
    分析每个专家的置零统计（clamp前的原始值）
    检测每个专家输出和加权平均后的值，统计会被clamp到0或1的数量
    
    Args:
        output_pred: 模型输出（decoder的原始输出，可能包含负值）
        q: 专家权重 (B, n_experts, H*W)
        device: 设备
    
    Returns:
        dict: 每个专家的置零统计信息
    """
    all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
    if all_expert_preds is None:
        return None
    
    # 确保形状正确
    if all_expert_preds.dim() == 4:
        all_expert_preds = all_expert_preds.squeeze(-1)  # (B, k, n_nm)
    elif all_expert_preds.dim() == 3:
        pass
    
    B, n_experts, n_points = all_expert_preds.shape
    
    # 计算加权平均（clamp前）
    weighted_pred = torch.sum(q * all_expert_preds, dim=1)  # (B, H*W)
    
    # 统计每个专家的原始输出范围
    expert_stats = {}
    for expert_idx in range(n_experts):
        expert_output = all_expert_preds[0, expert_idx, :]  # (n_points,)
        
        # 统计专家输出的原始值（decoder输出，可能包含负值）
        below_zero = torch.sum(expert_output < 0.0).item()
        above_one = torch.sum(expert_output > 1.0).item()
        total_points = expert_output.numel()
        
        # 统计加权后的贡献（q * expert_output）
        expert_weighted = q[0, expert_idx, :] * expert_output  # (n_points,)
        weighted_below_zero = torch.sum(expert_weighted < 0.0).item()
        weighted_above_one = torch.sum(expert_weighted > 1.0).item()
        
        expert_stats[expert_idx] = {
            'expert_output_min': expert_output.min().item(),
            'expert_output_max': expert_output.max().item(),
            'expert_output_mean': expert_output.mean().item(),
            'expert_below_zero': below_zero,
            'expert_above_one': above_one,
            'expert_below_zero_pct': below_zero / total_points * 100 if total_points > 0 else 0,
            'expert_above_one_pct': above_one / total_points * 100 if total_points > 0 else 0,
            'weighted_below_zero': weighted_below_zero,
            'weighted_above_one': weighted_above_one,
            'weighted_contribution_mean': expert_weighted.mean().item(),
        }
    
    # 统计加权平均后的整体置零情况（这是实际会被clamp的值）
    weighted_below_zero_total = torch.sum(weighted_pred < 0.0).item()
    weighted_above_one_total = torch.sum(weighted_pred > 1.0).item()
    total_weighted_points = weighted_pred.numel()
    
    expert_stats['weighted_avg'] = {
        'weighted_pred_min': weighted_pred.min().item(),
        'weighted_pred_max': weighted_pred.max().item(),
        'weighted_pred_mean': weighted_pred.mean().item(),
        'weighted_below_zero_total': weighted_below_zero_total,
        'weighted_above_one_total': weighted_above_one_total,
        'weighted_below_zero_pct': weighted_below_zero_total / total_weighted_points * 100 if total_weighted_points > 0 else 0,
        'weighted_above_one_pct': weighted_above_one_total / total_weighted_points * 100 if total_weighted_points > 0 else 0,
    }
    
    return expert_stats


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
    
    # 🔥 关键修复：在clamp前计算loss，避免梯度死区
    # 1. 重建损失 (MSE) - 使用clamp前的原始值计算loss，确保梯度可以传播
    recon_loss = torch.mean((weighted_pred - gt_img_flat.squeeze(-1)) ** 2) * recon_weight
    
    # clamp只用于可视化/保存，不影响训练
    weighted_pred_clamped = torch.clamp(weighted_pred, 0.0, 1.0)
    
    # 2. Manager平衡损失
    # 计算每个专家的使用频率
    n_experts_actual = q.shape[1]
    if n_experts_actual > 1:
        expert_usage = q.mean(dim=2)  # (B, n_experts)
        usage_variance = torch.var(expert_usage, dim=1).mean()
        balance_loss = usage_variance
    else:
        expert_usage = q.mean(dim=2)  # (B, 1)
        balance_loss = torch.tensor(0.0, device=q.device)

    # 3. 总损失
    total_loss = recon_loss + balance_weight * balance_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'balance_loss': balance_loss,
        'expert_usage': expert_usage.mean(dim=0),  # 平均使用频率
        'weighted_pred': weighted_pred_clamped,  # 返回clamp后的值用于可视化
        'weighted_pred_raw': weighted_pred  # 返回clamp前的原始值（用于调试）
    }


def evaluate_test_set(model, dataloader, device, cfg):
    """
    在测试集上评估多专家模型
    使用加权平均进行预测（所有专家加权）
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
            
            # 获取专家权重q
            q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
            if q is None:
                n_experts = cfg['MODEL']['n_experts']
                q = torch.ones(B, n_experts, H * W, device=device) / n_experts
            
            # 获取所有专家的预测结果
            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is None:
                raise ValueError("无法获取模型预测结果")
            
            # 确保形状正确
            if all_expert_preds.dim() == 4:
                all_expert_preds = all_expert_preds.squeeze(-1)  # (B, k, n_nm)
            elif all_expert_preds.dim() == 3:
                pass
            
            # 计算加权预测（使用加权平均）
            weighted_pred = torch.sum(q * all_expert_preds, dim=1)  # (B, H*W)
            
            # 限制预测值到[0,1]范围 (用于指标计算和可视化)
            weighted_pred_clamped = torch.clamp(weighted_pred, 0.0, 1.0)
            
            # 计算指标 (使用clamp后的值)
            # 对于灰度图，gt_img_flat.squeeze(-1) 得到 (B, H*W)
            # 对于RGB图，使用第一个通道
            if C == 1:
                gt_flat = gt_img_flat.squeeze(-1)  # (B, H*W)
            else:
                gt_flat = gt_img_flat[:, :, 0]  # (B, H*W) 使用第一个通道
            
            # 确保形状匹配
            if weighted_pred_clamped.dim() == 2:
                if weighted_pred_clamped.shape[0] == 1:
                    weighted_pred_clamped = weighted_pred_clamped.squeeze(0)  # (H*W,)
            elif weighted_pred_clamped.dim() == 1:
                pass
            
            if gt_flat.dim() == 2:
                if gt_flat.shape[0] == 1:
                    gt_flat = gt_flat.squeeze(0)  # (H*W,)
            
            # 计算指标（使用clamp后的值）
            mse = torch.mean((weighted_pred_clamped - gt_flat) ** 2)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
            mae = torch.mean(torch.abs(weighted_pred_clamped - gt_flat))
            
            # 计算SSIM (使用clamp后的值)
            pred_2d = weighted_pred_clamped.reshape(B, H, W, 1).detach().cpu().numpy()
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
                if weighted_pred_clamped.dim() == 1:
                    pred_img_2d = weighted_pred_clamped.reshape(1, H, W)
                elif weighted_pred_clamped.dim() == 2:
                    pred_img_2d = weighted_pred_clamped.reshape(B, H, W)
                else:
                    pred_img_2d = weighted_pred_clamped.reshape(B, H, W)
                
                # 确保是(B, H, W)格式，然后添加通道维度
                if pred_img_2d.dim() == 2:
                    pred_img_2d = pred_img_2d.unsqueeze(0)
                pred_img = pred_img_2d.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
                
                # 对于灰度图，复制为RGB格式；对于RGB图，保持原样
                if C == 1:
                    pred_img = pred_img.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
                    gt_img_rgb = gt_img.repeat(1, 3, 1, 1)  # (B, 1, H, W) -> (B, 3, H, W)
                else:
                    gt_img_rgb = gt_img
                
                # 🔥 确保输入值在[0,1]范围内（LPIPS归一化的前提条件）
                pred_img = torch.clamp(pred_img, 0.0, 1.0)
                gt_img_rgb = torch.clamp(gt_img_rgb, 0.0, 1.0)
                
                # LPIPS需要输入在[-1, 1]范围内（官方要求）
                pred_img_lpips = pred_img * 2.0 - 1.0  # [0, 1] -> [-1, 1]
                gt_img_lpips = gt_img_rgb * 2.0 - 1.0  # [0, 1] -> [-1, 1]
                
                # 计算LPIPS (返回的是tensor，需要取平均值)
                lpips_val = lpips_model(pred_img_lpips, gt_img_lpips)
                # lpips_model返回的是(B,)形状的tensor，取平均值
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
    
    # 如果LPIPS可用，添加到结果中
    if all_lpips:
        result['lpips'] = np.mean(all_lpips)
    else:
        result['lpips'] = None
    
    return result


def save_prediction_samples_with_gradient_info(model, dataloader, device, output_dir, epoch, num_samples=3, prefix='train'):
    """保存预测样本，包含梯度信息
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        output_dir: 输出目录
        epoch: epoch编号
        num_samples: 保存的样本数量
        prefix: 文件名前缀 ('train' 或 'test')
    """
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
            img_pil.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_sample_{i}_gradient_fixed.png'))
            
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
                expert_img.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_expert_selection_{i}_gradient_fixed.png'))
                
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
                    expert_img_color.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_expert_selection_color_{i}_gradient_fixed.png'))
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
                    expert_img_color.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_expert_selection_color_{i}_gradient_fixed.png'))
                
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
    
    # 使用全局归一化版本的数据集（关键修复！）
    use_global_norm = cfg['DATA'].get('use_global_norm', True)
    print(f"使用全局归一化: {use_global_norm}")
    
    # 检查是否使用uvfdata2标定矩阵
    use_uvfdata2_calibration = cfg['DATA'].get('use_uvfdata2_calibration', False)
    print("=" * 80)
    if use_uvfdata2_calibration:
        print("✅ 启用uvfdata2标定矩阵变换:")
        print("   - S矩阵 (scaling_from_pixel_to_mm): 像素单位 → 毫米单位")
        print("     * x方向: 1像素 = 0.2294 mm")
        print("     * y方向: 1像素 = 0.2209 mm")
        print("   - C矩阵 (spatial_calibration): 图像坐标系 → 工具坐标系")
        print("     * 包含旋转和平移变换")
        print("   - 变换流程: 像素坐标 → S矩阵 → C矩阵 → R/T矩阵 → 世界坐标")
    else:
        print("❌ 未启用uvfdata2标定矩阵")
        print("   - 直接使用像素坐标进行R/T变换")
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
    recon_weight, balance_weight = parse_loss_weights_from_loss_type(loss_type)
    print(f"损失类型: {loss_type}")
    print(f"重建损失权重: {recon_weight}")
    print(f"平衡损失权重: {balance_weight}")

    # 模型和损失函数
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")

    # 训练阶段（可选）：如果 YAML 里配置了 TRAINING.stages，则启用“阶段冻结/解冻 + 分阶段 loss 权重”
    stage_schedule, use_stages = compute_stage_schedule(cfg)
    if use_stages:
        print(f"✅ 检测到 TRAINING.stages，启用分阶段训练，共 {len(stage_schedule)} 个阶段")
        for i, s in enumerate(stage_schedule):
            print(f"  - Stage {i}: end_epoch={s['end_epoch']}, params={s['params']}, loss_type={s['loss_type']}")
    else:
        print("ℹ️ 未配置 TRAINING.stages，使用单阶段训练（保持原逻辑）")

    # 收集参数组（支持阶段训练时冻结/解冻；并支持单独 lr）
    named_groups = collect_named_param_groups(SINR, cfg)
    if named_groups.get('hash_encoding'):
        print(f"检测到哈希编码(Expert)，参数数量: {sum(p.numel() for p in named_groups['hash_encoding']):,}")
    if named_groups.get('manager_hash'):
        print(f"检测到哈希编码(Manager)，参数数量: {sum(p.numel() for p in named_groups['manager_hash']):,}")

    # 学习率策略：保持你原来的默认（hash=5x，manager=1.5x，expert=1x），并允许 TRAINING.lr 是 float 或 dict
    if isinstance(cfg['TRAINING']['lr'], dict):
        base_lr = float(cfg['TRAINING']['lr'].get('all', 1.0e-4))
    else:
        base_lr = float(cfg['TRAINING']['lr'])
    hash_lr = base_lr * 5.0
    manager_lr = base_lr * 1.5
    expert_lr = base_lr

    # 可选：encoder 单独 lr（默认跟随对应分支）
    experts_encoder_lr = expert_lr
    manager_encoder_lr = manager_lr
    manager_hash_lr = hash_lr

    lr_by_group = {
        'manager': manager_lr,
        'experts': expert_lr,
        'experts_encoder': experts_encoder_lr,
        'manager_encoder': manager_encoder_lr,
        'hash_encoding': hash_lr,
        'manager_hash': manager_hash_lr,
        'manager_conditioner': manager_lr,
    }

    # 初始化 optimizer：把所有潜在组都放进去，阶段切换时通过 requires_grad + lr=0 双保险
    param_groups = []
    for gname in ['manager', 'experts', 'experts_encoder', 'manager_encoder', 'hash_encoding', 'manager_hash', 'manager_conditioner']:
        params = named_groups.get(gname, [])
        if params:
            param_groups.append({'params': params, 'lr': lr_by_group[gname], 'name': gname})
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.999))
    
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
    test_interval = 1  # 每个epoch都测试
    
    # 早停机制
    best_loss = float('inf')
    patience = 200  # 200个epoch没有改善就停止
    patience_counter = 0
    
    print(f"\n开始训练...")
    print(f"总epoch数: {total_epochs}")
    print(f"Manager学习率: {manager_lr:.2e}")
    print(f"Expert学习率: {expert_lr:.2e}")
    if named_groups.get('hash_encoding'):
        print(f"哈希编码(Expert)学习率: {hash_lr:.2e} (高5倍，加速学习)")
    if named_groups.get('manager_hash'):
        print(f"哈希编码(Manager)学习率: {manager_hash_lr:.2e}")
    print(f"模型保存间隔: {save_interval}")
    print(f"可视化间隔: {vis_interval}")
    print(f"测试集评估间隔: {test_interval}")
    print(f"早停耐心值: {patience} epochs")
    print("=" * 80)

    start_time = time.time()
    
    current_stage_idx = -1
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        SINR.train()

        # 阶段切换（按 epoch）：冻结/解冻参数 + 更新 loss 权重 + 更新各组 lr
        if use_stages:
            # 找到当前 epoch 属于哪个 stage
            stage_idx = 0
            for i, s in enumerate(stage_schedule):
                if (epoch + 1) <= s['end_epoch']:
                    stage_idx = i
                    break
            if stage_idx != current_stage_idx:
                current_stage_idx = stage_idx
                stage = stage_schedule[current_stage_idx]

                cfg['LOSS']['loss_type'] = stage['loss_type']
                loss_type = stage['loss_type']
                recon_weight, balance_weight = parse_loss_weights_from_loss_type(loss_type)

                enabled = stage_to_enabled_groups(stage['params'])
                set_requires_grad_for_groups(named_groups, enabled)
                apply_stage_to_optimizer(optimizer, lr_by_group, enabled)

                print(f"\n{'='*80}")
                print(f"切换到 Stage {current_stage_idx}: params={stage['params']}, end_epoch={stage['end_epoch']}")
                print(f"loss_type={loss_type} | recon_weight={recon_weight} | balance_weight={balance_weight}")
                print(f"启用参数组: {sorted(list(enabled))}")
                print(f"{'='*80}\n")
        
        epoch_losses = []
        epoch_recon_losses = []
        epoch_balance_losses = []
        epoch_metrics = {'mse': [], 'psnr': [], 'mae': [], 'ssim': []}
        epoch_expert_usage = []
        
        for batch_idx, data in enumerate(train_loader):
            coords = data['coords'].to(device)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)
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
            
            # 调试信息（仅第一个epoch的前几个batch）
            if epoch == 0 and batch_idx < 3:
                print(f"  Batch {batch_idx}: q形状={q.shape}, q范围=[{q.min().item():.6f}, {q.max().item():.6f}]")
                expert_selection = torch.argmax(q, dim=1) if q.dim() == 3 else None
                if expert_selection is not None:
                    unique_experts = torch.unique(expert_selection)
                    print(f"    选择的专家: {unique_experts.cpu().numpy()}")
            
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
            
            # 计算指标（使用与测试集相同的评估逻辑，确保一致性）
            if batch_idx % 10 == 0:
                # 🔥 关键修复：使用eval模式重新前向传播，与测试集评估逻辑保持一致
                with torch.no_grad():
                    SINR.eval()
                    # 重新前向传播（使用eval模式）
                    output_pred_eval = SINR(coords, dino=dino, img=gt_img)
                    SINR.train()  # 恢复训练模式
                    
                    # 获取专家权重q（与测试集评估逻辑一致）
                    q_eval = output_pred_eval.get('nonmnfld_q', output_pred_eval.get('mnfld_q', None))
                    if q_eval is None:
                        n_experts = cfg['MODEL']['n_experts']
                        q_eval = torch.ones(B, n_experts, H * W, device=device) / n_experts
                    
                    # 🔥 使用加权平均进行预测（与测试集评估逻辑一致）
                    all_expert_preds_eval = output_pred_eval.get('nonmanifold_pnts_pred', None)
                    if all_expert_preds_eval is None:
                        raise ValueError("无法获取模型预测结果")
                    
                    # 确保形状正确
                    if all_expert_preds_eval.dim() == 4:
                        all_expert_preds_eval = all_expert_preds_eval.squeeze(-1)  # (B, k, n_nm)
                    elif all_expert_preds_eval.dim() == 3:
                        pass
                    
                    # 计算加权预测（使用加权平均）
                    weighted_pred_eval = torch.sum(q_eval * all_expert_preds_eval, dim=1)  # (B, H*W)
                    
                    # 限制预测值到[0,1]范围（与测试集评估逻辑一致）
                    weighted_pred_clamped = torch.clamp(weighted_pred_eval, 0.0, 1.0)
                    
                    # 处理gt_flat形状（与测试集评估逻辑一致）
                    if C == 1:
                        gt_flat_eval = gt_img_flat.squeeze(-1)  # (B, H*W)
                    else:
                        gt_flat_eval = gt_img_flat[:, :, 0]  # (B, H*W) 使用第一个通道
                    
                    # 确保形状匹配
                    if weighted_pred_clamped.dim() == 2:
                        if weighted_pred_clamped.shape[0] == 1:
                            weighted_pred_clamped = weighted_pred_clamped.squeeze(0)  # (H*W,)
                    elif weighted_pred_clamped.dim() == 1:
                        pass
                    
                    if gt_flat_eval.dim() == 2:
                        if gt_flat_eval.shape[0] == 1:
                            gt_flat_eval = gt_flat_eval.squeeze(0)  # (H*W,)
                    
                    # 计算指标（与测试集评估逻辑一致）
                    # 使用clamp后的值计算PSNR（与实际显示一致）
                    mse = torch.mean((weighted_pred_clamped - gt_flat_eval) ** 2)
                    psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
                    mae = torch.mean(torch.abs(weighted_pred_clamped - gt_flat_eval))
                    
                    # 计算SSIM（与测试集评估逻辑一致）
                    pred_2d = weighted_pred_clamped.reshape(B, H, W, 1).detach().cpu().numpy()
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
                
                epoch_metrics['mse'].append(mse.item())
                epoch_metrics['psnr'].append(psnr.item())
                epoch_metrics['mae'].append(mae.item())
                epoch_metrics['ssim'].append(avg_ssim_batch)

        # 计算epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_recon_loss = np.mean(epoch_recon_losses)
        avg_balance_loss = np.mean(epoch_balance_losses)
        avg_mse = np.mean(epoch_metrics['mse']) if epoch_metrics['mse'] else 0
        avg_psnr = np.mean(epoch_metrics['psnr']) if epoch_metrics['psnr'] else 0
        avg_mae = np.mean(epoch_metrics['mae']) if epoch_metrics['mae'] else 0
        avg_ssim = np.mean(epoch_metrics['ssim']) if epoch_metrics['ssim'] else 0
        
        # 计算专家使用统计
        expert_usage_avg = np.mean(epoch_expert_usage, axis=0) if epoch_expert_usage else np.zeros(cfg['MODEL']['n_experts'])
        
        # 🔥 在epoch 100时检测每个专家的置零统计
        if epoch == 100:
            print(f"\n{'='*80}")
            print(f"Epoch 100: 检测每个专家的置零统计")
            print(f"{'='*80}")
            
            # 收集一个batch的数据进行统计
            SINR.eval()
            all_expert_stats = []
            with torch.no_grad():
                for batch_idx, data in enumerate(train_loader):
                    if batch_idx >= 5:  # 只检查前5个batch
                        break
                    
                    coords = data['coords'].to(device)
                    coords = coords.squeeze(1)
                    gt_img = data['gt_img'].to(device)
                    dino = data['dino'].to(device)
                    
                    output_pred = SINR(coords, dino=dino, img=gt_img)
                    B, C, H, W = gt_img.shape
                    
                    q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
                    if q is None:
                        n_experts = cfg['MODEL']['n_experts']
                        q = torch.ones(B, n_experts, H * W, device=device) / n_experts
                    
                    # 分析置零统计
                    stats = analyze_expert_clamping_stats(output_pred, q, device)
                    if stats is not None:
                        all_expert_stats.append(stats)
            
            SINR.train()
            
            # 汇总统计信息
            if all_expert_stats:
                n_experts = cfg['MODEL']['n_experts']
                print(f"\n每个专家的置零统计（基于{len(all_expert_stats)}个batch的平均）:")
                print(f"{'-'*80}")
                
                for expert_idx in range(n_experts):
                    expert_below_zero_list = [s[expert_idx]['expert_below_zero'] for s in all_expert_stats]
                    expert_above_one_list = [s[expert_idx]['expert_above_one'] for s in all_expert_stats]
                    expert_below_zero_pct_list = [s[expert_idx]['expert_below_zero_pct'] for s in all_expert_stats]
                    expert_above_one_pct_list = [s[expert_idx]['expert_above_one_pct'] for s in all_expert_stats]
                    expert_output_min_list = [s[expert_idx]['expert_output_min'] for s in all_expert_stats]
                    expert_output_max_list = [s[expert_idx]['expert_output_max'] for s in all_expert_stats]
                    expert_output_mean_list = [s[expert_idx]['expert_output_mean'] for s in all_expert_stats]
                    
                    avg_below_zero = np.mean(expert_below_zero_list)
                    avg_above_one = np.mean(expert_above_one_list)
                    avg_below_zero_pct = np.mean(expert_below_zero_pct_list)
                    avg_above_one_pct = np.mean(expert_above_one_pct_list)
                    avg_min = np.mean(expert_output_min_list)
                    avg_max = np.mean(expert_output_max_list)
                    avg_mean = np.mean(expert_output_mean_list)
                    
                    print(f"专家 {expert_idx}:")
                    print(f"  输出范围: [{avg_min:.6f}, {avg_max:.6f}], 均值: {avg_mean:.6f}")
                    print(f"  小于0的值: {avg_below_zero:.0f} ({avg_below_zero_pct:.2f}%)")
                    print(f"  大于1的值: {avg_above_one:.0f} ({avg_above_one_pct:.2f}%)")
                
                # 加权平均后的统计
                weighted_below_zero_list = [s['weighted_avg']['weighted_below_zero_total'] for s in all_expert_stats]
                weighted_above_one_list = [s['weighted_avg']['weighted_above_one_total'] for s in all_expert_stats]
                weighted_below_zero_pct_list = [s['weighted_avg']['weighted_below_zero_pct'] for s in all_expert_stats]
                weighted_above_one_pct_list = [s['weighted_avg']['weighted_above_one_pct'] for s in all_expert_stats]
                weighted_min_list = [s['weighted_avg']['weighted_pred_min'] for s in all_expert_stats]
                weighted_max_list = [s['weighted_avg']['weighted_pred_max'] for s in all_expert_stats]
                weighted_mean_list = [s['weighted_avg']['weighted_pred_mean'] for s in all_expert_stats]
                
                avg_weighted_below_zero = np.mean(weighted_below_zero_list)
                avg_weighted_above_one = np.mean(weighted_above_one_list)
                avg_weighted_below_zero_pct = np.mean(weighted_below_zero_pct_list)
                avg_weighted_above_one_pct = np.mean(weighted_above_one_pct_list)
                avg_weighted_min = np.mean(weighted_min_list)
                avg_weighted_max = np.mean(weighted_max_list)
                avg_weighted_mean = np.mean(weighted_mean_list)
                
                print(f"\n加权平均后的统计（实际会被clamp的值）:")
                print(f"  输出范围: [{avg_weighted_min:.6f}, {avg_weighted_max:.6f}], 均值: {avg_weighted_mean:.6f}")
                print(f"  小于0的值: {avg_weighted_below_zero:.0f} ({avg_weighted_below_zero_pct:.2f}%)")
                print(f"  大于1的值: {avg_weighted_above_one:.0f} ({avg_weighted_above_one_pct:.2f}%)")
                print(f"{'='*80}\n")
        
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
                  f"SSIM: {avg_ssim:.4f} | "
                  f"MAE: {avg_mae:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Progress: {progress:.1f}%")
            
            # 打印专家使用情况
            expert_usage_str = " | ".join([f"E{i}: {usage:.3f}" for i, usage in enumerate(expert_usage_avg)])
            print(f"  专家使用: {expert_usage_str}")

        # 测试集评估（每个epoch）
        if (epoch + 1) % test_interval == 0 or epoch == total_epochs - 1:
            print(f"\n{'='*80}")
            print(f"测试集评估 (Epoch {epoch+1})...")
            test_metrics = evaluate_test_set(SINR, test_loader, device, cfg)
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
        save_prediction_samples_with_gradient_info(SINR, test_loader, device, vis_outdir, epoch, num_samples=10, prefix='test')
        print(f"  测试集可视化保存到: {vis_outdir}")

        # 保存模型
        if epoch % save_interval == 0 or epoch == total_epochs - 1:
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_gradient_fixed_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.6f}")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_prediction_samples_with_gradient_info(SINR, train_loader, device, vis_outdir, epoch, num_samples=3, prefix='train')
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
    parser = argparse.ArgumentParser(description='Training RGBPose3D with Gradient Fixed Manager Network')
    parser.add_argument('--config', type=str, default='configs/manager_fixed.yaml',
                        help='Path to config file')
    parser.add_argument('--logdir', type=str, default='./log/pose3d_gradient_fixed',
                        help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


if __name__ == '__main__':
    main()

