#!/usr/bin/env python3
"""
Switch NeRF++ 训练脚本
基于论文: "Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields"
参照 hash_multi_expert.py 的输入输出格式和可视化方式
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

from models.INR_SwitchNeRF_PlusPlus import SwitchNeRF_PlusPlus
from models.rgb_losses import RGBImageLoss
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
    all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)  # (1, n_experts, n_nm)
    if all_expert_preds is None:
        raise ValueError("无法找到nonmanifold_pnts_pred")
    
    # 确保形状正确: (B, n_experts, n_nm)
    if all_expert_preds.dim() == 3:  # (1, n_experts, n_nm)
        pass
    elif all_expert_preds.dim() == 4:  # (1, n_experts, n_nm, out_dim)
        all_expert_preds = all_expert_preds.squeeze(-1)  # (1, n_experts, n_nm)
    else:
        raise ValueError(f"意外的all_expert_preds形状: {all_expert_preds.shape}")
    
    # 确保q的形状正确: (B, n_experts, H*W)
    if q.dim() == 3 and q.shape[1] == all_expert_preds.shape[1]:  # (B, n_experts, H*W)
        pass
    else:
        raise ValueError(f"q形状不匹配: {q.shape}, 期望: (B, {all_expert_preds.shape[1]}, H*W)")
    
    # 关键：使用加权平均而不是argmax选择
    # q: (B, n_experts, H*W), all_expert_preds: (B, n_experts, H*W)
    # 计算加权平均: sum(q * all_expert_preds, dim=1)
    weighted_pred = torch.sum(q * all_expert_preds, dim=1)  # (B, H*W)
    
    # 重建损失 (MSE) - 使用clamp前的原始值计算loss，确保梯度可以传播
    recon_loss = torch.mean((weighted_pred - gt_img_flat.squeeze(-1)) ** 2) * recon_weight
    
    # clamp只用于可视化/保存，不影响训练（梯度在clamp前已经计算）
    weighted_pred_clamped = torch.clamp(weighted_pred, 0.0, 1.0)
    
    # Manager平衡损失
    # 计算每个专家的使用频率
    expert_usage = q.mean(dim=2)  # (B, n_experts)
    
    # 计算专家使用频率的方差（鼓励平衡使用）
    usage_variance = torch.var(expert_usage, dim=1).mean()
    
    # 平衡损失：最小化使用频率的方差
    balance_loss = usage_variance
    
    # 总损失
    total_loss = recon_loss + balance_weight * balance_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'balance_loss': balance_loss,
        'expert_usage': expert_usage.mean(dim=0),  # 平均使用频率
        'weighted_pred': weighted_pred_clamped  # 返回clamp后的预测用于可视化
    }


def evaluate_test_set(model, dataloader, device, cfg):
    """
    在测试集上评估Switch NeRF++模型
    使用加权平均进行预测
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
            
            # 使用加权平均进行预测
            q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
            if q is None:
                n_experts = cfg['MODEL']['n_experts']
                q = torch.ones(B, n_experts, H * W, device=device) / n_experts
            
            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is None:
                raise ValueError("无法获取模型预测结果")
            
            if all_expert_preds.dim() == 4:
                all_expert_preds = all_expert_preds.squeeze(-1)  # (B, k, n_nm)
            elif all_expert_preds.dim() == 3:
                pass
            
            weighted_pred = torch.sum(q * all_expert_preds, dim=1)  # (B, H*W)
            
            # 确保形状正确
            if weighted_pred.dim() == 3:  # (1, n_nm, out_dim)
                weighted_pred = weighted_pred.squeeze(0)  # (n_nm, out_dim)
            elif weighted_pred.dim() == 2:  # (1, n_nm) 或 (n_nm, out_dim)
                if weighted_pred.shape[0] == 1:
                    weighted_pred = weighted_pred.squeeze(0)  # (n_nm,)
            elif weighted_pred.dim() == 1:  # (n_nm,)
                pass
            
            # 对于灰度图，确保是1维的 (n_nm,)
            if weighted_pred.dim() == 2 and weighted_pred.shape[-1] == 1:
                weighted_pred = weighted_pred.squeeze(-1)  # (n_nm,)
            
            # 限制预测值到[0,1]范围 (用于指标计算和可视化)
            weighted_pred_clamped = torch.clamp(weighted_pred, 0.0, 1.0)
            
            # 计算指标 (使用clamp后的值)
            if C == 1:
                gt_flat = gt_img_flat.squeeze(-1)  # (B, H*W)
            else:
                gt_flat = gt_img_flat[:, :, 0]  # (B, H*W) 使用第一个通道
            
            mse = torch.mean((weighted_pred_clamped - gt_flat) ** 2)
            psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float('inf'))
            mae = torch.mean(torch.abs(weighted_pred_clamped - gt_flat))
            
            # 计算SSIM
            pred_2d = weighted_pred_clamped.reshape(B, H, W, 1).detach().cpu().numpy()
            gt_2d = gt_img_flat.reshape(B, H, W, C).detach().cpu().numpy()
            
            ssim_values = []
            for i in range(B):
                if C == 1:  # 灰度图
                    pred_slice = pred_2d[i, :, :, 0]
                    gt_slice = gt_2d[i, :, :, 0]
                else:  # RGB图
                    pred_slice = pred_2d[i, :, :, 0]
                    gt_slice = gt_2d[i, :, :, 0]
                
                ssim_val = ssim(gt_slice, pred_slice, 
                               data_range=1.0,
                               channel_axis=None)
                ssim_values.append(ssim_val)
            
            avg_ssim_batch = np.mean(ssim_values)
            
            # 计算LPIPS (如果可用)
            if lpips_model is not None:
                if weighted_pred_clamped.dim() == 1:
                    pred_img_2d = weighted_pred_clamped.reshape(1, H, W)
                elif weighted_pred_clamped.dim() == 2:
                    pred_img_2d = weighted_pred_clamped.reshape(B, H, W)
                else:
                    pred_img_2d = weighted_pred_clamped.reshape(B, H, W)
                
                if pred_img_2d.dim() == 2:
                    pred_img_2d = pred_img_2d.unsqueeze(0)
                pred_img = pred_img_2d.unsqueeze(1)  # (B, 1, H, W)
                
                if C == 1:
                    pred_img = pred_img.repeat(1, 3, 1, 1)
                    gt_img_rgb = gt_img.repeat(1, 3, 1, 1)
                else:
                    gt_img_rgb = gt_img
                
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


def save_prediction_samples_with_expert_info(model, dataloader, device, output_dir, epoch, num_samples=3, prefix='train'):
    """保存预测样本，包含专家选择信息
    
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
            
            # 获取专家权重q（用于可视化）
            q = output_pred.get('nonmnfld_q', None)
            if q is None:
                n_experts = 4
                q = torch.ones(1, n_experts, coords.shape[1], device=device) / n_experts
            
            # 使用加权平均进行预测
            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is None:
                raise ValueError("无法获取模型预测结果")
            
            if all_expert_preds.dim() == 4:
                all_expert_preds = all_expert_preds.squeeze(-1)
            weighted_pred = torch.sum(q * all_expert_preds, dim=1)
            
            # 确保形状正确
            if weighted_pred.dim() == 3:
                weighted_pred = weighted_pred.squeeze(0)
            elif weighted_pred.dim() == 2:
                if weighted_pred.shape[0] == 1:
                    weighted_pred = weighted_pred.squeeze(0)
            
            if weighted_pred.dim() == 2 and weighted_pred.shape[-1] == 1:
                weighted_pred = weighted_pred.squeeze(-1)
            
            # 重塑为图像格式
            B, C, H, W = gt_img.shape
            if weighted_pred.dim() == 2:
                weighted_pred = weighted_pred.reshape(B, H, W).unsqueeze(1)
            elif weighted_pred.dim() == 1:
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
            img_pil.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_sample_{i}_switch_nerf_plusplus.png'))
            
            # 保存专家权重可视化
            if q is not None:
                expert_selection = torch.argmax(q, dim=1)  # (1, H*W)
                expert_map = expert_selection.squeeze().reshape(H, W).cpu().numpy()
                
                # 颜色编码
                n_experts = expert_map.max() + 1
                if n_experts == 2:
                    expert_map_normalized = (expert_map * 255).astype(np.uint8)
                elif n_experts == 4:
                    expert_map_normalized = (expert_map * 85).astype(np.uint8)
                elif n_experts == 8:
                    expert_map_normalized = (expert_map * 36).astype(np.uint8)
                else:
                    expert_map_normalized = (expert_map / expert_map.max() * 255).astype(np.uint8)
                
                expert_img = Image.fromarray(expert_map_normalized, mode='L')
                expert_img.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_expert_selection_{i}_switch_nerf_plusplus.png'))
                
                # 创建彩色专家选择图
                if n_experts == 4:
                    colors = np.array([
                        [255, 0, 0],    # 红色 - 专家0 (低分辨率)
                        [0, 255, 0],    # 绿色 - 专家1 (中低分辨率)
                        [0, 0, 255],    # 蓝色 - 专家2 (中高分辨率)
                        [255, 255, 0]   # 黄色 - 专家3 (高分辨率)
                    ])
                    expert_map_color = colors[expert_map]
                    expert_img_color = Image.fromarray(expert_map_color.astype(np.uint8), mode='RGB')
                    expert_img_color.save(os.path.join(output_dir, f'epoch_{epoch}_{prefix}_expert_selection_color_{i}_switch_nerf_plusplus.png'))
                
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
    
    # 使用全局归一化版本的数据集
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
    recon_weight_match = re.match(r'(\d+(?:\.\d+)?)rgbrecon', loss_type)
    recon_weight = float(recon_weight_match.group(1)) if recon_weight_match else 100.0
    
    balance_weight_match = re.search(r'\+(\d+(?:\.\d+)?)balance', loss_type)
    balance_weight = float(balance_weight_match.group(1)) if balance_weight_match else 10.0
    
    print(f"损失类型: {loss_type}")
    print(f"重建损失权重: {recon_weight}")
    print(f"平衡损失权重: {balance_weight}")

    # 创建Switch NeRF++模型
    print("\n" + "=" * 80)
    print("创建Switch NeRF++模型...")
    print("=" * 80)
    SINR = SwitchNeRF_PlusPlus(cfg)
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")
    
    # 分别设置Manager、Expert和Hash Encoding的学习率
    # 注意：manager_net.hash_encoder的参数已经包含在manager_net.parameters()中
    # 需要手动分离，避免重复
    
    # 获取Manager网络的参数（包括哈希编码器和MLP）
    all_manager_params = dict(SINR.manager_net.named_parameters())
    
    # 分离Manager哈希编码器和Manager MLP的参数
    manager_hash_params = []
    manager_mlp_params = []
    
    for name, param in all_manager_params.items():
        if 'hash_encoder' in name:
            manager_hash_params.append(param)
        else:
            manager_mlp_params.append(param)
    
    # 其他参数组
    # decoder_experts是一个ModuleList，需要收集所有专家的参数
    expert_params = []
    for decoder in SINR.decoder_experts:
        expert_params.extend(list(decoder.parameters()))
    hash_encoder_params = list(SINR.heterogeneous_hash_encoder.parameters())
    
    # 调整学习率策略
    base_lr = cfg['TRAINING']['lr']
    hash_lr = base_lr * 5.0      # 哈希编码学习率更高
    manager_lr = base_lr * 1.5   # Manager稍高
    expert_lr = base_lr          # Expert基础学习率
    
    # 构建优化器参数组（确保没有重复参数）
    param_groups = []
    
    if manager_mlp_params:
        param_groups.append({'params': manager_mlp_params, 'lr': manager_lr, 'name': 'manager_mlp'})
    if manager_hash_params:
        param_groups.append({'params': manager_hash_params, 'lr': hash_lr, 'name': 'manager_hash'})
    if expert_params:
        param_groups.append({'params': expert_params, 'lr': expert_lr, 'name': 'experts'})
    if hash_encoder_params:
        param_groups.append({'params': hash_encoder_params, 'lr': hash_lr, 'name': 'heterogeneous_hash_encoding'})
    
    # 验证参数没有重复
    all_param_ids = set()
    for group in param_groups:
        for param in group['params']:
            param_id = id(param)
            if param_id in all_param_ids:
                raise ValueError(f"发现重复参数: {group['name']}")
            all_param_ids.add(param_id)
    
    print(f"优化器参数组设置:")
    for group in param_groups:
        param_count = sum(p.numel() for p in group['params'])
        print(f"  - {group['name']}: {param_count:,} 参数, 学习率: {group['lr']:.2e}")
    
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
    patience = 200
    patience_counter = 0
    
    print(f"\n开始训练Switch NeRF++模型...")
    print(f"总epoch数: {total_epochs}")
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
                
                # 计算SSIM
                pred_2d = weighted_pred.reshape(B, H, W, 1).detach().cpu().numpy()
                gt_2d = gt_img_flat.reshape(B, H, W, C).detach().cpu().numpy()
                
                ssim_values = []
                for j in range(B):
                    if C == 1:  # 灰度图
                        pred_slice = pred_2d[j, :, :, 0]
                        gt_slice = gt_2d[j, :, :, 0]
                    else:
                        pred_slice = pred_2d[j, :, :, 0]
                        gt_slice = gt_2d[j, :, :, 0]
                    
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
        
        # 每个epoch都保存测试集可视化（全部测试集）
        print(f"  生成测试集可视化 (Epoch {epoch+1})...")
        save_prediction_samples_with_expert_info(SINR, test_loader, device, vis_outdir, epoch, num_samples=len(test_set), prefix='test')
        print(f"  测试集可视化保存到: {vis_outdir}")

        # 保存模型
        if epoch % save_interval == 0 or epoch == total_epochs - 1:
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_switch_nerf_plusplus_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}, MAE: {avg_mae:.6f}")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_prediction_samples_with_expert_info(SINR, train_loader, device, vis_outdir, epoch, num_samples=3, prefix='train')
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
    parser = argparse.ArgumentParser(description='Training RGBPose3D with Switch NeRF++')
    parser.add_argument('--config', type=str, default='configs/comparison_methods/switch_nerf_plusplus.yaml',
                        help='Path to config file')
    parser.add_argument('--logdir', type=str, default='./log/switch_nerf_plusplus',
                        help='Directory to save logs and models')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


if __name__ == '__main__':
    main()

