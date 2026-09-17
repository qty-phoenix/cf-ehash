#!/usr/bin/env python3
"""
全功能版 RGB Pose3D 训练脚本（MoE 管线）

相对于 `gradient_fixed.py` 的关键补强：
1. 引入 soft-MoE 训练流程（TrainingStageHandler + RGBImageLoss）
2. 所有专家参与重建与路由损失，支持论文中的平衡/分割/熵等项
3. 支持 Manager 预训练权重、双编码器输入与温度可学习 softmax
4. 训练阶段与 YAML 参数完全对齐，确保网络结构一致
"""

import argparse
import os
import sys
import time
import shutil
from typing import Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets.RGBPose3D_Cached import RGBPose3DDataset
from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm


def parse_args():
    parser = argparse.ArgumentParser(description='Pose3D RGB MoE Training (full pipeline)')
    parser.add_argument('--config', type=str, default='configs/config_RGB_pose3d_manager_fixed.yaml',
                        help='配置文件路径')
    parser.add_argument('--logdir', type=str, default='./log/pose3d_moe_full',
                        help='日志与模型输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--use_global_norm', action='store_true',
                        help='强制使用全局归一化版本数据集（默认读取 YAML）')
    parser.add_argument('--save_visual_interval', type=int, default=200,
                        help='可视化保存间隔（按 epoch 计）')
    return parser.parse_args()


def setup_log_directories(logdir: str, config_path: str) -> Dict[str, str]:
    os.makedirs(logdir, exist_ok=True)
    model_outdir = os.path.join(logdir, 'models')
    recon_outdir = os.path.join(logdir, 'reconstructions')
    meta_outdir = os.path.join(logdir, 'meta')
    os.makedirs(model_outdir, exist_ok=True)
    os.makedirs(recon_outdir, exist_ok=True)
    os.makedirs(meta_outdir, exist_ok=True)

    shutil.copyfile(config_path, os.path.join(meta_outdir, 'config_backup.yaml'))
    shutil.copy(__file__, os.path.join(meta_outdir, 'train_script_backup.py'))

    models_src_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models_dst_dir = os.path.join(meta_outdir, 'models_snapshot')
    shutil.copytree(models_src_dir, models_dst_dir, dirs_exist_ok=True)

    return {
        'model': model_outdir,
        'recon': recon_outdir,
        'meta': meta_outdir
    }


def build_pose3d_dataloader(cfg: Dict[str, Any], use_global_norm: bool):
    DatasetClass = RGBPose3DDatasetGlobalNorm if use_global_norm else RGBPose3DDataset
    dataset = DatasetClass(
        images_dir=cfg['DATA']['dataset_path'],
        pose_file=cfg['DATA']['pose_file'],
        copy_to_gpu=False,
        grayscale=cfg['MODEL']['out_dim'] == 1,
        crop_size=cfg['DATA'].get('crop_size', None),
        angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
        mode='train'
    )

    batch_size = cfg['TRAINING'].get('batch_size', 1)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=cfg['DATA'].get('num_workers', 0),
                              drop_last=False)
    return train_loader, dataset


def align_cfg_with_dataset(cfg: Dict[str, Any], dataset):
    sample = dataset[0]
    img_channels = sample['gt_img'].shape[0]
    dino_dim = sample['dino'].numel()
    n_segments = int(sample['segments'].max().item()) + 1

    cfg['MODEL']['out_dim'] = img_channels
    cfg['MODEL']['dino_dim'] = dino_dim
    cfg['DATA']['n_segments'] = max(n_segments, cfg['DATA'].get('n_segments', 1))
    assert cfg['DATA']['n_segments'] <= cfg['MODEL']['n_experts'], \
        "Number of segments must not exceed number of experts"

    cfg['TRAINING']['n_samples'] = cfg['TRAINING']['num_epochs']


def create_optimizer_and_scheduler(cfg, stage_handler):
    lr = cfg['TRAINING']['lr'] if isinstance(cfg['TRAINING']['lr'], float) else cfg['TRAINING']['lr']['all']
    optimizer = optim.Adam(stage_handler.get_trainable_params(), lr=lr, betas=(0.9, 0.999))
    stage_handler.freeze_params()
    scheduler = stage_handler.get_scheduler(optimizer)
    return optimizer, scheduler


def maybe_load_manager_checkpoint(cfg, model, device):
    if cfg['MODEL'].get('load_pt_manager', False):
        path = cfg['MODEL'].get('manager_pt_path', '')
        if path and os.path.exists(path):
            state_dict = torch.load(path, map_location=device)
            manager_state = {k: v for k, v in state_dict.items()
                             if k.startswith('manager_net')
                             or k.startswith('manager_input_encoding_module')
                             or k.startswith('manager_conditioner')}
            model_state = model.state_dict()
            for k, v in manager_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
            model.load_state_dict(model_state)
            print(f"✓ 已加载 Manager 预训练权重: {path}（匹配到的 Manager 参数）")
        else:
            print("⚠ 指定了加载 Manager 但路径不存在，跳过。")


def format_loss_dict(loss_dict):
    msg = []
    for k, v in loss_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if torch.isinf(v) or torch.isnan(v):
            msg.append(f"{k}: nan")
        else:
            msg.append(f"{k}: {v.item():.6f}")
    return " | ".join(msg)


def save_reconstruction(output_pred, gt_img, out_path):
    B, C, H, W = gt_img.shape
    pred = output_pred['selected_nonmanifold_pnts_pred']
    if pred.dim() == 2:
        pred = pred.view(B, H, W, 1)
    elif pred.dim() == 3:
        pred = pred.view(B, H, W, -1)
    else:
        pred = pred.view(B, H, W, C)

    pred_np = torch.clamp(pred[0], 0.0, 1.0).detach().cpu().numpy()
    gt_np = torch.clamp(gt_img[0].permute(1, 2, 0), 0.0, 1.0).detach().cpu().numpy()

    pred_u8 = (pred_np * 255).astype(np.uint8)
    gt_u8 = (gt_np * 255).astype(np.uint8)
    if pred_u8.shape[-1] == 1:
        pred_u8 = pred_u8.squeeze(-1)
        gt_u8 = gt_u8.squeeze(-1)

    combined = np.concatenate([gt_u8, pred_u8], axis=1)
    Image.fromarray(combined).save(out_path)


def train():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    use_global_norm = args.use_global_norm or cfg['DATA'].get('use_global_norm', True)
    train_loader, train_set = build_pose3d_dataloader(cfg, use_global_norm)
    align_cfg_with_dataset(cfg, train_set)

    dirs = setup_log_directories(args.logdir, args.config)
    log_file_path = os.path.join(dirs['meta'], 'training_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    SINR, _ = build_model(cfg, cfg['LOSS'])
    print(f"模型参数数量: {sum(p.numel() for p in SINR.parameters() if p.requires_grad):,}")

    maybe_load_manager_checkpoint(cfg, SINR, device)
    SINR.to(device)

    stage_handler = TrainingStageHandler(cfg['TRAINING']['stages'], SINR, cfg)
    criterion = stage_handler.criterion
    optimizer, scheduler = create_optimizer_and_scheduler(cfg, stage_handler)

    gt_aux_choices = {'img': None, 'dino': None}

    total_epochs = cfg['TRAINING']['num_epochs']
    save_interval = cfg['TRAINING'].get('save_interval', 100)
    vis_interval = args.save_visual_interval
    grad_clip = cfg['TRAINING'].get('grad_clip_norm', 0.0)

    start_time = time.time()
    for epoch in range(total_epochs):
        SINR.train()
        epoch_losses = []

        for batch_idx, data in enumerate(train_loader):
            coords = data['coords'].to(device)
            if coords.dim() == 4:
                coords = coords.squeeze(1)
            B = coords.shape[0]

            gt_img = data['gt_img'].to(device)
            if gt_img.dim() == 4:
                B_img, C, H, W = gt_img.shape
                gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B_img, -1, C)
            else:
                # already flattened (B, N, C)
                B_img, N, C = gt_img.shape
                H = W = int(np.sqrt(N))
                gt_img_flat = gt_img

            if coords.shape[1] != gt_img_flat.shape[1]:
                coords = coords.reshape(B, -1, coords.shape[-1])
                if coords.shape[1] != gt_img_flat.shape[1]:
                    raise ValueError(f"坐标数量 {coords.shape[1]} 与像素数量 {gt_img_flat.shape[1]} 不匹配")

            segments = data['segments'].to(device)
            if segments.dim() == 2:
                segments_flat = segments
            else:
                segments_flat = segments.view(B, -1)

            dino = data['dino'].to(device)
            if dino.dim() >= 3:
                dino_flat = dino.view(B, -1, dino.shape[-1])
            else:
                dino_flat = torch.zeros(B, gt_img_flat.shape[1], 1, device=device, dtype=gt_img_flat.dtype)

            gt_aux_choices['img'] = gt_img_flat
            gt_aux_choices['dino'] = dino_flat
            aux_type = cfg['MODEL'].get('aux_type', 'img')
            gt_aux_value = gt_aux_choices.get(aux_type, None)

            output_pred = SINR(coords, dino=dino_flat, img=gt_img_flat)
            loss_dict = criterion(
                output_pred=output_pred,
                coords=coords,
                gt={'img': gt_img_flat,
                    'segment': segments_flat,
                    'aux': gt_aux_value},
                model=SINR
            )

            optimizer.zero_grad(set_to_none=True)
            loss_dict['loss'].backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(SINR.parameters(), grad_clip)

            optimizer.step()

            epoch_losses.append(loss_dict['loss'].item())

            if (batch_idx % 25 == 0) or (batch_idx == len(train_loader) - 1):
                msg = (f"[Epoch {epoch+1}/{total_epochs}] Batch {batch_idx+1}/{len(train_loader)} "
                       f"Loss: {loss_dict['loss'].item():.6f} "
                       f"Recon: {loss_dict.get('rgbrecon_term', torch.tensor(0.)).item():.6f} "
                       f"Balance: {loss_dict.get('balance_term', torch.tensor(0.)).item():.6f}")
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

            if epoch % save_interval == 0 and batch_idx == 0:
                model_path = os.path.join(dirs['model'], f"{cfg['MODEL']['model_name']}_epoch_{epoch}.pth")
                torch.save(SINR.state_dict(), model_path)

        scheduler.step()

        if epoch % vis_interval == 0:
            SINR.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                coords = sample_batch['coords'].to(device)
                if coords.dim() == 4:
                    coords = coords.squeeze(1)
                gt_img = sample_batch['gt_img'].to(device)
                B_img, C, H, W = gt_img.shape
                gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B_img, -1, C)
                dino = sample_batch['dino'].to(device)
                if dino.dim() >= 3:
                    dino_flat = dino.view(B_img, -1, dino.shape[-1])
                else:
                    dino_flat = torch.zeros(B_img, gt_img_flat.shape[1], 1, device=device,
                                             dtype=gt_img_flat.dtype)
                output_pred = SINR(coords, dino=dino_flat, img=gt_img_flat)
                recon_path = os.path.join(dirs['recon'], f"epoch_{epoch:05d}.png")
                save_reconstruction(output_pred, gt_img, recon_path)
                print(f"✓ 已保存可视化重建: {recon_path}")
            SINR.train()

        if epoch > stage_handler.get_end_iteration():
            print(">>> 进入下一训练阶段, 解冻/冻结参数并更新损失...")
            stage_handler.move_to_the_next_training_stage(optimizer, scheduler)
            criterion = stage_handler.criterion

    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time/60:.1f} 分钟")
    log_file.write(f"Training finished in {total_time/60:.1f} minutes\n")
    log_file.close()


if __name__ == '__main__':
    train()

