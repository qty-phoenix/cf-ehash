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

from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets.RGBPose3D import RGBPose3DDataset


def fix_data_shapes(data, device):
    """修复数据形状问题"""
    coords = data['coords'].to(device)
    gt_img = data['gt_img'].to(device)
    segments = data['segments'].to(device)
    dino = data['dino'].to(device)
    
    # 修复坐标形状
    if coords.dim() == 4 and coords.shape[1] == 1:
        coords = coords.squeeze(1)  # (1, 1, H*W, 3) -> (1, H*W, 3)
        print(f"  🔧 修复坐标形状: {coords.shape}")
    
    # 修复segments形状
    if segments.dim() == 2 and segments.shape[0] == 1:
        segments = segments.squeeze(0)  # (1, H*W) -> (H*W,)
        print(f"  🔧 修复segments形状: {segments.shape}")
    
    # 修复dino形状
    if dino.dim() == 2 and dino.shape[0] == 1:
        dino = dino.squeeze(0)  # (1, 1) -> (1,)
        print(f"  🔧 修复dino形状: {dino.shape}")
    
    return coords, gt_img, segments, dino


def calculate_metrics_improved(pred, gt):
    """改进的指标计算"""
    with torch.no_grad():
        # 确保pred和gt的形状一致
        if pred.shape != gt.shape:
            if pred.dim() == 4 and gt.dim() == 4:
                B, C, H, W = gt.shape
                if pred.shape[2] == H * W:
                    pred = pred.squeeze(0).squeeze(-1).reshape(H, W).unsqueeze(0).unsqueeze(0)
                elif pred.shape[1] == H * W:
                    pred = pred.squeeze(0).reshape(H, W).unsqueeze(0).unsqueeze(0)
        
        # 计算MSE (在[-1,1]范围内)
        mse = torch.mean((pred - gt) ** 2)
        
        # 计算PSNR (假设数据范围在[-1,1]，最大值为2)
        if mse > 0:
            psnr = 10 * torch.log10(4.0 / mse)  # 4 = (1-(-1))^2
        else:
            psnr = torch.tensor(float('inf'))
        
        # 计算MAE
        mae = torch.mean(torch.abs(pred - gt))
        
        return {
            'mse': mse.item(),
            'psnr': psnr.item(),
            'mae': mae.item()
        }


def save_training_visualization(model, dataloader, device, output_dir, epoch, num_samples=3):
    """保存训练可视化"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_samples:
                break
                
            # 修复数据形状
            coords, gt_img, segments, dino = fix_data_shapes(data, device)
            
            # 前向传播
            output_pred = model(coords, dino=dino, img=gt_img)
            
            # 获取预测结果
            if 'selected_nonmanifold_pnts_pred' in output_pred:
                pred = output_pred['selected_nonmanifold_pnts_pred']
            else:
                pred = output_pred['nonmanifold_pnts_pred']
                if pred.dim() == 3 and pred.shape[0] != gt_img.shape[0]:
                    pred = pred[0]
            
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
            
            # 转换到[0,1]范围用于可视化
            pred_vis = (pred_np + 1) / 2
            gt_vis = (gt_np + 1) / 2
            
            # 确保是灰度图像
            if len(pred_vis.shape) == 3 and pred_vis.shape[2] == 1:
                pred_vis = pred_vis.squeeze(2)
            if len(gt_vis.shape) == 3 and gt_vis.shape[2] == 1:
                gt_vis = gt_vis.squeeze(2)
            
            # 创建对比图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 第一行：原图、预测图、误差图
            axes[0, 0].imshow(gt_vis, cmap='gray')
            axes[0, 0].set_title('Ground Truth')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(pred_vis, cmap='gray')
            axes[0, 1].set_title('Prediction')
            axes[0, 1].axis('off')
            
            error = np.abs(pred_vis - gt_vis)
            im = axes[0, 2].imshow(error, cmap='hot')
            axes[0, 2].set_title('Absolute Error')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2])
            
            # 第二行：坐标分布和专家权重
            coords_np = coords.detach().cpu().numpy()
            axes[1, 0].scatter(coords_np[:, 0], coords_np[:, 1], c=coords_np[:, 2], cmap='viridis', s=1)
            axes[1, 0].set_title('3D Coordinates (XY view)')
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Y')
            
            axes[1, 1].scatter(coords_np[:, 0], coords_np[:, 2], c=coords_np[:, 1], cmap='viridis', s=1)
            axes[1, 1].set_title('3D Coordinates (XZ view)')
            axes[1, 1].set_xlabel('X')
            axes[1, 1].set_ylabel('Z')
            
            # 专家权重可视化
            if 'nonmnfld_q' in output_pred:
                q = output_pred['nonmnfld_q'].squeeze(0).detach().cpu().numpy()  # (H*W, n_experts)
                q_reshaped = q.reshape(H, W, -1)
                expert_usage = q_reshaped.mean(axis=(0, 1))
                axes[1, 2].bar(range(len(expert_usage)), expert_usage)
                axes[1, 2].set_title('Expert Usage')
                axes[1, 2].set_xlabel('Expert Index')
                axes[1, 2].set_ylabel('Average Weight')
            else:
                axes[1, 2].text(0.5, 0.5, 'No Expert Info', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Expert Usage')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'training_epoch_{epoch:06d}_sample_{i:04d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    model.train()


def main(args):
    cfg = yaml.safe_load(open(args.config))
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # 创建输出目录
    os.makedirs(args.logdir, exist_ok=True)
    model_outdir = os.path.join(args.logdir, 'trained_models')
    vis_outdir = os.path.join(args.logdir, 'training_vis')
    os.makedirs(model_outdir, exist_ok=True)
    os.makedirs(vis_outdir, exist_ok=True)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据集和数据加载器
    images_dir = cfg['DATA']['dataset_path']
    pose_file = cfg['DATA'].get('pose_file', cfg['DATA'].get('coords3d_path', ''))
    is_grayscale = (cfg['MODEL']['out_dim'] == 1)
    
    print(f"加载数据集...")
    print(f"图像目录: {images_dir}")
    print(f"姿态文件: {pose_file}")
    
    train_set = RGBPose3DDataset(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                                 crop_size=cfg['DATA'].get('crop_size', None),
                                 angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                                 mode='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    
    print(f"训练集大小: {len(train_set)}")

    # 模型和损失函数
    cfg['MODEL']['out_dim'] = cfg['MODEL']['out_dim']
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"模型参数数量: {n_parameters:,}")

    # 训练阶段处理器
    cfg['TRAINING']['n_samples'] = cfg['TRAINING']['num_epochs']
    training_stage_handler = TrainingStageHandler(cfg['TRAINING']['stages'], SINR, cfg)
    criterion = training_stage_handler.criterion

    # 优化器和调度器
    lr = cfg['TRAINING']['lr'] if isinstance(cfg['TRAINING']['lr'], float) else cfg['TRAINING']['lr']['all']
    optimizer = optim.Adam(training_stage_handler.get_trainable_params(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    training_stage_handler.freeze_params()
    scheduler = training_stage_handler.get_scheduler(optimizer)

    SINR.to(device)
    
    # 训练统计
    total_epochs = cfg['TRAINING']['num_epochs']
    save_interval = 100
    vis_interval = 200
    
    print(f"\n开始训练...")
    print(f"总epoch数: {total_epochs}")
    print(f"学习率: {lr}")
    print(f"模型保存间隔: {save_interval}")
    print(f"可视化间隔: {vis_interval}")
    print("=" * 80)

    start_time = time.time()
    best_psnr = 0
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        SINR.train()
        
        epoch_losses = []
        epoch_metrics = {'mse': [], 'psnr': [], 'mae': []}
        
        for batch_idx, data in enumerate(train_loader):
            # 修复数据形状
            coords, gt_img, segments, dino = fix_data_shapes(data, device)
            
            coords.requires_grad_()

            output_pred = SINR(coords, dino=dino, img=gt_img)
            B, C, H, W = gt_img.shape
            gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
            loss_dict = criterion(output_pred=output_pred,
                                  coords=coords,
                                  gt={'img': gt_img_flat, 'segment': segments, 'aux': gt_img_flat},
                                  model=SINR)

            optimizer.zero_grad(set_to_none=True)
            loss_dict['loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(SINR.parameters(), max_norm=cfg['TRAINING'].get('grad_clip_norm', 1.0))
            
            optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss_dict['loss'].item())
            
            # 计算指标
            if batch_idx % 10 == 0:
                metrics = calculate_metrics_improved(output_pred.get('selected_nonmanifold_pnts_pred', 
                                                          output_pred['nonmanifold_pnts_pred']), 
                                          gt_img_flat)
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])

        # 计算epoch统计
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_metrics['mse']) if epoch_metrics['mse'] else 0
        avg_psnr = np.mean(epoch_metrics['psnr']) if epoch_metrics['psnr'] else 0
        avg_mae = np.mean(epoch_metrics['mae']) if epoch_metrics['mae'] else 0
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        
        # 打印进度信息
        if epoch % 10 == 0 or epoch == total_epochs - 1:
            progress = (epoch + 1) / total_epochs * 100
            eta = elapsed_time / (epoch + 1) * (total_epochs - epoch - 1)
            
            print(f"Epoch {epoch+1:4d}/{total_epochs} | "
                  f"Loss: {avg_loss:.6f} | "
                  f"MSE: {avg_mse:.6f} | "
                  f"PSNR: {avg_psnr:.2f}dB | "
                  f"MAE: {avg_mae:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Progress: {progress:.1f}%")

        # 保存模型
        if epoch % save_interval == 0 or epoch == total_epochs - 1:
            model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_fixed_model_{epoch}.pth')
            torch.save(SINR.state_dict(), model_path)
            
            print(f"✓ 保存模型: {model_path}")
            print(f"  当前指标 - Loss: {avg_loss:.6f}, MSE: {avg_mse:.6f}, PSNR: {avg_psnr:.2f}dB, MAE: {avg_mae:.6f}")
            
            # 保存最佳模型
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_model_path = os.path.join(model_outdir, f'{cfg["MODEL"]["model_name"]}_best.pth')
                torch.save(SINR.state_dict(), best_model_path)
                print(f"✓ 保存最佳模型: {best_model_path} (PSNR: {best_psnr:.2f}dB)")
            
            # 保存训练可视化
            if epoch % vis_interval == 0 or epoch == total_epochs - 1:
                print(f"  生成训练可视化...")
                save_training_visualization(SINR, train_loader, device, vis_outdir, epoch, num_samples=3)
                print(f"  可视化保存到: {vis_outdir}")

        scheduler.step()
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"训练完成!")
    print(f"总训练时间: {total_time/60:.1f}分钟")
    print(f"最佳PSNR: {best_psnr:.2f}dB")
    print(f"最终模型保存到: {model_outdir}")
    print(f"训练可视化保存到: {vis_outdir}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Fixed Training RGBPose3D with MoE-INR')
    parser.add_argument('--config', default='../configs/config_RGB_pose3d.yaml', type=str)
    parser.add_argument('--logdir', default='./log/pose3d_fixed', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
