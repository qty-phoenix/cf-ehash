#!/usr/bin/env python3
"""
Step 4: Evaluate Plan A — compare GT tracker poses vs NR-Rec-FUS predicted poses,
and assess image reconstruction quality using neural-ex INR.

This script:
1. Loads GT poses and predicted poses
2. Computes pose prediction errors (LPE, GPE in mm; LLE, GLE in pixels)
3. Trains neural-ex INR with GT poses and predicted poses separately
4. Compares reconstruction quality (PSNR, SSIM, MAE)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Add neural-ex to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import build_model
from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm


def compute_pose_errors(gt_excel, pred_excel, images_dir,
                        H=640, W=480, use_calib=True):
    """
    Compute pose prediction errors.

    Returns dict with:
      - LPE (Local Position Error): mean point distance for adjacent frames (mm)
      - GPE (Global Position Error): mean point distance to frame 0 (mm)
      - LLE (Local Landmark Error): mean corner pixel error for adjacent frames (px)
      - GLE (Global Landmark Error): mean corner pixel error to frame 0 (px)
    """
    df_gt = pd.read_excel(gt_excel)
    df_pred = pd.read_excel(pred_excel)

    assert len(df_gt) == len(df_pred), \
        f"Frame count mismatch: GT={len(df_gt)}, Pred={len(df_pred)}"

    n_frames = len(df_gt)

    def euler_to_matrix(angles_deg, translation):
        rx, ry, rz = np.deg2rad(angles_deg)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T

    # Build transforms from poses
    T_gt = np.zeros((n_frames, 4, 4))
    T_pred = np.zeros((n_frames, 4, 4))

    for i in range(n_frames):
        row_gt = df_gt.iloc[i]
        row_pred = df_pred.iloc[i]

        trans_gt = row_gt[['Position X', 'Position Y', 'Position Z']].values.astype(float)
        rot_gt = row_gt[['Orientation X', 'Orientation Y', 'Orientation Z']].values.astype(float)
        T_gt[i] = euler_to_matrix(rot_gt, trans_gt)

        trans_pred = row_pred[['Position X', 'Position Y', 'Position Z']].values.astype(float)
        rot_pred = row_pred[['Orientation X', 'Orientation Y', 'Orientation Z']].values.astype(float)
        T_pred[i] = euler_to_matrix(rot_pred, trans_pred)

    # Compute calibration matrices (uvfdata2)
    S = np.array([
        [0.229389190673828, 0, 0, 0],
        [0, 0.220979690551758, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    C = np.array([
        [0.231064671309448, -0.218052035, 0.948189025293473, -70.74132919],
        [-0.190847036, -0.965787273, -0.175591436, -80.6505661],
        [0.954036962825936, -0.140386088, -0.264773903, -46.17662239],
        [0, 0, 0, 1],
    ])

    # Create corner points (4 corners of image)
    corners_img = np.array([
        [0, 0, 0, 1],
        [W - 1, 0, 0, 1],
        [0, H - 1, 0, 1],
        [W - 1, H - 1, 0, 1],
    ]).T  # (4, 4)

    # Transform corners to tool coordinates (via S and C)
    corners_tool = (C @ (S @ corners_img).T).T[:, :3]  # (4, 3)

    # Compute LPE/LLE: adjacent frame errors
    local_pos_errors = []
    local_lmk_errors = []
    for i in range(1, n_frames):
        # GT relative transform: frame i -> frame i-1
        T_gt_rel = np.linalg.inv(T_gt[i - 1]) @ T_gt[i]
        T_pred_rel = np.linalg.inv(T_pred[i - 1]) @ T_pred[i]

        # Apply to corners
        pts_gt = (T_gt_rel[:3, :3] @ corners_tool.T + T_gt_rel[:3, 3:4]).T  # (4, 3)
        pts_pred = (T_pred_rel[:3, :3] @ corners_tool.T + T_pred_rel[:3, 3:4]).T

        local_pos_errors.append(np.mean(np.sqrt(np.sum((pts_pred - pts_gt) ** 2, axis=1))))
        local_lmk_errors.append(local_pos_errors[-1] / 0.23)  # Approx mm->px

    # Compute GPE/GLE: to frame 0 errors
    global_pos_errors = []
    global_lmk_errors = []
    for i in range(1, n_frames):
        T_gt_rel = T_gt[i]  # frame i relative to frame 0
        T_pred_rel = T_pred[i]

        pts_gt = (T_gt_rel[:3, :3] @ corners_tool.T + T_gt_rel[:3, 3:4]).T
        pts_pred = (T_pred_rel[:3, :3] @ corners_tool.T + T_pred_rel[:3, 3:4]).T

        global_pos_errors.append(np.mean(np.sqrt(np.sum((pts_pred - pts_gt) ** 2, axis=1))))
        global_lmk_errors.append(global_pos_errors[-1] / 0.23)

    results = {
        'LPE_mm': float(np.mean(local_pos_errors)),
        'LLE_px': float(np.mean(local_lmk_errors)),
        'GPE_mm': float(np.mean(global_pos_errors)),
        'GLE_px': float(np.mean(global_lmk_errors)),
        'LPE_std_mm': float(np.std(local_pos_errors)),
        'GPE_std_mm': float(np.std(global_pos_errors)),
    }
    return results


def train_and_evaluate_inr(config_path, pose_file, logdir, gpu, label, num_epochs=100,
                           crop_size=None):
    """
    Train neural-ex INR with given pose file and evaluate reconstruction quality.
    Returns dict of (PSNR, SSIM, MAE, MSE) on test set.
    """
    print(f"\n{'='*60}")
    print(f"Training INR with {label} poses: {pose_file}")
    if crop_size:
        print(f"  Crop size: {crop_size} (for speed)")
    print(f"{'='*60}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    # Override pose file and crop size
    cfg['DATA']['pose_file'] = pose_file
    if crop_size is not None:
        cfg['DATA']['crop_size'] = crop_size

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Setup output dirs
    model_outdir = os.path.join(logdir, 'models')
    vis_outdir = os.path.join(logdir, 'visualizations')
    os.makedirs(model_outdir, exist_ok=True)
    os.makedirs(vis_outdir, exist_ok=True)

    # Data — use separate cache dir per pose_file to avoid cross-contamination
    is_grayscale = cfg['MODEL']['out_dim'] == 1
    use_global_norm = cfg['DATA'].get('use_global_norm', True)
    use_uvfdata2_calibration = cfg['DATA'].get('use_uvfdata2_calibration', False)
    crop = cfg['DATA'].get('crop_size', None)
    import hashlib
    pose_hash = hashlib.md5(pose_file.encode()).hexdigest()[:8]
    cache_dir = os.path.join(cfg['DATA']['dataset_path'], '.cache', f'pose_{pose_hash}')

    DatasetClass = RGBPose3DDatasetGlobalNorm
    train_set = DatasetClass(
        cfg['DATA']['dataset_path'], pose_file,
        copy_to_gpu=False, grayscale=is_grayscale,
        crop_size=crop,
        angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
        mode='train',
        use_uvfdata2_calibration=use_uvfdata2_calibration,
        cache_dir=cache_dir
    )
    test_set = DatasetClass(
        cfg['DATA']['dataset_path'], pose_file,
        copy_to_gpu=False, grayscale=is_grayscale,
        crop_size=crop,
        angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
        mode='test',
        use_uvfdata2_calibration=use_uvfdata2_calibration,
        cache_dir=cache_dir
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

    # Model
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_params = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Loss weights
    import re
    loss_type = cfg['LOSS']['loss_type']
    recon_match = re.match(r'(\d+(?:\.\d+)?)rgbrecon', loss_type)
    recon_weight = float(recon_match.group(1)) if recon_match else 100.0
    balance_match = re.search(r'\+(\d+(?:\.\d+)?)balance', loss_type)
    balance_weight = float(balance_match.group(1)) if balance_match else 0.1

    lr = cfg['TRAINING']['lr']
    if isinstance(lr, dict):
        lr = float(lr.get('all', 1e-4))
    else:
        lr = float(lr)

    optimizer = torch.optim.Adam(SINR.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=cfg['TRAINING']['lr_gamma'])
    SINR.to(device)

    # Training loop
    for epoch in range(num_epochs):
        SINR.train()
        epoch_losses = []

        for data in train_loader:
            coords = data['coords'].to(device).squeeze(1)
            gt_img = data['gt_img'].to(device)
            dino = data['dino'].to(device)

            B, C, H_img, W_img = gt_img.shape
            gt_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H_img * W_img, C)

            output_pred = SINR(coords, dino=dino, img=gt_img)

            q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
            if q is None:
                n_experts = cfg['MODEL']['n_experts']
                q = torch.ones(B, n_experts, H_img * W_img, device=device) / n_experts

            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is None:
                continue

            if all_expert_preds.dim() == 4:
                all_expert_preds = all_expert_preds.squeeze(-1)

            # Weighted average prediction
            weighted_pred = torch.sum(q * all_expert_preds, dim=1)

            recon_loss = torch.mean((weighted_pred - gt_flat.squeeze(-1)) ** 2) * recon_weight

            expert_usage = q.mean(dim=2)
            balance_loss = torch.var(expert_usage, dim=1).mean()
            loss = recon_loss + balance_weight * balance_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(SINR.parameters(),
                                           cfg['TRAINING']['grad_clip_norm'])
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()

    # ---- Evaluation on test set ----
    SINR.eval()
    all_psnr, all_ssim, all_mae, all_mse = [], [], [], []

    with torch.no_grad():
        for data in test_loader:
            coords = data['coords'].to(device).squeeze(1)
            gt_img = data['gt_img'].to(device)
            dino = data['dino'].to(device)

            B, C, H_img, W_img = gt_img.shape
            gt_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H_img * W_img, C)

            output_pred = SINR(coords, dino=dino, img=gt_img)

            q = output_pred.get('nonmnfld_q', output_pred.get('mnfld_q', None))
            if q is None:
                n_experts = cfg['MODEL']['n_experts']
                q = torch.ones(B, n_experts, H_img * W_img, device=device) / n_experts

            all_expert_preds = output_pred.get('nonmanifold_pnts_pred', None)
            if all_expert_preds is None:
                continue
            if all_expert_preds.dim() == 4:
                all_expert_preds = all_expert_preds.squeeze(-1)

            weighted_pred = torch.sum(q * all_expert_preds, dim=1)
            weighted_pred = torch.clamp(weighted_pred, 0.0, 1.0)

            gt_1d = gt_flat.squeeze(-1) if C == 1 else gt_flat[:, :, 0]

            if weighted_pred.dim() == 2 and weighted_pred.shape[0] == 1:
                weighted_pred = weighted_pred.squeeze(0)
            if gt_1d.dim() == 2 and gt_1d.shape[0] == 1:
                gt_1d = gt_1d.squeeze(0)

            mse = torch.mean((weighted_pred - gt_1d) ** 2).item()
            psnr_val = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            mae_val = torch.mean(torch.abs(weighted_pred - gt_1d)).item()

            pred_2d = weighted_pred.reshape(H_img, W_img).cpu().numpy()
            gt_2d = gt_1d.reshape(H_img, W_img).cpu().numpy()
            ssim_val = ssim(gt_2d, pred_2d, data_range=1.0)

            all_mse.append(mse)
            all_psnr.append(psnr_val)
            all_mae.append(mae_val)
            all_ssim.append(ssim_val)

    results = {
        'PSNR': float(np.mean(all_psnr)),
        'SSIM': float(np.mean(all_ssim)),
        'MAE': float(np.mean(all_mae)),
        'MSE': float(np.mean(all_mse)),
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Plan A: GT vs Predicted poses for INR reconstruction')
    parser.add_argument('--gt_pose', type=str,
                        default='./MyMoE/coords3d/uvfdata2.xlsx')
    parser.add_argument('--pred_pose', type=str,
                        default='./bridge_nr_rec_fus/data/predicted_poses_uvfdata2.xlsx')
    parser.add_argument('--config', type=str,
                        default='./configs/config_RGB_pose3d_uvfdata2.yaml')
    parser.add_argument('--images_dir', type=str,
                        default='./MyMoE/uvfdata2')
    parser.add_argument('--logdir_gt', type=str,
                        default='./bridge_nr_rec_fus/results/inr_gt_poses')
    parser.add_argument('--logdir_pred', type=str,
                        default='./bridge_nr_rec_fus/results/inr_pred_poses')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Crop images to this size for faster INR training')
    parser.add_argument('--skip_inr', action='store_true',
                        help='Skip INR training, only compute pose errors')
    args = parser.parse_args()

    print("=" * 70)
    print("Plan A Evaluation: NR-Rec-FUS Pose Estimator -> neural-ex INR")
    print("=" * 70)

    # Step 1: Compute pose errors
    print("\n--- Pose Prediction Errors ---")
    pose_results = compute_pose_errors(
        args.gt_pose, args.pred_pose, args.images_dir)
    print(f"  LPE (Local Position Error):    {pose_results['LPE_mm']:.3f} ± {pose_results['LPE_std_mm']:.3f} mm")
    print(f"  LLE (Local Landmark Error):    {pose_results['LLE_px']:.3f} px")
    print(f"  GPE (Global Position Error):   {pose_results['GPE_mm']:.3f} ± {pose_results['GPE_std_mm']:.3f} mm")
    print(f"  GLE (Global Landmark Error):   {pose_results['GLE_px']:.3f} px")

    if args.skip_inr:
        return

    # Step 2: Train INR with GT poses
    print("\n--- INR with GT (Tracker) Poses ---")
    gt_inr_results = train_and_evaluate_inr(
        args.config, args.gt_pose, args.logdir_gt, args.gpu,
        "GT", args.num_epochs, crop_size=args.crop_size)

    # Step 3: Train INR with predicted poses
    print("\n--- INR with NR-Rec-FUS Predicted Poses ---")
    pred_inr_results = train_and_evaluate_inr(
        args.config, args.pred_pose, args.logdir_pred, args.gpu,
        "Predicted", args.num_epochs, crop_size=args.crop_size)

    # Step 4: Compare
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<12} {'GT Tracker':>14} {'NR-Rec-FUS':>14} {'Delta':>14}")
    print("-" * 56)
    for metric in ['PSNR', 'SSIM', 'MAE', 'MSE']:
        gt_val = gt_inr_results[metric]
        pred_val = pred_inr_results[metric]
        delta = pred_val - gt_val
        direction = '↓' if metric in ['MAE', 'MSE'] else '↑'
        print(f"{metric:<12} {gt_val:>14.4f} {pred_val:>14.4f} {delta:>+13.4f} {direction}")

    print(f"\n{'Pose Error':<12} {'Value':>14}")
    print("-" * 28)
    for k, v in pose_results.items():
        if not k.endswith('_std'):
            print(f"{k:<12} {v:>14.3f}")

    # Save results
    os.makedirs(os.path.dirname(args.logdir_pred), exist_ok=True)
    results = {
        'pose_errors': pose_results,
        'inr_gt': gt_inr_results,
        'inr_pred': pred_inr_results,
    }
    import json
    results_path = os.path.join(os.path.dirname(args.logdir_pred), 'plan_a_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
