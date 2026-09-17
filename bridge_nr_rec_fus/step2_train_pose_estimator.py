#!/usr/bin/env python3
"""
Step 2: Train NR-Rec-FUS EfficientNet pose estimator on uvfdata2 data.

This is a simplified training script that trains only the EfficientNet
to predict relative frame-to-frame transformations from US image sequences.
The VoxelMorph registration network is not used here (Plan A only needs poses).
"""

import os
import sys
# Prevent matplotlib from loading (avoids GLIBCXX version issues)
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add NR-Rec-FUS to path
_nr_path = os.path.join(os.path.dirname(__file__), '..', '..', 'NR-Rec-FUS')
sys.path.insert(0, _nr_path)

from utils.loader import SSFrameDataset
from utils.network import build_model
from data.calib import read_calib_matrices
from utils.transform import LabelTransform, PredictionTransform


def reference_image_points(image_H, image_W, density=2):
    """Create reference image corner points in homogeneous coordinates."""
    if isinstance(density, int):
        density = (density, density)
    image_points = torch.cartesian_prod(
        torch.linspace(0, image_W - 1, density[0]),
        torch.linspace(0, image_H - 1, density[1])
    ).t()
    image_points = torch.cat([
        image_points,
        torch.zeros(1, image_points.shape[1]),
        torch.ones(1, image_points.shape[1])
    ], axis=0)
    return image_points


def compute_dimension(label_pred_type, num_points_each_frame=None, num_frames=None, type_option=None):
    if type_option == 'pred':
        num_frames = num_frames - 1
    type_dim_dict = {
        "transform": 12 * num_frames,
        "parameter": 6 * num_frames,
        "point": 3 * 4 * num_frames,
        "quaternion": 7 * num_frames
    }
    return type_dim_dict[label_pred_type]


def param_to_transform_static(params):
    """Convert 6-DoF params (rx,ry,rz,tx,ty,tz, ZYX) to 4x4 matrices."""
    cos_x = torch.cos(params[..., 2])
    sin_x = torch.sin(params[..., 2])
    cos_y = torch.cos(params[..., 1])
    sin_y = torch.sin(params[..., 1])
    cos_z = torch.cos(params[..., 0])
    sin_z = torch.sin(params[..., 0])
    return torch.cat((
        torch.stack([cos_y * cos_z, sin_x * sin_y * cos_z - cos_x * sin_z,
                     cos_x * sin_y * cos_z + sin_x * sin_z, params[..., 3]], dim=-1)[..., None, :],
        torch.stack([cos_y * sin_z, sin_x * sin_y * sin_z + cos_x * cos_z,
                     cos_x * sin_y * sin_z - sin_x * cos_z, params[..., 4]], dim=-1)[..., None, :],
        torch.stack([-sin_y, sin_x * cos_y, cos_x * cos_y, params[..., 5]], dim=-1)[..., None, :],
        torch.cat((torch.zeros_like(params[..., 0:3])[..., None, :],
                   torch.ones_like(params[..., 0])[..., None, None]), dim=-1)
    ), dim=-2)


def data_pairs_adjacent(num_frames):
    """Create pairs: each frame relative to frame 0."""
    return torch.tensor([[0, n0] for n0 in range(num_frames)])


class PointDistance:
    def __init__(self, paired=True):
        self.paired = paired

    def __call__(self, preds, labels):
        if self.paired:
            return ((preds - labels) ** 2).sum(dim=2).sqrt().mean(dim=(0, 2))
        else:
            return ((preds - labels) ** 2).sum(dim=2).sqrt().mean()


def parse_args():
    parser = argparse.ArgumentParser(description='Train NR-Rec-FUS pose estimator on uvfdata2')
    parser.add_argument('--data_dir', type=str, default='./bridge_nr_rec_fus/data')
    parser.add_argument('--h5_file', type=str, default='uvfdata2_res4.h5')
    parser.add_argument('--calib_file', type=str,
                        default='./bridge_nr_rec_fus/data/uvfdata2_calib.csv')
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='./bridge_nr_rec_fus/models')
    parser.add_argument('--pred_type', type=str, default='parameter')
    parser.add_argument('--label_type', type=str, default='transform')
    parser.add_argument('--model_name', type=str, default='efficientnet_b1')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load data ----
    h5_path = os.path.join(args.data_dir, args.h5_file)
    data_path = args.data_dir

    fold_files = {
        'train': [f'fold_0{i}_seqlen{args.num_samples}_scan_uvfdata2.json' for i in range(3)],
        'val': [f'fold_03_seqlen{args.num_samples}_scan_uvfdata2.json'],
        'test': [f'fold_04_seqlen{args.num_samples}_scan_uvfdata2.json'],
    }

    dset_train_list = [
        SSFrameDataset.read_json(data_path, f, args.h5_file)
        for f in fold_files['train']
    ]
    dset_train = dset_train_list[0]
    for ds in dset_train_list[1:]:
        dset_train = dset_train + ds

    dset_val = SSFrameDataset.read_json(data_path, fold_files['val'][0], args.h5_file)
    print(f"Train scans: {len(dset_train)}, Val scans: {len(dset_val)}")

    train_loader = torch.utils.data.DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        dset_val, batch_size=1, shuffle=False, num_workers=0)

    # ---- Setup transforms ----
    data_pairs = data_pairs_adjacent(args.num_samples)

    sample_data = dset_train[0]
    frame_H, frame_W = sample_data[0].shape[1], sample_data[0].shape[2]

    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        filename_calib=args.calib_file,
        resample_factor=4,
        device=device
    )

    image_points = reference_image_points(frame_H, frame_W, density=2).to(device)
    pred_dim = compute_dimension(args.pred_type, image_points.shape[1], args.num_samples, 'pred')
    label_dim = compute_dimension(args.label_type, image_points.shape[1], args.num_samples, 'label')

    transform_label = LabelTransform(
        args.label_type,
        pairs=data_pairs,
        image_points=image_points,
        in_image_coords=True,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T
    )

    transform_prediction = PredictionTransform(
        args.pred_type,
        "transform",
        num_pairs=data_pairs.shape[0] - 1,
        image_points=image_points,
        in_image_coords=True,
        tform_image_to_tool=tform_calib,
        tform_image_mm_to_tool=tform_calib_R_T
    )

    criterion = nn.MSELoss()
    metrics = PointDistance()

    # ---- Build model ----
    model = build_model(
        type('Opt', (), {
            'model_name': args.model_name,
        })(),
        in_frames=args.num_samples,
        pred_dim=pred_dim,
        label_dim=label_dim,
        image_points=image_points,
        tform_calib=tform_calib,
        tform_calib_R_T=tform_calib_R_T
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_name}, Trainable params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # ---- Training loop ----
    best_val_dist = float('inf')
    best_model_path = os.path.join(args.save_dir, 'best_pose_estimator.pth')

    for epoch in range(args.num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_dist_sum = 0.0

        for step, (frames, tforms, tforms_inv) in enumerate(train_loader):
            frames = frames.to(device).float() / 255.0
            tforms = tforms.to(device)
            tforms_inv = tforms_inv.to(device)

            # Build GT labels (point positions in image coords of frame 0)
            tforms_each_to_0 = transform_label(tforms, tforms_inv)
            if args.label_type == 'transform':
                labels = tforms_each_to_0
            else:
                labels = torch.matmul(
                    torch.linalg.inv(tform_calib_R_T),
                    torch.matmul(tforms_each_to_0, torch.matmul(tform_calib, image_points))
                )[:, :, 0:3, ...]

            optimizer.zero_grad()
            outputs = model(frames)

            # Predicted transforms (NUM_SAMPLES-1 relative transforms)
            pred_transfs = transform_prediction(outputs)
            predframe0 = torch.eye(4, 4, device=device)[None, None, ...].repeat(
                pred_transfs.shape[0], 1, 1, 1)
            pred_transfs = torch.cat([predframe0, pred_transfs], dim=1)

            # Compute predicted points
            pred_pts = torch.matmul(
                torch.linalg.inv(tform_calib_R_T),
                torch.matmul(pred_transfs, torch.matmul(tform_calib, image_points))
            )[:, :, 0:3, ...]
            gt_pts = torch.matmul(
                torch.linalg.inv(tform_calib_R_T),
                torch.matmul(tforms_each_to_0, torch.matmul(tform_calib, image_points))
            )[:, :, 0:3, ...]

            loss = criterion(pred_pts, gt_pts)
            dist = metrics(pred_pts, gt_pts)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_dist_sum += dist.mean().item()

        train_loss_avg = train_loss_sum / (step + 1)
        train_dist_avg = train_dist_sum / (step + 1)

        scheduler.step()

        # ---- Validation ----
        if epoch % 5 == 0 or epoch == args.num_epochs - 1:
            model.eval()
            val_loss_sum = 0.0
            val_dist_sum = 0.0

            with torch.no_grad():
                for step, (fr_val, tf_val, tf_val_inv) in enumerate(val_loader):
                    fr_val = fr_val.to(device).float() / 255.0
                    tf_val = tf_val.to(device)
                    tf_val_inv = tf_val_inv.to(device)

                    tforms_each_to_0_val = transform_label(tf_val, tf_val_inv)
                    out_val = model(fr_val)
                    pr_transfs_val = transform_prediction(out_val)
                    predframe0_val = torch.eye(4, 4, device=device)[None, None, ...].repeat(
                        pr_transfs_val.shape[0], 1, 1, 1)
                    pr_transfs_val = torch.cat([predframe0_val, pr_transfs_val], dim=1)

                    pred_pts_val = torch.matmul(
                        torch.linalg.inv(tform_calib_R_T),
                        torch.matmul(pr_transfs_val, torch.matmul(tform_calib, image_points))
                    )[:, :, 0:3, ...]
                    gt_pts_val = torch.matmul(
                        torch.linalg.inv(tform_calib_R_T),
                        torch.matmul(tforms_each_to_0_val, torch.matmul(tform_calib, image_points))
                    )[:, :, 0:3, ...]

                    loss_val = criterion(pred_pts_val, gt_pts_val)
                    dist_val = metrics(pred_pts_val, gt_pts_val)

                    val_loss_sum += loss_val.item()
                    val_dist_sum += dist_val.mean().item()

            val_loss_avg = val_loss_sum / (step + 1)
            val_dist_avg = val_dist_sum / (step + 1)

            print(f"Epoch {epoch:4d} | Train Loss: {train_loss_avg:.4f} Dist: {train_dist_avg:.3f} | "
                  f"Val Loss: {val_loss_avg:.4f} Dist: {val_dist_avg:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            if val_dist_avg < best_val_dist:
                best_val_dist = val_dist_avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dist': val_dist_avg,
                    'val_loss': val_loss_avg,
                }, best_model_path)
                print(f"  -> Best model saved (dist={val_dist_avg:.3f})")

    print(f"\nTraining complete. Best val dist: {best_val_dist:.3f}")
    print(f"Model saved to: {best_model_path}")


if __name__ == '__main__':
    main()
