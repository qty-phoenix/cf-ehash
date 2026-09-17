#!/usr/bin/env python3
"""
Step 3: Run NR-Rec-FUS EfficientNet inference on uvfdata2 images,
predict frame-to-frame relative transforms, and convert to neural-ex format.

Output: Excel file with predicted poses in the same format as the original
        uvfdata2 pose file (Frame Index, Position X/Y/Z, Orientation X/Y/Z).
"""

import os
import sys
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Add NR-Rec-FUS to path
_nr_path = os.path.join(os.path.dirname(__file__), '..', '..', 'NR-Rec-FUS')
sys.path.insert(0, _nr_path)

from utils.network import build_model


def reference_image_points(image_H, image_W, density=2):
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
    }
    return type_dim_dict[label_pred_type]


def param_to_transform(params):
    """Convert 6-DoF parameters (rx,ry,rz,tx,ty,tz) to 4x4 matrix (ZYX convention)."""
    cos_x = torch.cos(params[..., 2])
    sin_x = torch.sin(params[..., 2])
    cos_y = torch.cos(params[..., 1])
    sin_y = torch.sin(params[..., 1])
    cos_z = torch.cos(params[..., 0])
    sin_z = torch.sin(params[..., 0])
    return torch.cat((
        torch.stack([cos_y * cos_z, sin_x * sin_y * cos_z - cos_x * sin_z,
                     cos_x * sin_y * cos_z + sin_x * sin_z, params[..., 3]], axis=2)[:, :, None, :],
        torch.stack([cos_y * sin_z, sin_x * sin_y * sin_z + cos_x * cos_z,
                     cos_x * sin_y * sin_z - sin_x * cos_z, params[..., 4]], axis=2)[:, :, None, :],
        torch.stack([-sin_y, sin_x * cos_y, cos_x * cos_y, params[..., 5]], axis=2)[:, :, None, :],
        torch.cat((torch.zeros_like(params[..., 0:3])[..., None, :],
                   torch.ones_like(params[..., 0])[..., None, None]), axis=3)
    ), axis=2)


def matrix_to_euler_zyx(R):
    """Convert 3x3 rotation matrix to ZYX Euler angles (degrees).

    R = Rz(rz) @ Ry(ry) @ Rx(rx)
    R[2,0] = -sin(ry), R[1,0] = cos(ry)*sin(rz), R[0,0] = cos(ry)*cos(rz)
    R[2,1] = cos(ry)*sin(rx), R[2,2] = cos(ry)*cos(rx)
    """
    sy = -R[2, 0]
    cy = torch.sqrt(torch.clamp(1 - sy ** 2, min=1e-8))
    rx = torch.atan2(R[2, 1] / cy, R[2, 2] / cy)
    ry = torch.atan2(sy, cy)
    rz = torch.atan2(R[1, 0] / cy, R[0, 0] / cy)
    return torch.rad2deg(torch.stack([rx, ry, rz]))


def predict_poses(images_dir, model_path, output_excel, num_samples=30,
                  resample_factor=4, gpu='0', pred_type='parameter',
                  model_name='efficientnet_b1'):
    """
    Predict frame poses using trained NR-Rec-FUS EfficientNet.

    Strategy: dense overlap + averaging to suppress error accumulation.
    - Uses stride=1 (dense sliding windows)
    - Each frame is covered by up to num_samples windows
    - For each frame, collects all window predictions of T_{i->0}
    - Averages across windows (rotation via SVD), reducing noise by ~√N
    - Sequential processing: frame i is finalized only after all windows
      covering it have been processed
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_paths = sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if f.endswith('.png')
    ])
    n_total = len(image_paths)
    print(f"Total images: {n_total}")

    sample_img = Image.open(image_paths[0]).convert('L')
    W_orig, H_orig = sample_img.size
    H_res, W_res = H_orig // resample_factor, W_orig // resample_factor
    print(f"Original: {H_orig}x{W_orig}, Resampled: {H_res}x{W_res}")

    print("Loading images...")
    all_frames = np.zeros((n_total, H_res, W_res), dtype=np.uint8)
    for i, p in enumerate(image_paths):
        img = Image.open(p).convert('L')
        img = img.resize((W_res, H_res), Image.BILINEAR)
        all_frames[i] = np.array(img, dtype=np.uint8)

    # ---- Build model ----
    image_points = reference_image_points(H_res, W_res, density=2)
    pred_dim = compute_dimension(pred_type, image_points.shape[1], num_samples, 'pred')

    from data.calib import read_calib_matrices
    calib_path = os.path.join(_nr_path, 'data', 'calib_matrix.csv')
    tform_calib_scale, tform_calib_R_T, tform_calib = read_calib_matrices(
        filename_calib=calib_path, resample_factor=resample_factor, device=device)

    model = build_model(
        type('Opt', (), {'model_name': model_name})(),
        in_frames=num_samples,
        pred_dim=pred_dim,
        label_dim=pred_dim,
        image_points=image_points.to(device),
        tform_calib=tform_calib,
        tform_calib_R_T=tform_calib_R_T
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_dist={checkpoint['val_dist']:.3f})")

    # ---- Dense overlap inference with averaging ----
    pred_type_dim = 6
    num_pairs = num_samples - 1
    stride = 1  # dense overlap

    # accum_transforms[i] = list of T_{i->0} estimates from different windows
    accum_transforms = [[] for _ in range(n_total)]
    # finalized_transforms[i] = averaged T_{i->0} after all windows covering i are done
    finalized_transforms = [None] * n_total
    finalized_transforms[0] = np.eye(4, dtype=np.float32)

    n_windows = n_total - num_samples + 1
    print(f"Running {n_windows} dense overlapping windows (stride={stride})...")

    for start in range(0, n_windows, stride):
        end = start + num_samples
        window_frames = torch.from_numpy(all_frames[start:end]).float() / 255.0
        window_frames = window_frames.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(window_frames)

        params = outputs.reshape(1, num_pairs, pred_type_dim)  # (1, N-1, 6)
        # transforms[offset] = T_{start+offset -> start} within this window
        transforms = param_to_transform(params)[0]  # (N-1, 4, 4)
        identity = torch.eye(4, device=device).unsqueeze(0)
        transforms = torch.cat([identity, transforms], dim=0)  # (N, 4, 4)
        transforms = transforms.cpu().numpy()

        # For each frame in this window, compute T_{frame -> 0}
        # Path: T_{i->0} = T_{i->start} @ T_{start->0}
        T_start_to_0 = finalized_transforms[start]
        if T_start_to_0 is None:
            # Reference frame not yet finalized, skip this window for now
            # (shouldn't happen with sequential processing)
            continue

        for offset in range(num_samples):
            frame_idx = start + offset
            T_rel = transforms[offset]  # T_{frame_idx -> start}
            T_i_to_0 = T_rel @ T_start_to_0
            # Re-orthogonalize
            U, _, Vt = np.linalg.svd(T_i_to_0[:3, :3])
            T_i_to_0[:3, :3] = U @ Vt
            accum_transforms[frame_idx].append(T_i_to_0)

        # After processing this window, check if frame (start+1) can be finalized.
        # Frame k is finalized when all windows starting at S < k that cover k
        # have been processed. This happens when start >= k.
        # Concretely: after window starting at 'start', frame (start+1) has been
        # covered by all windows starting at {0, 1, ..., start} that include it.
        # A window starting at S covers frame k if S <= k < S+num_samples.
        # For k = start+1, all windows with S <= start+1 and S > start+1-num_samples
        # cover it. Since we've already processed S=0..start, and the remaining
        # windows (S > start) don't cover k anymore, frame (start+1) is DONE.
        if start + 1 < n_total and finalized_transforms[start + 1] is None:
            k = start + 1
            if len(accum_transforms[k]) > 0:
                # Average all accumulated transforms
                stacked = np.stack(accum_transforms[k], axis=0)
                avg = np.mean(stacked, axis=0)
                U, _, Vt = np.linalg.svd(avg[:3, :3])
                avg[:3, :3] = U @ Vt
                finalized_transforms[k] = avg

        if (start + 1) % 50 == 0:
            print(f"  Frame {start+1}/{n_total} done")

    # Finalize remaining frames (last few frames)
    for i in range(n_total):
        if finalized_transforms[i] is None and len(accum_transforms[i]) > 0:
            stacked = np.stack(accum_transforms[i], axis=0)
            avg = np.mean(stacked, axis=0)
            U, _, Vt = np.linalg.svd(avg[:3, :3])
            avg[:3, :3] = U @ Vt
            finalized_transforms[i] = avg
        elif finalized_transforms[i] is None:
            # Edge case: copy from previous
            finalized_transforms[i] = finalized_transforms[i-1].copy()

    # ---- Convert to absolute world poses ----
    import pandas as pd
    pose_file = os.path.join(os.path.dirname(images_dir), 'coords3d', 'uvfdata2.xlsx')
    if not os.path.exists(pose_file):
        pose_file = os.path.join(os.path.dirname(os.path.dirname(images_dir)), 'coords3d', 'uvfdata2.xlsx')
    df_gt = pd.read_excel(pose_file)
    row0 = df_gt.iloc[0]
    trans0 = row0[['Position X', 'Position Y', 'Position Z']].values.astype(np.float64)
    rot0 = row0[['Orientation X', 'Orientation Y', 'Orientation Z']].values.astype(np.float64)
    T_0_to_world = np.eye(4, dtype=np.float64)
    rx, ry, rz = np.deg2rad(rot0)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    T_0_to_world[:3, :3] = Rz @ Ry @ Rx
    T_0_to_world[:3, 3] = trans0

    rows = []
    for i in range(n_total):
        T_i_to_0 = finalized_transforms[i]
        T_i_to_world = T_0_to_world @ T_i_to_0
        tx, ty, tz = float(T_i_to_world[0, 3]), float(T_i_to_world[1, 3]), float(T_i_to_world[2, 3])
        R = torch.from_numpy(T_i_to_world[:3, :3].astype(np.float32))
        euler = matrix_to_euler_zyx(R)
        rows.append({
            'Frame Index': i,
            'Position X': tx,
            'Position Y': ty,
            'Position Z': tz,
            'Orientation X': float(euler[0].item()),
            'Orientation Y': float(euler[1].item()),
            'Orientation Z': float(euler[2].item()),
        })

    df_pred = pd.DataFrame(rows)
    df_pred.to_excel(output_excel, index=False)
    print(f"Predicted poses saved to: {output_excel} ({len(df_pred)} frames)")

    return df_pred


def main():
    parser = argparse.ArgumentParser(description='Predict poses using NR-Rec-FUS model')
    parser.add_argument('--images_dir', type=str, default='./MyMoE/uvfdata2')
    parser.add_argument('--model_path', type=str,
                        default='./bridge_nr_rec_fus/models/best_pose_estimator.pth')
    parser.add_argument('--output_excel', type=str,
                        default='./bridge_nr_rec_fus/data/predicted_poses_uvfdata2.xlsx')
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--resample_factor', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    predict_poses(
        images_dir=args.images_dir,
        model_path=args.model_path,
        output_excel=args.output_excel,
        num_samples=args.num_samples,
        resample_factor=args.resample_factor,
        gpu=args.gpu,
    )


if __name__ == '__main__':
    main()
