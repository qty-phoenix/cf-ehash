#!/usr/bin/env python3
"""
Step 1: Convert uvfdata2 data to NR-Rec-FUS H5 format.

uvfdata2:
  - 519 grayscale PNG images at 640x480
  - Poses in Excel: Frame Index, Position X/Y/Z, Orientation X/Y/Z (Euler angles, degrees)
  - S and C calibration matrices (pixel->mm->tool)

NR-Rec-FUS H5 format:
  - /subXXX_framesYY: (N, H, W) uint8 image frames
  - /subXXX_tformsYY: (N, 4, 4) transformation matrices (tool->world)
  - /subXXX_tforms_invYY: (N, 4, 4) inverse transformation matrices (world->tool)
  - /frame_size: (2,) [H, W]
  - /num_frames: (n_subjects, n_scans) frame counts
  - /name_scan: scan protocol names
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import torch
from PIL import Image
from tqdm import tqdm
import json
import argparse

# Add NR-Rec-FUS to path for calib reading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'NR-Rec-FUS'))


def euler_to_matrix(angles_deg, translation):
    """Convert Euler angles (ZYX, degrees) + translation to 4x4 transform matrix."""
    rx, ry, rz = np.deg2rad(angles_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


def build_uvfdata2_tforms(pose_df, s_matrix, c_matrix, H, W):
    """
    Build tool->world transforms from uvfdata2 poses + calibration.

    In neural-ex, the coordinate flow is:
      pixel -> S_matrix (pixel to mm) -> C_matrix (image to tool) -> R/T (tool to world)

    The R/T from the pose file is applied AFTER S and C, meaning:
      world_coord = R * (C * (S * pixel_homogeneous)) + T

    Here we compute the full tool->world transform for each frame.
    The NR-Rec-FUS format expects tool->world (tforms) and world->tool (tforms_inv).
    """
    n_frames = len(pose_df)
    tforms = np.zeros((n_frames, 4, 4), dtype=np.float32)
    tforms_inv = np.zeros((n_frames, 4, 4), dtype=np.float32)

    for i in range(n_frames):
        row = pose_df.iloc[i]
        trans = row[['Position X', 'Position Y', 'Position Z']].values.astype(np.float32)
        rot = row[['Orientation X', 'Orientation Y', 'Orientation Z']].values.astype(np.float32)

        T = euler_to_matrix(rot, trans)
        tforms[i] = T
        tforms_inv[i] = np.linalg.inv(T)

    return tforms, tforms_inv


def load_and_resize_image(img_path, grayscale=True, resample_factor=4):
    """Load image and resize according to resample_factor."""
    img = Image.open(img_path)
    if grayscale:
        img = img.convert('L')
    W, H = img.size
    new_W, new_H = W // resample_factor, H // resample_factor
    img = img.resize((new_W, new_H), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def prepare_data(images_dir, pose_file, output_h5, resample_factor=4,
                 train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                 grayscale=True, num_samples=30, window_stride=5):
    """
    Convert uvfdata2 data to NR-Rec-FUS H5 format.

    uvfdata2 is a single continuous 519-frame scan. We create many overlapping
    windows of NUM_SAMPLES frames each to serve as training "scans".
    """
    # Load poses
    pose_df = pd.read_excel(pose_file)
    n_total = len(pose_df)
    print(f"Total frames: {n_total}")

    # Load images and get dimensions
    image_paths = sorted(
        [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
    assert len(image_paths) == n_total, \
        f"Image count {len(image_paths)} != pose count {n_total}"

    sample_img = load_and_resize_image(image_paths[0], grayscale, resample_factor)
    H, W = sample_img.shape
    print(f"Resampled image size: {H}x{W} (factor={resample_factor})")

    # Build transforms
    print("Building transforms...")
    tforms, tforms_inv = build_uvfdata2_tforms(pose_df, None, None, H, W)

    # Load all frames
    print("Loading and resizing all frames...")
    all_frames = np.zeros((n_total, H, W), dtype=np.uint8)
    for i in tqdm(range(n_total)):
        all_frames[i] = load_and_resize_image(image_paths[i], grayscale, resample_factor)

    # Create overlapping windows from the single sequence
    scans = []
    for start in range(0, n_total - num_samples + 1, window_stride):
        scans.append((start, start + num_samples))

    n_scans = len(scans)
    print(f"Created {n_scans} windows (NUM_SAMPLES={num_samples}, stride={window_stride})")

    # Assign to subjects: ~10 scans per subject
    scans_per_subject = min(n_scans, 10)
    n_subjects = (n_scans + scans_per_subject - 1) // scans_per_subject

    print(f"  {n_subjects} subjects, {scans_per_subject} scans per subject")

    # Create H5 file
    print(f"Creating H5 file: {output_h5}")

    with h5py.File(output_h5, 'w') as f:
        num_frames_array = np.zeros((n_subjects, scans_per_subject), dtype=np.int32)
        name_scan_list = []

        scan_idx = 0
        for sub in range(n_subjects):
            for scn in range(scans_per_subject):
                if scan_idx >= n_scans:
                    break
                start, end = scans[scan_idx]
                n_frames_in_scan = end - start

                f.create_dataset(
                    f'/sub{sub:03d}_frames{scn:02d}',
                    data=all_frames[start:end],
                    dtype=np.uint8
                )
                f.create_dataset(
                    f'/sub{sub:03d}_tforms{scn:02d}',
                    data=tforms[start:end],
                    dtype=np.float32
                )
                f.create_dataset(
                    f'/sub{sub:03d}_tforms_inv{scn:02d}',
                    data=tforms_inv[start:end],
                    dtype=np.float32
                )

                num_frames_array[sub, scn] = n_frames_in_scan
                name_scan_list.append(f'sub{sub:03d}_scan{scn:02d}')
                scan_idx += 1

        f.create_dataset('frame_size', data=np.array([H, W], dtype=np.int32))
        f.create_dataset('num_frames', data=num_frames_array)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('name_scan', data=np.array(name_scan_list, dtype=object), dtype=dt)

    n_actual = scan_idx
    print(f"H5 file created: {output_h5} ({n_actual} scans)")

    # Split into train/val/test folds at the SCAN level (not subject level),
    # since all windows come from the same sequence.
    indices = list(range(n_actual))
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(n_actual * train_ratio)
    n_val = int(n_actual * val_ratio)
    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    test_idx = sorted(indices[n_train + n_val:])

    def idx_to_tuple(idx):
        sub = idx // scans_per_subject
        scn = idx % scans_per_subject
        return [sub, scn]

    fold_splits = [
        ('fold_00', train_idx[:len(train_idx)//3]),
        ('fold_01', train_idx[len(train_idx)//3:2*len(train_idx)//3]),
        ('fold_02', train_idx[2*len(train_idx)//3:]),
        ('fold_03', val_idx),
        ('fold_04', test_idx),
    ]

    output_dir = os.path.dirname(output_h5)
    for fold_name, fold_indices in fold_splits:
        if len(fold_indices) == 0:
            continue
        index_tuples = [idx_to_tuple(i) for i in fold_indices]
        fold_file = os.path.join(output_dir, f'{fold_name}_seqlen{num_samples}_scan_uvfdata2.json')
        with open(fold_file, 'w') as f:
            json.dump({
                'min_scan_len': num_samples,
                'filename': output_h5,
                'indices_in_use': index_tuples,
                'num_samples': num_samples,
                'sample_range': num_samples,
            }, f, indent=2)
        print(f"Saved {fold_file} with {len(index_tuples)} scans")

    # Full sequence H5 (all 519 frames as one scan, for inference)
    full_h5 = output_h5.replace('.h5', '_full.h5')
    with h5py.File(full_h5, 'w') as f:
        f.create_dataset('/sub000_frames00', data=all_frames, dtype=np.uint8)
        f.create_dataset('/sub000_tforms00', data=tforms, dtype=np.float32)
        f.create_dataset('/sub000_tforms_inv00', data=tforms_inv, dtype=np.float32)
        f.create_dataset('frame_size', data=np.array([H, W], dtype=np.int32))
        f.create_dataset('num_frames', data=np.array([[n_total]], dtype=np.int32))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('name_scan', data=np.array(['full_sequence'], dtype=object), dtype=dt)
    print(f"Full sequence H5 created: {full_h5}")

    return output_h5


def main():
    parser = argparse.ArgumentParser(description='Prepare uvfdata2 data for NR-Rec-FUS')
    parser.add_argument('--images_dir', type=str,
                        default='./MyMoE/uvfdata2',
                        help='Path to uvfdata2 images')
    parser.add_argument('--pose_file', type=str,
                        default='./MyMoE/coords3d/uvfdata2.xlsx',
                        help='Path to pose Excel file')
    parser.add_argument('--output_h5', type=str,
                        default='./bridge_nr_rec_fus/data/uvfdata2_res4.h5',
                        help='Output H5 file path')
    parser.add_argument('--resample_factor', type=int, default=4,
                        help='Image resize factor')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of frames per training window')
    parser.add_argument('--window_stride', type=int, default=5,
                        help='Stride between training windows')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_h5), exist_ok=True)
    prepare_data(
        images_dir=args.images_dir,
        pose_file=args.pose_file,
        output_h5=args.output_h5,
        resample_factor=args.resample_factor,
        num_samples=args.num_samples,
        window_stride=args.window_stride,
    )


if __name__ == '__main__':
    main()
