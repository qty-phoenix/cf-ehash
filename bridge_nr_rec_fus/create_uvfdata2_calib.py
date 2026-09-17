#!/usr/bin/env python3
"""Create uvfdata2 calibration CSV in NR-Rec-FUS format."""

import numpy as np
import os


def create_uvfdata2_calib(resample_factor=4):
    """
    NR-Rec-FUS calibration format: 8x4 CSV
    Rows 0-3: pixel_to_mm (4x4 matrix, includes resampling)
    Rows 4-7: mm_to_tool (4x4 matrix, spatial calibration)

    For uvfdata2:
      S_matrix: pixel to mm (diagonal scaling)
      C_matrix: image mm to tool (rigid transform)
    """

    # S matrix: pixel -> mm
    S = np.array([
        [0.229389190673828, 0, 0, 0],
        [0, 0.220979690551758, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # C matrix: image mm -> tool
    C = np.array([
        [0.231064671309448, -0.218052035, 0.948189025293473, -70.74132919],
        [-0.190847036, -0.965787273, -0.175591436, -80.6505661],
        [0.954036962825936, -0.140386088, -0.264773903, -46.17662239],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # In NR-Rec-FUS, the calibration reads:
    #   tform_calib_scale = pixel_mm @ diag(resample, resample, 1, 1)
    #   tform_calib_R_T = mm_tool
    #   tform_calib = mm_tool @ pixel_mm @ diag(resample, resample, 1, 1)
    #
    # We want tform_calib to map from RESAMPLED pixel coords to tool coords.
    # So pixel_mm should include the resample factor:
    #   pixel_mm_resampled = S @ diag(resample, resample, 1, 1)
    #   mm_tool = C # NOTE: In NR-Rec-FUS code, tform_calib = rows[4:8] @ rows[0:4] @ resample_mat
    #
    # Looking at read_calib_matrices:
    #   tform_calib_scale = rows[0:4] @ diag(resample, resample, 1, 1)  # pixel->mm with resample
    #   tform_calib_R_T = rows[4:8]                                      # mm->tool
    #   tform_calib = rows[4:8] @ rows[0:4] @ diag(resample, resample, 1, 1)
    #
    # So rows[0:4] should be the pixel->mm scaling WITHOUT resampling
    # (resampling is applied separately)
    # And rows[4:8] should be the mm->tool transform

    # pixel_mm: without resampling (resampling applied by read_calib_matrices)
    pixel_mm = S.copy()

    # mm_tool
    mm_tool = C.copy()

    # Stack as 8x4
    calib = np.vstack([pixel_mm, mm_tool])

    output_path = os.path.join(os.path.dirname(__file__), 'data', 'uvfdata2_calib.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, calib, delimiter=',', fmt='%.16f')
    print(f"Created uvfdata2 calibration: {output_path}")
    print(f"  pixel_mm (rows 0-3): diag({S[0,0]:.4f}, {S[1,1]:.4f}, 1, 1)")
    print(f"  mm_tool (rows 4-7): C matrix")
    print(f"  Resample factor: {resample_factor}")
    return output_path


if __name__ == '__main__':
    create_uvfdata2_calib()
