"""
Evaluate Ultra-NeRF on uvfdata2 test set.
Loads saved checkpoint and computes PSNR/SSIM/MSE/MAE.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from models import build_model
from datasets.RGBPose3D_GlobalNorm_Cached import RGBPose3DDatasetGlobalNorm
from datasets.RGBPose3D_Cached import RGBPose3DDataset

def evaluate():
    config_path = 'configs/config_RGB_pose3d_ultranerf_uvfdata2.yaml'
    ckpt_path = 'log/ultranerf_uvfdata2/models/inr_ultranerf_rgb_gradient_fixed_0.pth'
    device = torch.device('cuda:0')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg_data = cfg['DATA']
    use_global_norm = cfg_data.get('use_global_norm', False)
    DatasetClass = RGBPose3DDatasetGlobalNorm if use_global_norm else RGBPose3DDataset
    test_set = DatasetClass(
        cfg_data['dataset_path'], cfg_data['pose_file'],
        copy_to_gpu=False, grayscale=True,
        crop_size=cfg_data.get('crop_size', None),
        angles_in_degrees=cfg_data.get('angles_in_degrees', False),
        mode='test',
        use_uvfdata2_calibration=cfg_data.get('use_uvfdata2_calibration', False)
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    print(f'Test set: {len(test_set)} images')

    model, _ = build_model(cfg, cfg['LOSS'])
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    all_mse, all_psnr, all_mae, all_ssim = [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            coords = data['coords'].to(device).squeeze(1)  # (1, H*W, 3)
            gt_img = data['gt_img'].to(device)              # (1, C, H, W)
            B, C, H, W = gt_img.shape

            out = model(coords)
            pred = out['nonmanifold_pnts_pred']  # (B, out_dim, N)
            pred = pred.squeeze(0).squeeze(0).reshape(H, W)  # (H, W)
            pred = torch.clamp(pred, 0.0, 1.0)

            gt = gt_img.squeeze(0).squeeze(0)  # (H, W)

            mse = torch.mean((pred - gt) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
            mae = torch.mean(torch.abs(pred - gt)).item()

            pred_np = pred.cpu().numpy()
            gt_np = gt.cpu().numpy()
            ssim_val = ssim(gt_np, pred_np, data_range=1.0, channel_axis=None)

            all_mse.append(mse); all_psnr.append(psnr); all_mae.append(mae); all_ssim.append(ssim_val)

            if i < 3 or i % 20 == 0:
                print(f'  {i:3d}: PSNR={psnr:.2f}  SSIM={ssim_val:.4f}  MSE={mse:.6f}  MAE={mae:.4f}')

    print(f'\n===== Ultra-NeRF on uvfdata2: Test Set Metrics =====')
    print(f'  PSNR:  {np.mean(all_psnr):.3f} dB')
    print(f'  SSIM:  {np.mean(all_ssim):.4f}')
    print(f'  MSE:   {np.mean(all_mse):.6f}')
    print(f'  MAE:   {np.mean(all_mae):.4f}')
    return np.mean(all_psnr), np.mean(all_ssim)

if __name__ == '__main__':
    evaluate()
