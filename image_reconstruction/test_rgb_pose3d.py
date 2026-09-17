import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from datasets.RGBPose3D import RGBPose3DDataset
import importlib


def convert_to_uint8(img):
    if img.shape[-1] == 1:
        img = img.squeeze()
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    else:
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


def main(args):
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # build model from training logdir
    spec = importlib.util.spec_from_file_location('build_model_from_logdir', os.path.join(args.logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    SINR, _ = build_model_from_logdir.build_model_from_logdir(args.logdir, cfg, cfg['LOSS']).get()
    SINR.to(device)

    # dataset
    images_dir = cfg['DATA']['dataset_path']
    pose_file = cfg['DATA'].get('pose_file', cfg['DATA'].get('coords3d_path', ''))
    is_grayscale = (cfg['MODEL']['out_dim'] == 1)
    test_set = RGBPose3DDataset(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                                crop_size=cfg['DATA'].get('crop_size', None),
                                angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                                mode='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model_dir = os.path.join(args.logdir, 'trained_models')
    output_dir = os.path.join(args.logdir, 'vis_pose3d')
    os.makedirs(output_dir, exist_ok=True)

    checkpoints = range(cfg['TESTING']['epoch_n_eval'][0], cfg['TESTING']['epoch_n_eval'][1], cfg['TESTING']['epoch_n_eval'][2])

    for epoch in checkpoints:
        model_file = os.path.join(model_dir, '%s_model_%d.pth' % (cfg['MODEL']['model_name'], epoch))
        if not os.path.isfile(model_file):
            continue
        SINR.load_state_dict(torch.load(model_file, map_location=device))
        SINR.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                coords = data['coords'].to(device)
                gt_img = data['gt_img'].to(device)
                dino = data['dino'].to(device)
                out = SINR(coords, dino=dino, img=gt_img)

                if 'selected_nonmanifold_pnts_pred' in out:
                    pred = out['selected_nonmanifold_pnts_pred']  # (1, H*W, C) or (1, C, H, W)
                    if pred.dim() == 3:
                        H = gt_img.shape[-2]
                        W = gt_img.shape[-1]
                        C = gt_img.shape[0]
                        pred = pred.reshape(1, H, W, C).permute(0, 3, 1, 2)
                else:
                    pred = out['nonmanifold_pnts_pred']  # (1, C, H, W) or (1, K, H, W)
                    if pred.dim() == 4 and pred.shape[1] != gt_img.shape[0]:
                        # pick top-1 expert output if parallel dimension exists
                        pred = pred[:, 0]

                pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                pred_u8 = convert_to_uint8(pred_np)
                Image.fromarray(pred_u8.squeeze()).save(os.path.join(output_dir, f'epoch_{epoch:06d}_sample_{i:04d}.png'))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGBPose3D with MoE-INR')
    parser.add_argument('--config', default='../configs/config_RGB_pose3d.yaml', type=str)
    parser.add_argument('--logdir', default='./log/pose3d', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)


