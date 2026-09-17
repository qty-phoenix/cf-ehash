import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from models import build_model
from models.stage_handler import TrainingStageHandler
from datasets.RGBPose3D import RGBPose3DDataset


def main(args):
    cfg = yaml.safe_load(open(args.config))
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])

    # dirs
    os.makedirs(args.logdir, exist_ok=True)
    model_outdir = os.path.join(args.logdir, 'trained_models')
    os.makedirs(model_outdir, exist_ok=True)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # dataset & dataloader
    images_dir = cfg['DATA']['dataset_path']
    pose_file = cfg['DATA'].get('pose_file', cfg['DATA'].get('coords3d_path', ''))
    is_grayscale = (cfg['MODEL']['out_dim'] == 1)
    train_set = RGBPose3DDataset(images_dir, pose_file, copy_to_gpu=False, grayscale=is_grayscale,
                                 crop_size=cfg['DATA'].get('crop_size', None),
                                 angles_in_degrees=cfg['DATA'].get('angles_in_degrees', False),
                                 mode='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    # model & loss
    cfg['MODEL']['out_dim'] = cfg['MODEL']['out_dim']
    SINR, _ = build_model(cfg, cfg['LOSS'])
    n_parameters = sum(p.numel() for p in SINR.parameters() if p.requires_grad)
    print("#params:", n_parameters)

    # ensure n_samples exists for stage handler (RGB hack similar to train_rgbimage.py)
    cfg['TRAINING']['n_samples'] = cfg['TRAINING']['num_epochs']
    training_stage_handler = TrainingStageHandler(cfg['TRAINING']['stages'], SINR, cfg)
    criterion = training_stage_handler.criterion

    lr = cfg['TRAINING']['lr'] if isinstance(cfg['TRAINING']['lr'], float) else cfg['TRAINING']['lr']['all']
    optimizer = optim.Adam(training_stage_handler.get_trainable_params(), lr=lr, betas=(0.9, 0.999))
    training_stage_handler.freeze_params()
    scheduler = training_stage_handler.get_scheduler(optimizer)

    SINR.to(device)

    for epoch in range(cfg['TRAINING']['num_epochs']):
        SINR.train()
        for batch_idx, data in enumerate(train_loader):
            coords = data['coords'].to(device)           # (1, H*W, 3)
            coords = coords.squeeze(1)
            gt_img = data['gt_img'].to(device)           # (C, H, W)
            segments = data['segments'].to(device)       # (H*W,)
            dino = data['dino'].to(device)

            coords.requires_grad_()

            output_pred = SINR(coords, dino=dino, img=gt_img)
            B, C, H, W = gt_img.shape  # 1, 1, 160, 160
            gt_img_flat = gt_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
            loss_dict = criterion(output_pred=output_pred,
                                  coords=coords,
                                  gt={'img': gt_img_flat, 'segment': segments, 'aux': gt_img_flat},
                                  model=SINR)
            #loss_dict = criterion(output_pred=output_pred, coords=coords, gt={'img': gt_img, 'segment': segments, 'aux': gt_img}, model=SINR)

            optimizer.zero_grad(set_to_none=True)
            loss_dict['loss'].backward()
            optimizer.step()

        if epoch % 100 == 0:
            torch.save(SINR.state_dict(), os.path.join(model_outdir, '%s_model_%d.pth' % (cfg['MODEL']['model_name'], epoch)))
        scheduler.step()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Training RGBPose3D with MoE-INR')
    parser.add_argument('--config', default='../configs/config_RGB_pose3d.yaml', type=str)
    parser.add_argument('--logdir', default='./log/pose3d', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


