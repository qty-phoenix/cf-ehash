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


def safe_forward(model, coords, dino=None, img=None):
    """安全的forward调用，处理维度不匹配问题"""
    try:
        return model(coords, dino=dino, img=img)
    except RuntimeError as e:
        if "Index tensor must have the same number of dimensions" in str(e):
            print(f"警告: 遇到维度不匹配错误，尝试使用备用方法: {e}")
            # 尝试不使用dino和img参数
            try:
                return model(coords)
            except:
                # 如果还是失败，返回None
                print("错误: 无法进行前向传播")
                return None
        else:
            raise e


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
                dino = data.get('dino', None)
                if dino is not None:
                    dino = dino.to(device)
                
                # 使用安全的forward调用
                out = safe_forward(SINR, coords, dino=dino, img=gt_img)
                
                if out is None:
                    print(f"跳过 epoch {epoch}, sample {i}")
                    continue

                # 处理预测结果
                pred = None
                if 'selected_nonmanifold_pnts_pred' in out:
                    pred = out['selected_nonmanifold_pnts_pred']
                    print(f"使用 selected_nonmanifold_pnts_pred, 形状: {pred.shape}")
                elif 'nonmanifold_pnts_pred' in out:
                    pred = out['nonmanifold_pnts_pred']
                    print(f"使用 nonmanifold_pnts_pred, 形状: {pred.shape}")
                else:
                    print(f"警告: 未找到预期的预测输出键，可用键: {list(out.keys())}")
                    continue

                # 处理不同的维度情况
                if pred.dim() == 4:
                    # (1, K, H*W, C) 或 (1, C, H, W) 或 (1, K, H, W)
                    if pred.shape[1] != gt_img.shape[0] and pred.shape[1] > 1:
                        # 多个专家，选择第一个
                        pred = pred[:, 0]
                    elif pred.shape[1] == gt_img.shape[0]:
                        # 已经是正确的形状
                        pass
                    else:
                        # 需要重新整形
                        H = gt_img.shape[-2]
                        W = gt_img.shape[-1]
                        C = gt_img.shape[0]
                        if pred.shape[-1] == C:
                            pred = pred.reshape(1, H, W, C).permute(0, 3, 1, 2)
                        else:
                            pred = pred.reshape(1, H, W, -1).permute(0, 3, 1, 2)
                elif pred.dim() == 3:
                    # (1, H*W, C) 或 (1, C, H*W)
                    H = gt_img.shape[-2]
                    W = gt_img.shape[-1]
                    C = gt_img.shape[0]
                    if pred.shape[-1] == C:
                        pred = pred.reshape(1, H, W, C).permute(0, 3, 1, 2)
                    else:
                        pred = pred.reshape(1, H, W, -1).permute(0, 3, 1, 2)
                elif pred.dim() == 2:
                    # (1, H*W) - 灰度图像
                    H = gt_img.shape[-2]
                    W = gt_img.shape[-1]
                    pred = pred.reshape(1, 1, H, W)

                # 确保pred的形状与gt_img匹配
                if pred.shape != gt_img.shape:
                    print(f"警告: pred形状 {pred.shape} 与 gt_img形状 {gt_img.shape} 不匹配")
                    # 尝试调整到匹配的形状
                    if pred.shape[0] == gt_img.shape[0] and pred.shape[2:] == gt_img.shape[2:]:
                        # 通道数不同，取第一个通道
                        pred = pred[:, :1]
                    elif pred.shape[1:] == gt_img.shape[1:]:
                        # batch维度不同
                        pred = pred[:1]

                pred_np = pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                pred_u8 = convert_to_uint8(pred_np)
                Image.fromarray(pred_u8.squeeze()).save(os.path.join(output_dir, f'epoch_{epoch:06d}_sample_{i:04d}.png'))
                print(f"保存图像: epoch_{epoch:06d}_sample_{i:04d}.png")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGBPose3D with MoE-INR (Fixed Version)')
    parser.add_argument('--config', default='../configs/config_RGB_pose3d.yaml', type=str)
    parser.add_argument('--logdir', default='./log/pose3d', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)

