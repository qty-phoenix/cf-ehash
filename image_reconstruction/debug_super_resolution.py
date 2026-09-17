# Debug version of super resolution script
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import torch.nn.parallel
import importlib
import yaml
from PIL import Image
from datasets import build_dataloader
import argparse
import cv2


def debug_tensor_info(tensor, name):
    """Debug function to print tensor information"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")


def main(args):
    # Load configuration
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Setup device
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")
    
    # Load model
    spec = importlib.util.spec_from_file_location('build_model_from_logdir', 
                                                 os.path.join(args.logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    SINR, criterion = build_model_from_logdir.build_model_from_logdir(args.logdir, cfg, cfg['LOSS']).get()
    SINR.to(device)

    # Load the last trained model
    model_dir = os.path.join(args.logdir, 'trained_models')
    model_name = cfg['MODEL']['model_name']
    
    # Find the last model file
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f'{model_name}_model_') and f.endswith('.pth')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Get the epoch number from the last model file
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
    last_epoch = max(epochs)
    last_model_file = f'{model_name}_model_{last_epoch}.pth'
    
    print(f"Loading model from epoch {last_epoch}: {last_model_file}")
    model_path = os.path.join(model_dir, last_model_file)
    SINR.load_state_dict(torch.load(model_path, map_location=device))
    SINR.eval()

    # Load original image data for reference
    test_dataloader, test_set = build_dataloader(cfg, args.image_id, training=False)
    cfg['MODEL']['out_dim'] = test_set.img_channels
    cfg['MODEL']['dino_dim'] = test_set.dino_dim

    # Get original image data
    _, test_data = next(enumerate(test_dataloader))
    gt_img = test_data['gt_img'].to(device)
    dino = test_data['dino'].to(device) if 'dino' in test_data else None
    
    # Debug tensor shapes
    debug_tensor_info(gt_img, "gt_img")
    if dino is not None and dino != 0:
        debug_tensor_info(dino, "dino")
    
    # Get original image dimensions
    original_height, original_width = test_set.sidelength[0], test_set.sidelength[1]
    print(f"Original image size: {original_width}x{original_height}")
    print(f"Model aux_type: {cfg['MODEL'].get('aux_type')}")
    
    # Test with a small scale factor first
    scale_factor = 2
    new_height = original_height * scale_factor
    new_width = original_width * scale_factor
    
    print(f"Target resolution: {new_width}x{new_height}")
    
    # Create high resolution coordinate grid
    y_coords = torch.linspace(-1, 1, new_height, device=device)
    x_coords = torch.linspace(-1, 1, new_width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    debug_tensor_info(coords, "coords")
    
    # Test with a small batch first
    batch_size = 100
    batch_coords = coords[:batch_size]
    debug_tensor_info(batch_coords, "batch_coords")
    
    # Prepare input data
    input_data = {'coords': batch_coords}
    
    # Handle auxiliary inputs
    if cfg['MODEL'].get('aux_type') == 'img':
        print("Processing img auxiliary input...")
        # gt_img shape is (H*W, C), need to reshape to (C, H, W) for interpolation
        gt_img_reshaped = gt_img.reshape(original_height, original_width, -1).permute(2, 0, 1).unsqueeze(0)
        debug_tensor_info(gt_img_reshaped, "gt_img_reshaped")
        
        img_resized = torch.nn.functional.interpolate(
            gt_img_reshaped, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).reshape(-1, gt_img.shape[-1])
        debug_tensor_info(img_resized, "img_resized")
        
        # For the current batch, we need to select the corresponding pixels
        batch_img = img_resized[:batch_size]
        debug_tensor_info(batch_img, "batch_img")
        input_data['img'] = batch_img
    
    if dino is not None and dino != 0 and cfg['MODEL'].get('aux_type') == 'dino':
        print("Processing dino auxiliary input...")
        # Similar processing for DINO features
        dino_reshaped = dino.reshape(original_height, original_width, -1).permute(2, 0, 1).unsqueeze(0)
        debug_tensor_info(dino_reshaped, "dino_reshaped")
        
        dino_resized = torch.nn.functional.interpolate(
            dino_reshaped, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).reshape(-1, dino.shape[-1])
        debug_tensor_info(dino_resized, "dino_resized")
        
        batch_dino = dino_resized[:batch_size]
        debug_tensor_info(batch_dino, "batch_dino")
        input_data['dino'] = batch_dino
    
    print("Input data keys:", list(input_data.keys()))
    for key, value in input_data.items():
        debug_tensor_info(value, f"input_data[{key}]")
    
    # Forward pass
    print("Running forward pass...")
    try:
        output_pred = SINR(**input_data)
        print("Forward pass successful!")
        print("Output keys:", list(output_pred.keys()))
        for key, value in output_pred.items():
            if torch.is_tensor(value):
                debug_tensor_info(value, f"output_pred[{key}]")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def parse_args():
    parser = argparse.ArgumentParser(description='Debug Super Resolution RGB Image')
    parser.add_argument('--config', default='../configs/config_RGB.yaml', type=str, help='config file')
    parser.add_argument('--logdir', default='./log/debug_rgb', type=str, help='path to log directory')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--image_id', default='kodim19.png', type=str, help='image to process')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
