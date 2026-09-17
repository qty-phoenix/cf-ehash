# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import utils.visualizations as vis
import utils.diff_operators as diff_operators
import utils.dataio as dataio
import utils.utils as utils
import numpy as np
import torch.nn.parallel
import importlib
import yaml
from PIL import Image
from datasets import build_dataloader
from models.rgb_losses import PSNR, RGBReconLossSingle
import pickle
import cmapy
import cv2


def create_high_res_coords(height, width, device):
    """Create coordinate grid for high resolution output using the same method as original dataset"""
    # Use exactly the same coordinate generation as the original dataset
    # Create the same grid as get_mgrid function
    pixel_coords = np.stack(np.mgrid[:height, :width], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (height - 1)  # x coordinates
    pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (width - 1)   # y coordinates
    
    pixel_coords -= 0.5
    pixel_coords *= 2.
    
    # Convert to torch tensor and reshape
    coords = torch.from_numpy(pixel_coords).to(device).view(-1, 2).unsqueeze(0)
    return coords


def convert_to_uint8(img):
    if img.shape[-1] == 1:
        img = img.squeeze()
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    else:
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img


def main(args):
    # get training and testing parameters
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # get data loaders
    test_dataloader, test_set = build_dataloader(cfg, args.image_id, training=False)
    cfg['MODEL']['out_dim'] = test_set.img_channels

    # get model
    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")

    spec = importlib.util.spec_from_file_location('build_model_from_logdir', os.path.join(args.logdir, 'models', '__init__.py'))
    build_model_from_logdir = importlib.util.module_from_spec(spec)
    sys.modules['build_model_from_logdir'] = build_model_from_logdir
    spec.loader.exec_module(build_model_from_logdir)
    SINR, criterion = build_model_from_logdir.build_model_from_logdir(args.logdir, cfg, cfg['LOSS']).get()
    SINR.to(device)

    model_dir = os.path.join(args.logdir, 'trained_models')
    output_dir = os.path.join(args.logdir, 'vis')
    recon_img_outdir = os.path.join(args.logdir, 'reconstructed_images')
    high_res_dir = os.path.join(args.logdir, 'high_resolution')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(high_res_dir, exist_ok=True)
    # get loss


    _, test_data = next(enumerate(test_dataloader))
    SINR.eval()
    coords, gt_img, dino = test_data['coords'].to(device), test_data['gt_img'].to(device), test_data['dino'].to(device)

    epoch_n_eval = cfg['TESTING']['epoch_n_eval']
    eval_epochs = range(epoch_n_eval[0], epoch_n_eval[1], epoch_n_eval[2])

    psnr_module = PSNR()
    mse_module = RGBReconLossSingle()
    eval_metrics = {'Epochs': [], 'PSNR': [], 'MSE': []}

    for epoch in eval_epochs:
        print("Visualizing  {} epoch {}".format(args.image_id, epoch))

        model_filename = os.path.join(model_dir, '%s_model_%d.pth' % (cfg['MODEL']['model_name'], epoch))
        SINR.load_state_dict(torch.load(model_filename, map_location=device))
        SINR.to(device)

        coords.requires_grad_()
        output_pred = SINR(coords, dino=dino, img=gt_img)
        # loss_dict = criterion(output_pred=output_pred,  coords=coords, gt={'img': gt_img, 'segment': test_data['segments']})

        img_dict = {}
        if "moe" in cfg['MODEL']['model_name']:
            recon_img = output_pred['selected_nonmanifold_pnts_pred'].detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                             test_set.sidelength[1], -1)

            recon_img_tensor = output_pred['selected_nonmanifold_pnts_pred'].squeeze(0)
            selected_expert_idx = output_pred['nonmnfld_selected_expert_idx']
            q = output_pred['nonmnfld_q']
            if cfg['TESTING'].get('plot_q_grad', True):
                q_grad = utils.experts_gradient(coords, q).norm(2, dim=-1)
                q_grad = q_grad.gather(dim=1, index=selected_expert_idx[None, None, :]).cpu().detach().numpy().reshape(
                    test_set.sidelength[0], test_set.sidelength[1])
            else:
                q_grad = None
            selected_expert_idx = selected_expert_idx.cpu().detach().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1])
            q = q.squeeze().cpu().detach().numpy().reshape(-1, test_set.sidelength[0], test_set.sidelength[1])
            (img_dict['experts'], img_dict['experts_heatmap'], img_dict['q_grad'], img_dict['q_image_array_list'],
             img_dict['q_distributions'], img_dict['q_dist_array'], _) = (
                vis.plot_rgb_experts(selected_expert_idx, q, q_grad, example_idx=0, clim=(0.0, 0.5)))
            img_dict['reconstructed_img_per_experts'] = []
            for i in range(output_pred['nonmanifold_pnts_pred'].shape[1]):
                recon_img_e = output_pred['nonmanifold_pnts_pred'][:, i].detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                             test_set.sidelength[1], -1)
                recon_img_e = convert_to_uint8(recon_img_e)
                img_dict['reconstructed_img_per_experts'].append(recon_img_e)
        else:
            recon_img = output_pred['nonmanifold_pnts_pred'].permute(0, 2, 1).detach().cpu().numpy().reshape(test_set.sidelength[0],
                                                                                    test_set.sidelength[1], -1)
            recon_img_tensor = output_pred['nonmanifold_pnts_pred']

        img_gradient = diff_operators.gradient(recon_img_tensor, coords)
        img_laplace = diff_operators.laplace(recon_img_tensor, coords)
        pred_grad = dataio.grads2img(dataio.lin2img(img_gradient,
                                                    image_resolution=test_set.sidelength)).permute(1, 2, 0).squeeze().detach().cpu().numpy()

        pred_lapl = cv2.cvtColor(cv2.applyColorMap(dataio.to_uint8(dataio.rescale_img(
            dataio.lin2img(img_laplace, image_resolution=test_set.sidelength), perc=2).permute(0, 2, 3, 1).squeeze(0).detach().cpu().numpy()),
                                                   cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)

        img_dict['gradient_img'] = convert_to_uint8(pred_grad)
        img_dict['laplacian_img'] = pred_lapl

        error_img = (recon_img - gt_img.cpu().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1], -1))**2

        img_dict['error_img'] = convert_to_uint8(error_img)
        img_dict['reconstructed_img'] = convert_to_uint8(recon_img).squeeze() #.transpose(1, 0, 2)

        eval_metrics['Epochs'].append(epoch)
        mse = mse_module.compute_loss(torch.tensor(recon_img, device=gt_img.device),
                                      gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1))
        psnr = psnr_module(gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1), mse)
        eval_metrics['MSE'].append(mse.item())
        eval_metrics['PSNR'].append(psnr.item())
        eval_metric_img_dict = vis.plot_eval_metrics(eval_metrics, data_range={'x':[0, max(eval_epochs)],
                                                                               'y':[0, 100]})
        img_dict.update(eval_metric_img_dict)

        # save the generated images
        for key, val in img_dict.items():
            print('Saving images: ', key)
            if val is not None:
                if type(val) is list:
                    os.makedirs(os.path.join(output_dir, key), exist_ok=True)
                    for i, v in enumerate(val):
                        im = Image.fromarray(v)
                        im.save(os.path.join(output_dir, key, "expert_" + str(i) + "_" + str(epoch).zfill(6) + ".png"))
                else:
                    im = Image.fromarray(val)
                    im.save(os.path.join(output_dir, key + "_" + str(epoch).zfill(6) + ".png"))

        del img_laplace, img_gradient # free memory
        
        # Generate high resolution image
        if args.high_resolution:
            print(f"Generating high resolution image for epoch {epoch}")
            high_res_height = args.high_res_height if args.high_res_height else test_set.sidelength[0] * 4  # Default 4x upscaling
            high_res_width = args.high_res_width if args.high_res_width else test_set.sidelength[1] * 4
            
            # Create high resolution coordinates
            high_res_coords = create_high_res_coords(high_res_height, high_res_width, device)
            
            # Process in batches to avoid memory issues
            # Adjust batch size based on resolution to avoid memory issues
            total_pixels = high_res_height * high_res_width
            if total_pixels > 100000:  # For very high resolution
                batch_size = min(args.batch_size, 5000)
            else:
                batch_size = args.batch_size if args.batch_size else 10000
            high_res_img = torch.zeros((high_res_height, high_res_width, cfg['MODEL']['out_dim']), device=device)
            
            with torch.no_grad():
                for i in range(0, high_res_coords.shape[1], batch_size):
                    batch_coords = high_res_coords[:, i:i+batch_size, :]
                    
                    # Prepare auxiliary inputs for high resolution
                    input_data = {}
                    if cfg['MODEL'].get('aux_type') == 'dino' and dino is not None and dino != 0:
                        # Interpolate DINO features to high resolution
                        dino_reshaped = dino.reshape(test_set.sidelength[0], test_set.sidelength[1], -1).permute(2, 0, 1).unsqueeze(0)
                        dino_resized = torch.nn.functional.interpolate(
                            dino_reshaped, 
                            size=(high_res_height, high_res_width), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0).reshape(-1, dino.shape[-1])
                        batch_dino = dino_resized[i:i+batch_size]
                        input_data['dino'] = batch_dino
                    elif cfg['MODEL'].get('aux_type') == 'img':
                        # Interpolate image to high resolution
                        gt_img_reshaped = gt_img.reshape(test_set.sidelength[0], test_set.sidelength[1], -1).permute(2, 0, 1).unsqueeze(0)
                        img_resized = torch.nn.functional.interpolate(
                            gt_img_reshaped, 
                            size=(high_res_height, high_res_width), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze(0).permute(1, 2, 0).reshape(-1, gt_img.shape[-1])
                        batch_img = img_resized[i:i+batch_size]
                        input_data['img'] = batch_img
                    
                    # Forward pass
                    output_pred_hr = SINR(batch_coords, **input_data)
                    
                    # Extract prediction
                    if "moe" in cfg['MODEL']['model_name']:
                        pred_hr = output_pred_hr['selected_nonmanifold_pnts_pred']
                    else:
                        pred_hr = output_pred_hr['nonmanifold_pnts_pred']
                    
                    # Store results - use the same reshaping as original dataset
                    pred_reshaped = pred_hr.reshape(-1, cfg['MODEL']['out_dim'])
                    batch_indices = torch.arange(i, min(i+batch_size, high_res_coords.shape[1]), device=device)
                    
                    # The coordinates are generated in the same order as the original dataset
                    # which uses np.mgrid[:height, :width], so we can directly use the indices
                    batch_y = batch_indices // high_res_width
                    batch_x = batch_indices % high_res_width
                    
                    # Ensure we don't exceed bounds
                    valid_indices = (batch_y < high_res_height) & (batch_x < high_res_width)
                    if valid_indices.any():
                        high_res_img[batch_y[valid_indices], batch_x[valid_indices]] = pred_reshaped[:len(batch_indices)][valid_indices]
            
            # Convert to numpy and save
            high_res_img_np = high_res_img.detach().cpu().numpy()
            high_res_img_uint8 = convert_to_uint8(high_res_img_np)
            
            # Debug: print image statistics
            print(f"High res image shape: {high_res_img_np.shape}")
            print(f"High res image range: [{high_res_img_np.min():.4f}, {high_res_img_np.max():.4f}]")
            
            if high_res_img_uint8.shape[-1] == 1:
                img_pil = Image.fromarray(high_res_img_uint8.squeeze(), mode='L')
            else:
                img_pil = Image.fromarray(high_res_img_uint8, mode='RGB')
            
            high_res_filename = f'high_res_{args.image_id}_{high_res_width}x{high_res_height}_epoch_{epoch}.png'
            high_res_path = os.path.join(high_res_dir, high_res_filename)
            img_pil.save(high_res_path)
            print(f"Saved high resolution image: {high_res_path}")
            
            # Also save a comparison with the original low-res reconstruction
            if epoch == eval_epochs[-1]:  # Save comparison for the last epoch
                # Get the original low-res reconstruction for comparison
                coords_lr = test_data['coords'].to(device)
                output_pred_lr = SINR(coords_lr, dino=dino, img=gt_img)
                if "moe" in cfg['MODEL']['model_name']:
                    recon_img_lr = output_pred_lr['selected_nonmanifold_pnts_pred'].detach().cpu().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1], -1)
                else:
                    recon_img_lr = output_pred_lr['nonmanifold_pnts_pred'].permute(0, 2, 1).detach().cpu().numpy().reshape(test_set.sidelength[0], test_set.sidelength[1], -1)
                
                recon_img_lr_uint8 = convert_to_uint8(recon_img_lr)
                if recon_img_lr_uint8.shape[-1] == 1:
                    img_lr_pil = Image.fromarray(recon_img_lr_uint8.squeeze(), mode='L')
                else:
                    img_lr_pil = Image.fromarray(recon_img_lr_uint8, mode='RGB')
                
                # Resize low-res to high-res for comparison
                img_lr_resized = img_lr_pil.resize((high_res_width, high_res_height), Image.NEAREST)
                comparison_filename = f'comparison_low_res_{args.image_id}_{high_res_width}x{high_res_height}_epoch_{epoch}.png'
                comparison_path = os.path.join(high_res_dir, comparison_filename)
                img_lr_resized.save(comparison_path)
                print(f"Saved comparison (low-res upscaled): {comparison_path}")
                
                # Load and save the original high-resolution image for comparison
                if args.compare_with_original:
                    try:
                        original_img_path = os.path.join(cfg['DATA']['dataset_path'], args.image_id)
                        if os.path.exists(original_img_path):
                            original_img = Image.open(original_img_path)
                            # Resize original to match high-res output size
                            original_resized = original_img.resize((high_res_width, high_res_height), Image.LANCZOS)
                            original_filename = f'original_{args.image_id}_{high_res_width}x{high_res_height}.png'
                            original_path = os.path.join(high_res_dir, original_filename)
                            original_resized.save(original_path)
                            print(f"Saved original image (resized): {original_path}")
                            
                            # Calculate and save PSNR comparison
                            # Convert images to numpy arrays for PSNR calculation
                            high_res_np = np.array(img_pil).astype(np.float32)
                            original_np = np.array(original_resized).astype(np.float32)
                            
                            # Calculate MSE and PSNR
                            mse = np.mean((high_res_np - original_np) ** 2)
                            if mse == 0:
                                psnr = float('inf')
                            else:
                                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                            
                            print(f"PSNR between high-res reconstruction and original: {psnr:.2f} dB")
                            
                            # Save PSNR to file
                            psnr_filename = f'psnr_comparison_{args.image_id}_{high_res_width}x{high_res_height}_epoch_{epoch}.txt'
                            psnr_path = os.path.join(high_res_dir, psnr_filename)
                            with open(psnr_path, 'w') as f:
                                f.write(f"High-resolution reconstruction vs Original image\n")
                                f.write(f"Resolution: {high_res_width}x{high_res_height}\n")
                                f.write(f"Epoch: {epoch}\n")
                                f.write(f"MSE: {mse:.6f}\n")
                                f.write(f"PSNR: {psnr:.2f} dB\n")
                            print(f"Saved PSNR comparison: {psnr_path}")
                        else:
                            print(f"Original image not found at: {original_img_path}")
                    except Exception as e:
                        print(f"Error loading original image: {e}")

    # save the evaluation metrics to a file
    eval_metrics_filename = os.path.join(args.logdir, 'eval_metrics.pickle')
    with open(eval_metrics_filename, 'wb') as f:
        pickle.dump(eval_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(eval_metrics)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Testing RGB MoE INR')
    parser.add_argument('--config', default='../configs/config_RGB.yaml', type=str, help='config file')
    parser.add_argument('--logdir', default='./log/debug_rgb', type=str, help='path to log firectory')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index to use')
    parser.add_argument('--image_id', default='kodim19.png', type=str, help='shape to load')
    parser.add_argument('--high_resolution', action='store_true', help='generate high resolution images')
    parser.add_argument('--high_res_height', default=None, type=int, help='high resolution height (default: 4x training resolution)')
    parser.add_argument('--high_res_width', default=None, type=int, help='high resolution width (default: 4x training resolution)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size for high resolution processing')
    parser.add_argument('--compare_with_original', action='store_true', help='compare high-res reconstruction with original image')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)


