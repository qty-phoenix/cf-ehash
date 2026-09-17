import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from ssim import SSIM
import time
from datetime import datetime
from PIL import Image
from model import LearnPose, OfficialNerf_siren_fr, encode_position, sample_from_matrix, OfficialNerf_siren, haar_wavelet_encode
from newdataset import CustomDatasetCompatible
from model import eulerAnglesToRotationMatrix_torch
from finer import FinerLayer,OfficialNerf_finer
from wavenerf import OfficialNerf_wave
from wavenerf_point import OfficialNerf_wave_point
import torch.fft


def compute_frequency_domain_errors(real_img, pred_img, save_path=None):
    """
    计算并可视化真实图像和预测图像的频域差异
    Args:
        real_img: 真实图像张量 [1, 1, H, W] (范围[0,1])
        pred_img: 预测图像张量 [1, 1, H, W] (范围[0,1])
        save_path: 频域对比图的保存路径（可选）
    Returns:
        low_freq_error: 低频误差（均值）
        high_freq_error: 高频误差（均值）
    """
    # 转换为灰度张量（如果尚未是单通道）
    real = real_img.squeeze().cpu().numpy()  # [H, W]
    pred = pred_img.squeeze().cpu().numpy()  # [H, W]

    # 傅里叶变换
    real_fft = torch.fft.fft2(torch.tensor(real))
    pred_fft = torch.fft.fft2(torch.tensor(pred))

    # 计算幅度谱
    real_mag = torch.abs(torch.fft.fftshift(real_fft))
    pred_mag = torch.abs(torch.fft.fftshift(pred_fft))

    # 分离低频和高频（定义中心区域为低频）
    H, W = real.shape
    radius = 0.1 * min(H, W)  # 低频区域半径
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_h, center_w = H // 2, W // 2
    mask = ((x - center_w) ** 2 + (y - center_h) ** 2) <= radius ** 2

    # 计算误差
    low_freq_error = torch.mean(torch.abs(real_mag * mask - pred_mag * mask))
    high_freq_error = torch.mean(torch.abs(real_mag * (~mask) - pred_mag * (~mask)))

    # 可视化频域对比
    if save_path is not None:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(real, cmap='gray')
        plt.title('Real Image')

        plt.subplot(132)
        plt.imshow(pred, cmap='gray')
        plt.title('Pred Image')

        plt.subplot(133)
        plt.imshow(torch.log(real_mag + 1e-6) - torch.log(pred_mag + 1e-6), cmap='jet')
        plt.title('Frequency Error (Real - Pred)')
        plt.colorbar()
        plt.savefig(save_path)
        plt.close()

    return low_freq_error.item(), high_freq_error.item()
def decompose_mat4x4_to_rt(mat):
    """
    输入: 4x4 变换矩阵 mat (Tensor)
    输出: R (3x3 旋转矩阵), t (3,)
    """
    assert mat.shape == (4, 4), "输入矩阵必须是 4x4 的"
    R = mat[:3, :3]
    t = mat[:3, 3]
    return R, t

def assemble_rt_to_mat4x4(R, t):
    """
    将旋转矩阵 R (3x3) 和平移向量 t (3,) 拼成一个 4x4 齐次变换矩阵
    """
    mat = torch.eye(4, device=R.device)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


# 定义固定的 matTToC 和 matPToR（uvf1的）
matTToC = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]).cuda()

matPToR = torch.tensor([
    [0.02512228, 0.99775797, -0.06203075, 4.26943016],
    [0.99962986, -0.02442457, 0.01198075, 0.026],
    [0.01043881, -0.06230878, -0.99800235, 0.444922],
    [0.0, 0.0, 0.0, 1.0]
]).cuda()

if __name__ == '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    # Hyperparameters
    N_EPOCH = 5001
    EVAL_INTERVAL = 100
    trainable = False

    # 数据路径
    image_folder = './mri2'#'./brain2''./uvfdata2'
    pose_file = './mri2.xlsx'#'./brain2.xlsx''./uvfdata2.xlsx'
    crop_size = 160 # 480 目前都是裁成正方形的，uvf那个我看两边也有纯黑色边缘，裁成480*480

    # 加载数据集
    training_set = CustomDatasetCompatible(image_folder, pose_file, mode='train', crop_size=crop_size)
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4, 'drop_last': False}
    training_generator = data.DataLoader(training_set, **params)

    # 构建参考网格 (3, H, W) 后续直接对这个参考网格进行平移旋转得到每个像素实际位置
    _, sample_img, _, _ = training_set[0]
    _, H, W = sample_img.shape
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    zz = torch.zeros_like(xx)
    grid_ref = torch.stack([xx, yy, zz], dim=0).float().cuda()
    # print("变换前grid范围:", grid_ref.min().item(), grid_ref.max().item())
    # 初始化模型 下面这一行是用于对帧位置的学习和修正，我后续的实验里这边都设置为false不学习这个
    pose_param_net = LearnPose(training_set._overall_store(), learn_R=trainable, learn_t=trainable).cuda()
    # nerf_model = OfficialNerf_siren(pos_in_dims=42, D=256).cuda()
    # nerf_model = OfficialNerf_siren_fr(pos_in_dims=63, D=256).cuda()
    nerf_model = OfficialNerf_finer(pos_in_dims=42, D=256).cuda()#weizhibianma
    # nerf_model = OfficialNerf_wave_point(pos_in_dims=42, D=256).cuda()
    ssim_loss = SSIM(window_size=5)

    opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
    # opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)关于pose的学习部分舍去

    scheduler_nerf = optim.lr_scheduler.MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 50)), gamma=0.9954)
    # scheduler_pose = optim.lr_scheduler.MultiStepLR(opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)关于pose的部分舍去

    # 日志
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./log_{start_time}.txt"
    with open(log_file, 'w') as f:
        f.write("Epoch\tLoss\tElapsed Time\n")

    ssim_log_file = f"./ssim_log_{start_time}.txt"
    with open(ssim_log_file, 'w') as f:
        f.write("Epoch\tAverage SSIM\n")

    for e in range(N_EPOCH):
        torch.cuda.synchronize()
        start_epoch = time.time()

        nerf_model.train()
        # pose_param_net.train()

        loss_epoch = []
        for (local_key, local_img, rot_ground, trans_ground) in training_generator:
            local_img = local_img.to(dtype=torch.float).cuda()
            rot_ground = rot_ground.cuda()
            trans_ground = trans_ground.cuda()
            _, _, H, W = local_img.size()
            key = (local_key.numpy())

            pos_enc = []
            for i, k in enumerate(key):
                # 旋转参数读取
                rot = eulerAnglesToRotationMatrix_torch(rot_ground[i]).cuda()
                # 平移参数读取
                trans = trans_ground[i]
                # 下面注释部分是uvf数据会用到的部分
                # matRToT = assemble_rt_to_mat4x4(rot, trans)
                # 计算总的变换矩阵 matPToC = matTToC * matRToT * matPToR
                # matPToC = matTToC @ matRToT @ matPToR
                # rot, trans = decompose_mat4x4_to_rt(matPToC)

                # 通过变换矩阵变换网格grid_ref就是前面初始化的那个原始网格
                grid = sample_from_matrix(grid_ref, rot, trans)
                # 位置编码
                # 计算每个样本的独立归一化因子
                min_val = grid.min()
                max_val = grid.max()
                normalized_grid = 2 * (grid - min_val) / (max_val - min_val) - 1  # 归一化到[-1,1]
                # print("变换后grid范围:", normalized_grid.min().item(), normalized_grid.max().item())
                # # 修改后的归一化方式（确保在[-1,1]范围内）
                # normalized_grid = (grid) /H  # 对于H=160的情况
                # print("归一化后范围:", normalized_grid.min().item(), normalized_grid.max().item())
                p = haar_wavelet_encode(normalized_grid, levels=4, inc_input=True)
                # p = encode_position(normalized_grid, levels=10, inc_input=True)
                pos_enc.append(p)
            pos_enc = torch.stack(pos_enc, dim=0)
            # 1 480 480 63
            pred = nerf_model(pos_enc)

            # 复合损失计算
            alpha = 0.5  # SSIM 权重
            beta = 0.3  # MSE 权重
            gamma = 0.2  # 高频损失权重

            pred_img = pred.squeeze(-1).unsqueeze(1)  # [B, 1, H, W]
            ssim_val = ssim_loss(pred_img, local_img)
            mse_val = F.mse_loss(pred_img, local_img)

            # 计算高频损失（通过拉普拉斯算子提取边缘）
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                            dtype=torch.float32).view(1, 1, 3, 3).cuda()
            real_edges = F.conv2d(local_img, laplacian_kernel, padding=1)
            pred_edges = F.conv2d(pred_img, laplacian_kernel, padding=1)
            high_freq_loss = F.l1_loss(pred_edges, real_edges)

            loss = -alpha * ssim_val + beta * mse_val + gamma * high_freq_loss

            loss.backward()

            opt_nerf.step()
            # opt_pose.step()  pose学习部分舍去
            opt_nerf.zero_grad()
            # opt_pose.zero_grad()   pose学习部分舍去

            loss_epoch.append(loss)

        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        loss_epoch_mean = torch.stack(loss_epoch).mean().item()

        with open(log_file, 'a') as f:
            f.write(f"{e}\t{loss_epoch_mean:.6f}\t{elapsed:.2f}\n")

        if e % 1 == 0:
            print(e, loss_epoch_mean, elapsed)

        scheduler_nerf.step()
        # scheduler_pose.step()

        # 可视化 & 保存
        if (e + 1) % EVAL_INTERVAL == 0:
            output_folder = f"./output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_folder, exist_ok=True)
            with torch.no_grad():
                fig = plt.figure(figsize=(10, 5))
                ax_img = fig.add_subplot(121)
                ax_pred = fig.add_subplot(122)
                gif_frames = []

                test_set = CustomDatasetCompatible(image_folder, pose_file, mode='test', crop_size=crop_size)
                test_store = test_set._overall_store()

                test_ssim_list = []
                low_freq_errors = []  # NEW: 存储低频误差
                high_freq_errors = []  # NEW: 存储高频误差

                for i, entry in test_store.items():

                    img = Image.open(entry['img_path']).convert('L')
                    # 对测试集进行裁剪 640*480->480*480
                    width, height = img.size
                    left = (width - crop_size) // 2
                    top = (height - crop_size) // 2
                    right = (width + crop_size) // 2
                    bottom = (height + crop_size) // 2
                    img = img.crop((left, top, right, bottom))  # 中心裁剪
                    img = np.array(img.resize((H, W)), dtype=np.float32) / 255.0  # 调整大小并归一化

                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
                    # 获取测试帧空间位置过程
                    rot = entry['rot_ground']
                    trans = entry['trans_ground']
                    rot = eulerAnglesToRotationMatrix_torch(torch.tensor(rot, dtype=torch.float32).cuda())
                    trans = torch.tensor(trans, dtype=torch.float32).cuda()
                    grid = sample_from_matrix(grid_ref, rot, trans)
                    # 计算每个样本的独立归一化因子
                    min_val = grid.min()
                    max_val = grid.max()
                    normalized_grid = 2 * (grid - min_val) / (max_val - min_val) - 1  # 归一化到[-1,1]
                    # 位置编码
                    pos_enc = haar_wavelet_encode(normalized_grid, levels=4, inc_input=True)
                    # pos_enc = encode_position(normalized_grid, levels=10, inc_input=True)
                    # 预测
                    pred = nerf_model(pos_enc).detach().squeeze(-1).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
                    # pred = torch.sigmoid(pred)# 后加的
                    ssim_val = ssim_loss(pred, img_tensor).item()
                    test_ssim_list.append(ssim_val)

                    # 计算频域误差并保存可视化
                    freq_save_path = os.path.join(output_folder, f"epoch_{e + 1}_freq_{i}.png")
                    low_err, high_err = compute_frequency_domain_errors(img_tensor, pred, freq_save_path)
                    low_freq_errors.append(low_err)
                    high_freq_errors.append(high_err)

                    ax_pred.cla()
                    ax_pred.imshow(pred.squeeze().cpu().numpy(), cmap='gray')
                    ax_pred.axis('off')

                    ax_img.cla()
                    ax_img.imshow(img, cmap='gray')
                    ax_img.set_title(i)
                    ax_img.axis('off')

                    plt.savefig(os.path.join(output_folder, f"epoch_{e + 1}_image_{i}.png"))
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    gif_frames.append(image)

                # 写入平均 SSIM 到日志
                avg_ssim = np.mean(test_ssim_list)
                avg_low_freq = np.mean(low_freq_errors)  # NEW
                avg_high_freq = np.mean(high_freq_errors)  # NEW
                with open(ssim_log_file, 'a') as f:
                    f.write(f"{e}\t{avg_ssim:.6f}\t{avg_low_freq:.6f}\t{avg_high_freq:.6f}\n")  # NEW: 添加频域误差
                    # f.write(f"{e}\t{avg_ssim:.6f}\n")

                plt.close('all')

                gif_path = os.path.join(output_folder, f"epoch_{e}.gif")
                imageio.mimsave(gif_path, gif_frames, duration=125)

                model_path = os.path.join(output_folder, f"epoch_{e}.pth")
                torch.save(nerf_model.state_dict(), model_path)
