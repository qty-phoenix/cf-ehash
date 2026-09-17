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
from newdataset import CustomDatasetCompatible
# 假设这些是您原有的模型组件
from model import LearnPose, OfficialNerf_siren, encode_position, sample_from_matrix, eulerAnglesToRotationMatrix_torch


# 多了反向传播时候的加权

def decompose_mat4x4_to_rt(mat):
    """输入: 4x4 变换矩阵 mat (Tensor) 输出: R (3x3 旋转矩阵), t (3,)"""
    assert mat.shape == (4, 4), "输入矩阵必须是 4x4 的"
    R = mat[:3, :3]
    t = mat[:3, 3]
    return R, t


def assemble_rt_to_mat4x4(R, t):
    """将旋转矩阵 R (3x3) 和平移向量 t (3,) 拼成一个 4x4 齐次变换矩阵"""
    mat = torch.eye(4, device=R.device)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def load_balancing_loss(gate_weights, eps=1e-8):
    """
    计算负载均衡损失，确保专家利用率均衡
    gate_weights: [B*N, num_experts] 门控权重
    """
    # 计算每个专家的使用概率（批量平均）
    expert_usage = gate_weights.mean(dim=0)  # [num_experts]
    # 计算负载均衡损失（鼓励均匀分布）
    balance_loss = torch.std(expert_usage) / (expert_usage.mean() + eps)
    return balance_loss


def soft_moe_loss(expert_outputs, gate_weights, gt_values):
    """
    Hinton的Soft Mixture of Experts损失
    expert_outputs: [B*H*W, num_experts, 1] 各专家输出
    gate_weights: [B*H*W, num_experts] 门控权重
    gt_values: [B*H*W, 1] 真实值
    """
    # 计算每个专家的平方误差
    delta = (expert_outputs - gt_values.unsqueeze(1)) ** 2  # [B*H*W, num_experts, 1]
    delta = delta.squeeze(-1)  # [B*H*W, num_experts]

    # 计算Soft MoE损失
    weighted_exp = gate_weights * torch.exp(-0.5 * delta)
    soft_moe_loss = -torch.log(weighted_exp.sum(dim=-1) + 1e-8).mean()

    return soft_moe_loss


def entropy_regularization(gate_weights, eps=1e-8):
    """
    熵正则化：鼓励门控分布更尖锐（减少不确定性）
    gate_weights: [B*H*W, num_experts] 门控权重
    """
    entropy = -torch.sum(gate_weights * torch.log(gate_weights + eps), dim=-1)
    return entropy.mean()


class MoENeRF(nn.Module):
    """
    同构多专家NeRF模型
    所有专家结构相同，参数不同，通过门控网络自动分配
    """

    def __init__(self, num_experts=4, pos_in_dims=63, D=128, hidden_dim=64):
        super(MoENeRF, self).__init__()
        self.num_experts = num_experts
        self.pos_in_dims = pos_in_dims

        # 同构专家网络列表 - 使用差异化初始化
        self.experts = nn.ModuleList([
            self._build_single_expert(pos_in_dims, D, expert_idx=i) for i in range(num_experts)
        ])

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(pos_in_dims, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

        # 初始化门控网络权重
        self._initialize_gate_weights()

        # 保存门控权重用于损失计算
        self.gate_weights = None

    def _build_single_expert(self, pos_in_dims, D, expert_idx):
        """构建单个专家网络并差异化初始化"""
        expert = nn.Sequential(
            nn.Linear(pos_in_dims, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 1),
            nn.Sigmoid()  # 输出在 [0, 1] 范围
        )

        # 为每个专家应用不同的初始化策略
        self._init_expert_differently(expert, expert_idx)
        return expert

    def _init_expert_differently(self, expert, expert_idx):
        """为不同专家设置不同的初始化策略"""
        with torch.no_grad():
            for layer_idx, layer in enumerate(expert):
                if isinstance(layer, nn.Linear):
                    # 根据专家索引使用不同的初始化方法
                    if expert_idx == 0:
                        # Xavier初始化
                        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    elif expert_idx == 1:
                        # Kaiming初始化
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    elif expert_idx == 2:
                        # 正交初始化
                        nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                    else:  # expert_idx == 3
                        # 正态分布初始化，使用不同的标准差
                        nn.init.normal_(layer.weight, mean=0.0, std=0.1 * (expert_idx + 1))

                    # 偏置项初始化
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                        # 为某些专家添加小的偏置变化
                        if expert_idx in [1, 3]:
                            nn.init.normal_(layer.bias, mean=0.0, std=0.01)

    def _initialize_gate_weights(self):
        """初始化门控网络，使其初始时对所有专家平等"""
        with torch.no_grad():
            # 最后一层初始化为接近均匀分布
            gate_final_layer = self.gate[-1]
            nn.init.constant_(gate_final_layer.weight, 0.0)
            nn.init.constant_(gate_final_layer.bias, 0.0)

    def forward(self, x, return_gate_weights=False):
        """
        x: 输入位置编码 [B, H, W, C] 或 [B, N, C]
        return_gate_weights: 是否返回门控权重
        """
        batch_size, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # [B*H*W, C]

        # 门控决策
        gate_logits = self.gate(x_flat)  # [B*H*W, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        self.gate_weights = gate_weights  # 保存用于损失计算

        # 收集所有专家输出并按门控权重比例回传梯度
        output = 0
        expert_outputs = []

        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)  # [B*H*W, 1]
            expert_outputs.append(expert_out)  # 保存专家输出用于Soft MoE损失

            # 获取当前专家的门控权重 [B*H*W, 1]
            gate_weight_i = gate_weights[:, i].unsqueeze(-1)  # [B*H*W, 1]

            # 关键修改：按门控权重比例回传梯度到专家
            # 使用detach实现梯度缩放：
            # - expert_out * gate_weight_i.detach(): 专家输出按权重缩放，梯度只回传到专家
            # - expert_out.detach() * gate_weight_i: 权重部分，梯度只回传到门控网络
            scaled_output = (expert_out * gate_weight_i.detach() +
                             expert_out.detach() * gate_weight_i)

            output = output + scaled_output

        if return_gate_weights:
            # 返回专家输出用于Soft MoE损失
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [B*H*W, num_experts, 1]
            return output.reshape(batch_size, H, W, 1), gate_weights, expert_outputs

        # 恢复原始形状 [B, H, W, 1]
        output = output.reshape(batch_size, H, W, 1)
        return output

    def get_expert_usage(self):
        """获取各专家的使用统计"""
        if self.gate_weights is None:
            return None
        usage = self.gate_weights.mean(dim=0)  # [num_experts]
        return usage.detach().cpu().numpy()

    def check_expert_differences(self):
        """检查专家之间的差异（用于调试）"""
        differences = []
        for i in range(1, self.num_experts):
            # 比较第一个专家和其他专家的权重差异
            weight_diff = torch.abs(self.experts[0][0].weight - self.experts[i][0].weight).mean().item()
            differences.append(weight_diff)
        return differences


# 定义固定的变换矩阵
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
    # 设置随机种子
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)

    # Hyperparameters
    N_EPOCH = 5001
    EVAL_INTERVAL = 100
    NUM_EXPERTS = 4  # 专家数量
    LAMBDA_BALANCE = 0.01  # 负载均衡损失权重
    LAMBDA_SOFT_MOE = 0.1  # Soft MoE损失权重
    LAMBDA_ENTROPY = 0.01  # 熵正则化权重

    # 数据路径
    image_folder = './uvfdata2'  # './mri2'
    pose_file = './uvfdata2.xlsx'  # './mri.xlsx'
    crop_size = 480

    # 加载数据集
    training_set = CustomDatasetCompatible(image_folder, pose_file, mode='train', crop_size=crop_size)
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4, 'drop_last': False}
    training_generator = data.DataLoader(training_set, **params)

    # 构建参考网格
    _, sample_img, _, _ = training_set[0]
    _, H, W = sample_img.shape
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    zz = torch.zeros_like(xx)
    grid_ref = torch.stack([xx, yy, zz], dim=0).float().cuda()

    # 初始化多专家模型
    pose_param_net = LearnPose(training_set._overall_store(), learn_R=False, learn_t=False).cuda()
    nerf_model = MoENeRF(num_experts=NUM_EXPERTS, pos_in_dims=63, D=128).cuda()

    # 检查专家初始化差异
    differences = nerf_model.check_expert_differences()
    print(f"专家初始化差异: {differences}")

    ssim_loss = SSIM(window_size=5)

    # 优化器 - 分别设置专家和门控网络的学习率
    opt_nerf = optim.Adam([
        {'params': nerf_model.experts.parameters(), 'lr': 0.001},
        {'params': nerf_model.gate.parameters(), 'lr': 0.0005}  # 门控网络学习率稍低
    ])

    scheduler_nerf = optim.lr_scheduler.MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 50)), gamma=0.9954)

    # 日志文件
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./moe_log_{start_time}.txt"
    expert_usage_file = f"./expert_usage_{start_time}.txt"

    with open(log_file, 'w') as f:
        f.write(
            "Epoch\tTotal_Loss\tSSIM_Loss\tMSE_Loss\tHF_Loss\tBalance_Loss\tSoftMoE_Loss\tEntropy_Loss\tElapsed_Time\n")

    with open(expert_usage_file, 'w') as f:
        f.write("Epoch\t" + "\t".join([f"Expert_{i}" for i in range(NUM_EXPERTS)]) + "\n")

    # 训练循环
    for epoch in range(N_EPOCH):
        torch.cuda.synchronize()
        start_time_epoch = time.time()

        nerf_model.train()
        loss_epoch = []
        ssim_loss_epoch = []
        mse_loss_epoch = []
        hf_loss_epoch = []
        balance_loss_epoch = []
        soft_moe_loss_epoch = []
        entropy_loss_epoch = []

        for (local_key, local_img, rot_ground, trans_ground) in training_generator:
            local_img = local_img.to(dtype=torch.float).cuda()
            rot_ground = rot_ground.cuda()
            trans_ground = trans_ground.cuda()
            key = local_key.numpy()

            pos_enc = []
            for i, k in enumerate(key):
                rot = eulerAnglesToRotationMatrix_torch(rot_ground[i]).cuda()
                trans = trans_ground[i]

                # 变换网格
                grid = sample_from_matrix(grid_ref, rot, trans)
                min_val = grid.min()
                max_val = grid.max()
                normalized_grid = 2 * (grid - min_val) / (max_val - min_val) - 1

                p = encode_position(normalized_grid, levels=10, inc_input=True)
                pos_enc.append(p)

            pos_enc = torch.stack(pos_enc, dim=0)  # [1, H, W, 63]

            # 前向传播（获取门控权重和专家输出）
            pred, gate_weights, expert_outputs = nerf_model(pos_enc, return_gate_weights=True)

            # 计算各项损失
            pred_img = pred.squeeze(-1).unsqueeze(1)  # [B, 1, H, W]
            ssim_val = ssim_loss(pred_img, local_img)
            mse_val = F.mse_loss(pred_img, local_img)

            # 高频损失
            laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                            dtype=torch.float32).view(1, 1, 3, 3).cuda()
            real_edges = F.conv2d(local_img, laplacian_kernel, padding=1)
            pred_edges = F.conv2d(pred_img, laplacian_kernel, padding=1)
            high_freq_loss = F.l1_loss(pred_edges, real_edges)

            # 负载均衡损失
            balance_loss = load_balancing_loss(gate_weights)

            # Soft MoE损失
            gt_flat = local_img.reshape(-1, 1)  # [B*H*W, 1]
            soft_moe_loss_val = soft_moe_loss(expert_outputs, gate_weights, gt_flat)

            # 熵正则化
            entropy_loss_val = entropy_regularization(gate_weights)

            # 总损失
            total_loss = (-ssim_val +
                          0.3 * mse_val +
                          0.2 * high_freq_loss +
                          LAMBDA_BALANCE * balance_loss +
                          LAMBDA_SOFT_MOE * soft_moe_loss_val +
                          LAMBDA_ENTROPY * entropy_loss_val)

            # 反向传播
            total_loss.backward()
            opt_nerf.step()
            opt_nerf.zero_grad()

            # 记录损失
            loss_epoch.append(total_loss.item())
            ssim_loss_epoch.append(-ssim_val.item())
            mse_loss_epoch.append(mse_val.item())
            hf_loss_epoch.append(high_freq_loss.item())
            balance_loss_epoch.append(balance_loss.item())
            soft_moe_loss_epoch.append(soft_moe_loss_val.item())
            entropy_loss_epoch.append(entropy_loss_val.item())

        # 记录专家使用情况
        expert_usage = nerf_model.get_expert_usage()
        if expert_usage is not None:
            with open(expert_usage_file, 'a') as f:
                f.write(f"{epoch}\t" + "\t".join([f"{usage:.6f}" for usage in expert_usage]) + "\n")

        # 计算平均损失
        avg_total_loss = np.mean(loss_epoch)
        avg_ssim_loss = np.mean(ssim_loss_epoch)
        avg_mse_loss = np.mean(mse_loss_epoch)
        avg_hf_loss = np.mean(hf_loss_epoch)
        avg_balance_loss = np.mean(balance_loss_epoch)
        avg_soft_moe_loss = np.mean(soft_moe_loss_epoch)
        avg_entropy_loss = np.mean(entropy_loss_epoch)

        # 记录日志
        torch.cuda.synchronize()
        elapsed = time.time() - start_time_epoch

        with open(log_file, 'a') as f:
            f.write(
                f"{epoch}\t{avg_total_loss:.6f}\t{avg_ssim_loss:.6f}\t{avg_mse_loss:.6f}\t{avg_hf_loss:.6f}\t{avg_balance_loss:.6f}\t{avg_soft_moe_loss:.6f}\t{avg_entropy_loss:.6f}\t{elapsed:.2f}\n")

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}: Total={avg_total_loss:.4f}, SSIM={avg_ssim_loss:.4f}, MSE={avg_mse_loss:.4f}, "
                f"HF={avg_hf_loss:.4f}, Balance={avg_balance_loss:.4f}, SoftMoE={avg_soft_moe_loss:.4f}, "
                f"Entropy={avg_entropy_loss:.4f}, Time={elapsed:.2f}s")
            if expert_usage is not None:
                print(f"Expert Usage: {expert_usage}")

        scheduler_nerf.step()

        # 评估和保存
        if (epoch + 1) % EVAL_INTERVAL == 0:
            output_folder = f"./moe_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_folder, exist_ok=True)

            with torch.no_grad():
                nerf_model.eval()
                test_set = CustomDatasetCompatible(image_folder, pose_file, mode='test', crop_size=crop_size)
                test_store = test_set._overall_store()

                test_ssim_list = []
                gif_frames = []

                fig = plt.figure(figsize=(15, 5))

                for i, entry in test_store.items():
                    # 加载测试图像
                    img = Image.open(entry['img_path']).convert('L')
                    width, height = img.size
                    left = (width - crop_size) // 2
                    top = (height - crop_size) // 2
                    right = (width + crop_size) // 2
                    bottom = (height + crop_size) // 2
                    img = img.crop((left, top, right, bottom))
                    img = np.array(img.resize((H, W)), dtype=np.float32) / 255.0

                    # 确保img_tensor是4维的 [1, 1, H, W]
                    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

                    # 获取测试帧位置
                    rot = entry['rot_ground']
                    trans = entry['trans_ground']
                    rot = eulerAnglesToRotationMatrix_torch(torch.tensor(rot, dtype=torch.float32).cuda())
                    trans = torch.tensor(trans, dtype=torch.float32).cuda()

                    grid = sample_from_matrix(grid_ref, rot, trans)
                    min_val = grid.min()
                    max_val = grid.max()
                    normalized_grid = 2 * (grid - min_val) / (max_val - min_val) - 1

                    pos_enc = encode_position(normalized_grid, levels=10, inc_input=True)

                    # 修改这里：确保输入是4维的 [1, H, W, C]
                    pos_enc_input = pos_enc.unsqueeze(0)  # [1, H, W, C]

                    # 前向传播
                    pred = nerf_model(pos_enc_input)  # [1, H, W, 1]

                    # 调整pred的维度为 [1, 1, H, W] 以匹配SSIM的输入要求
                    pred = pred.permute(0, 3, 1, 2)  # [1, 1, H, W]

                    # 确保pred和img_tensor维度一致
                    if pred.shape != img_tensor.shape:
                        # 如果维度不匹配，进行调整
                        pred = pred.reshape(img_tensor.shape)

                    # 计算SSIM
                    ssim_val = ssim_loss(pred, img_tensor).item()
                    test_ssim_list.append(ssim_val)

                    # 可视化
                    plt.clf()
                    plt.subplot(131)
                    plt.imshow(img, cmap='gray')
                    plt.title(f'Ground Truth - {i}')
                    plt.axis('off')

                    plt.subplot(132)
                    # 将pred转换回 [H, W] 格式用于显示
                    pred_display = pred.squeeze().cpu().numpy()
                    plt.imshow(pred_display, cmap='gray')
                    plt.title(f'Predicted - SSIM: {ssim_val:.4f}')
                    plt.axis('off')

                    plt.subplot(133)
                    # 显示专家使用热力图
                    usage = nerf_model.get_expert_usage()
                    if usage is not None:
                        plt.bar(range(NUM_EXPERTS), usage)
                        plt.title('Expert Usage')
                        plt.xlabel('Expert Index')
                        plt.ylabel('Usage Probability')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f"epoch_{epoch + 1}_test_{i}.png"))

                    # 添加到GIF
                    fig.canvas.draw()
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    gif_frames.append(image)

                # 保存模型和生成GIF
                avg_ssim = np.mean(test_ssim_list)
                print(f"Test Average SSIM: {avg_ssim:.4f}")

                model_path = os.path.join(output_folder, f"moe_model_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': nerf_model.state_dict(),
                    'optimizer_state_dict': opt_nerf.state_dict(),
                    'loss': avg_total_loss,
                    'ssim': avg_ssim
                }, model_path)

                if gif_frames:
                    gif_path = os.path.join(output_folder, f"test_results_epoch_{epoch + 1}.gif")
                    imageio.mimsave(gif_path, gif_frames, duration=125)

                plt.close('all')