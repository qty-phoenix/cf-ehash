# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:05:43 2021

@author: pemb5552
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from skimage.feature import match_template
# 下面为重参数化，mlp网络变成可学习网络×傅里叶基
import torch
import torch.nn as nn
import math
import numpy as np


class SinFRLayer(nn.Module):
    def __init__(self, in_features, out_features, high_freq_num=10, low_freq_num=0, phi_num=2, alpha=0.05, w0=30.0,
                 is_first=False):
        """
        SIREN + 傅里叶重参数化层
        Args:
            in_features:    输入维度
            out_features:   输出维度
            high_freq_num:  高频基数量 (默认5)
            low_freq_num:   低频基数量 (默认3)
            phi_num:        相位数量 (默认2)
            alpha:          基幅值缩放 (默认0.01)
            w0:             周期激活频率 (默认30.0)
            is_first:       是否是第一层 (影响初始化)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.high_freq_num = high_freq_num
        self.low_freq_num = low_freq_num
        self.phi_num = phi_num
        self.alpha = alpha
        self.w0 = w0
        self.is_first = is_first

        # 初始化傅里叶基和系数
        self.bases = self._init_bases()
        self.lamb = self._init_lamb()
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

    def _init_bases(self):
        """生成低频+高频傅里叶基"""
        phi_set = np.array([2 * math.pi * i / self.phi_num for i in range(self.phi_num)])
        high_freq = np.array([i + 1 for i in range(self.high_freq_num)])
        low_freq = np.array([(i + 1) / self.low_freq_num for i in range(self.low_freq_num)])

        # 确定采样区间
        T_max = 2 * math.pi / low_freq[0] if len(low_freq) > 0 else 2 * math.pi / min(high_freq)
        points = np.linspace(-T_max / 2, T_max / 2, self.in_features)

        # 构建基矩阵 [num_bases, in_features]
        num_bases = (self.high_freq_num + self.low_freq_num) * self.phi_num
        bases = torch.zeros(num_bases, self.in_features)

        idx = 0
        for freq in low_freq:
            for phi in phi_set:
                bases[idx] = torch.tensor([math.cos(freq * x + phi) for x in points])
                idx += 1
        for freq in high_freq:
            for phi in phi_set:
                bases[idx] = torch.tensor([math.cos(freq * x + phi) for x in points])
                idx += 1

        return nn.Parameter(self.alpha * bases, requires_grad=False)

    def _init_lamb(self):
        """初始化系数矩阵 (按w0和is_first缩放)"""
        num_bases = (self.high_freq_num + self.low_freq_num) * self.phi_num
        lamb = torch.zeros(self.out_features, num_bases)

        with torch.no_grad():
            for i in range(num_bases):
                base_norm = torch.norm(self.bases[i], p=2)
                if self.is_first:
                    scale = 1 / self.in_features
                else:
                    scale = math.sqrt(6 / num_bases) / (base_norm * self.w0)
                lamb[:, i].uniform_(-scale, scale)

        return nn.Parameter(lamb, requires_grad=True)

    def forward(self, x):
        weight = torch.matmul(self.lamb, self.bases)  # [out_features, in_features]
        output = torch.matmul(x, weight.T) + self.bias.T
        return torch.sin(self.w0 * output)


class OfficialNerf_siren_fr(nn.Module):
    def __init__(self, pos_in_dims, D=256):
        """
        完整版SIREN+FR NeRF模型
        Args:
            pos_in_dims: 位置编码维度 (C*(2L+1))
            D:           隐藏层维度 (默认256)
        """
        super().__init__()
        self.pos_in_dims = pos_in_dims

        # 第一阶段网络 (深层结构)
        self.layers0 = nn.Sequential(
            SinFRLayer(pos_in_dims, D, w0=30., is_first=True),  # 第一层特殊初始化
            SinFRLayer(D, D, w0=1.),
            #SinFRLayer(D, D, w0=1.),
            #SinFRLayer(D, D, w0=1.)
        )

        # 第二阶段网络 (跳跃连接)
        self.layers1 = nn.Sequential(
            SinFRLayer(D + pos_in_dims, D, w0=1.),
            SinFRLayer(D, D, w0=1.),
            #SinFRLayer(D, D, w0=1.),
            #SinFRLayer(D, D, w0=1.)
        )

        # 输出头
        self.fc_feature = nn.Linear(D, D)
        self.img_layers = SinFRLayer(D, D // 2, w0=1.)
        self.fc_img = nn.Linear(D // 2, 1)

        # 初始化偏置 (与原始实现一致)
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc):
        # 第一阶段
        x = self.layers0(pos_enc)  # [H,W,N_sample,D]

        # 跳跃连接
        x = torch.cat([x, pos_enc], dim=-1)  # [H,W,N_sample,D+pos_in_dims]
        x = self.layers1(x)

        # 输出预测
        feat = self.fc_feature(x)
        x = self.img_layers(feat)
        img = self.fc_img(x)  # [H,W,N_sample,1]

        return img


def eulerAnglesToRotationMatrix_torch(theta):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=theta.device)
    one = torch.ones(1, dtype=torch.float32, device=theta.device)
    x0 = torch.cat([ one,    zero,   zero])  # (3, 1)
    x1 = torch.cat([ zero,    torch.cos(theta[0:1]),   -torch.sin(theta[0:1])])  # (3, 1)
    x2 = torch.cat([ zero,    torch.sin(theta[0:1]),   torch.cos(theta[0:1])])  # (3, 1)
    x = torch.stack([x0, x1, x2], dim=0)  # (3, 3)
    
    y0 = torch.cat([ torch.cos(theta[1:2]),    zero,   torch.sin(theta[1:2])])  # (3, 1)
    y1 = torch.cat([ zero,    one,   zero])  # (3, 1)
    y2 = torch.cat([ -torch.sin(theta[1:2]),    zero,   torch.cos(theta[1:2])])  # (3, 1)
    y = torch.stack([y0, y1, y2], dim=0)  # (3, 3)
    
    z0 = torch.cat([ torch.cos(theta[2:3]),   -torch.sin(theta[2:3]),   zero])  # (3, 1)
    z1 = torch.cat([ torch.sin(theta[2:3]),   torch.cos(theta[2:3]),   zero])  # (3, 1)
    z2 = torch.cat([ zero,    zero,   one])  # (3, 1)
    z = torch.stack([z0, z1, z2], dim=0)  # (3, 3)
    
    R = torch.matmul(z, torch.matmul( y, x ))
    
    return R  # (3, 3)



def sample_from_matrix(ref, rot, trans):
    # 转换一下2D网格中各点的3d坐标（旋转＋平移）
    # grid = torch.einsum('imn, ij -> mnj', ref, rot) #HW3
    grid = torch.einsum('jmn, ij -> mni', ref, rot) #HW3
    
    grid_trans = grid+trans.unsqueeze(0).unsqueeze(0)
    
    # grid[40,40,:]
    # grid_trans[40,40,:]
    
    return grid_trans

def sample_from_matrix_point(ref, rot, trans):
    # ref: [1, N, 3]
    # rot: [3, 3]
    # trans: [3]
    grid = torch.matmul(ref, rot.T)  # shape: [1, N, 3]
    grid_trans = grid + trans.view(1, 1, 3)
    return grid_trans


class AdaptivePositionalEncoding(nn.Module):
    def __init__(self, levels=10, inc_input=True):
        """
        改进：为每个频率分量添加可学习权重
        输入输出维度与原始encode_position完全一致
        Args:
            levels: 频率级数L
            inc_input: 是否包含原始输入
        """
        super().__init__()
        self.levels = levels
        self.inc_input = inc_input

        # 为每个sin/cos对和原始输入（如果存在）定义可学习权重
        num_weights = 2 * levels + (1 if inc_input else 0)
        self.weights = nn.Parameter(torch.ones(num_weights))
        self._init_weights()

    def _init_weights(self):
        # 初始化为接近1的值，保留原始编码特性
        nn.init.uniform_(self.weights, 0.9, 1.1)

    def forward(self, input):
        result_list = [input] if self.inc_input else []

        # 生成所有频率分量（与原始函数一致）
        for i in range(self.levels):
            temp = 2.0 ** i * input
            result_list.append(torch.sin(temp))
            result_list.append(torch.cos(temp))

        # 应用可学习权重
        weighted_results = []
        for idx, feature in enumerate(result_list):
            weight = torch.sigmoid(self.weights[idx])  # 约束到[0,1]
            weighted_results.append(weight * feature)

        return torch.cat(weighted_results, dim=-1)
def encode_position(input, levels, inc_input):
    """
    位置编码
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.
    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.
    result_list = [input] if inc_input else []
    for i in range(levels):
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))
def encode_position1(input, levels, inc_input):
    """
    位置编码
    For each scalar, we encode it using a series of sin() and cos() functions with different frequency.
        - With L pairs of sin/cos function, each scalar is encoded to a vector that has 2L elements. Concatenating with
          itself results in 2L+1 elements.
        - With C channels, we get C(2L+1) channels output.
    :param input:   (..., C)            torch.float32
    :param levels:  scalar L            int
    :return:        (..., C*(2L+1))     torch.float32
    """

    # this is already doing 'log_sampling' in the official code.截断一部分的位置编码
    result_list = [input] if inc_input else []
    for i in range(5, levels + 1):  # 修改了range的范围
        temp = 2.0**i * input  # (..., C)
        result_list.append(torch.sin(temp))  # (..., C)
        result_list.append(torch.cos(temp))  # (..., C)

    result_list = torch.cat(result_list, dim=-1)  # (..., C*(2L+1)) The list has (2L+1) elements, with (..., C) shape each.
    return result_list  # (..., C*(2L+1))


import torch
import pywt  # 需要安装PyWavelets库：pip install PyWavelets

import torch
import torch.nn.functional as F
import pywt

def haar_wavelet_encode(input, levels=4, inc_input=True):
    """
    改进版Haar小波编码，解决尺寸不匹配问题
    Args:
        input: [H,W,3] 坐标张量（范围[-1,1]）
        levels: 小波分解层数
        inc_input: 是否附加原始输入
    Returns:
        [H,W,K] 编码特征
    """
    assert input.dim() == 3, f"输入应为[H,W,C]，实际得到{input.shape}"
    H, W, C = input.shape

    # 将输入从[-1,1]映射到[0,1]
    mapped = (input + 1) / 2
    result_list = []

    # 对每个坐标通道单独处理
    for c in range(C):
        # 单通道数据 [H,W]
        channel_data = mapped[..., c].cpu().numpy()

        # 小波分解
        coeffs = pywt.wavedec2(channel_data, 'haar', level=levels)

        features = []
        for i, level_coeffs in enumerate(coeffs):
            if i == 0:  # 近似系数 (cA)
                coeff = torch.from_numpy(level_coeffs).to(input.device)
                coeff = F.interpolate(coeff.view(1, 1, *coeff.shape), size=(H, W), mode='bilinear').squeeze()
                features.append(coeff)
            else:
                # 细节系数 (cH, cV, cD)
                for coeff_type in level_coeffs:
                    coeff = torch.from_numpy(coeff_type).to(input.device)
                    coeff = F.interpolate(coeff.view(1, 1, *coeff.shape), size=(H, W), mode='bilinear').squeeze()
                    features.append(coeff)

        # 合并单通道特征 [H,W,K]
        result_list.append(torch.stack(features, dim=-1))

    # 合并所有通道特征 [H,W,K*C]
    encoded = torch.cat(result_list, dim=-1)

    # 可选：添加原始输入
    if inc_input:
        encoded = torch.cat([input, encoded], dim=-1)

    return encoded


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

class LearnPose(nn.Module):
    def __init__(self, store_dict, learn_R=True, learn_t=True):
        super(LearnPose, self).__init__()
        self.learn_R = learn_R
        # 是否学习旋转参数
        self.learn_t = learn_t
        # 是否学习平移参数
        self.create_r_t(store_dict)
        

    def forward(self, pose_id):
        a = self.r[pose_id]  # (3,) axis-angle
        r = eulerAnglesToRotationMatrix_torch(a)    #()
        
        t = self.t[pose_id]  # (3, )
        return r, t
    
    def create_r_t(self, store_dict):
        r = np.zeros((len(store_dict), 3))
        t = np.zeros((len(store_dict), 3))
        for i in range(len(store_dict)):
            temp = store_dict[i]
            r[i] = temp['rot_ground']
            t[i] = temp['trans_ground']
        self.r = nn.Parameter(torch.from_numpy(r).float(), requires_grad=self.learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.from_numpy(t).float(), requires_grad=self.learn_t)  # (N, 3)




    
class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        # bias为true-》kx+b 否则没有b
        # w0是正弦激活函数的频率参数，默认1.0
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x) 
        out = self.activation(out)
        return out
    

    
class OfficialNerf_siren(nn.Module):
    # OfficialNerf_siren 是一个基于 SIREN 架构的神经网络，用于从编码后的位置信息中生成预测的图像强度值。
    # 输入是编码后的位置信息，形状为 (H, W, N_sample, pos_in_dims)，输出是预测的图像强度值，形状为 (H, W, N_sample, 1)
    def __init__(self, pos_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerf_siren, self).__init__()

        self.pos_in_dims = pos_in_dims

        self.layers0 = nn.Sequential(
            SirenLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            # SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            # SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )
        
        self.layers1 = nn.Sequential(
            SirenLayer(D+pos_in_dims, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            # SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
            # SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        # self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.img_layers = SirenLayer(D, D//2, use_bias=True, w0=1., is_first=False)
        self.fc_img = nn.Linear(D//2, 1)

        # self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :return: rgb_density (H, W, N_sample, 1)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        # x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.img_layers(feat)  # (H, W, N_sample, D/2)
        img = self.fc_img(x)  # (H, W, N_sample, 1)

        return img


class OfficialNerf_siren1(nn.Module):
    # OfficialNerf_siren1 是一个基于 SIREN 架构的神经网络，用于从编码后的位置信息中生成预测的图像强度值。
    # 输入是编码后的位置信息，形状为 (H, W, N_sample, pos_in_dims)，输出是预测的图像强度值，形状为 (H, W, N_sample, 1)
    def __init__(self, pos_in_dims, D):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(OfficialNerf_siren1, self).__init__()

        self.pos_in_dims = pos_in_dims

        self.layers0 = nn.Sequential(
            SirenLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        self.layers1 = nn.Sequential(
            SirenLayer(D + pos_in_dims, D, use_bias=True, w0=1., is_first=False),
            SirenLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        # self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.img_layers = SirenLayer(D, D // 2, use_bias=True, w0=1., is_first=False)
        self.fc_img = nn.Linear(D // 2, 1)

        # self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, pos_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :return: rgb_density (H, W, N_sample, 1)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        # x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.img_layers(feat)  # (H, W, N_sample, D/2)
        img = self.fc_img(x)  # (H, W, N_sample, 1)

        return img
