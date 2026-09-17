import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import Sine, SirenLayer


class MultiplicativeActivation(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta=1.0):
        super(MultiplicativeActivation, self).__init__()
        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim,)))
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, input):
        norm = (input ** 2).sum(dim=-1, keepdim=True) + (self.mu ** 2).sum(dim=-1,
                                                                           keepdim=True).T - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))


class WaveLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()
        self.activation = MultiplicativeActivation(input_dim, hidden_dim, alpha=6.0 / (hidden_dim + 1))

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, input):
        return self.activation(input)


class GaborFeatureProcessor(nn.Module):
    """专门处理Gabor特征的子网络"""

    def __init__(self, in_dim=4, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim // 4),
            nn.LayerNorm(out_dim // 4),
            Sine(w0=1.),
            nn.Linear(out_dim // 4, out_dim),
            Sine(w0=1.)
        )

    def forward(self, x):
        return self.net(x)


class OfficialNerf_wave(nn.Module):
    def __init__(self, pos_in_dims=63, gabor_dims=4, D=256):
        super().__init__()
        self.pos_in_dims = pos_in_dims
        self.gabor_dims = gabor_dims
        self.total_dims = pos_in_dims + gabor_dims

        # 主分支处理位置编码
        self.layers0 = nn.Sequential(
            WaveLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
            WaveLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        # Gabor特征处理分支（可选）
        self.gabor_branch = GaborFeatureProcessor(gabor_dims, D) if gabor_dims > 0 else None

        # 共享的后续层
        self.layers1 = nn.Sequential(
            WaveLayer(D + pos_in_dims, D, use_bias=True, w0=1., is_first=False),
            WaveLayer(D, D, use_bias=True, w0=1., is_first=False),
        )

        self.fc_feature = nn.Linear(D, D)
        self.img_layers = WaveLayer(D, D // 2, use_bias=True, w0=1., is_first=False)
        self.fc_img = nn.Linear(D // 2, 1)
        self.fc_img.bias.data = torch.tensor([0.02]).float()

    def forward(self, x):
        # 拆分输入
        pos_enc = x[..., :self.pos_in_dims]
        gabor_feats = x[..., self.pos_in_dims:] if x.size(-1) > self.pos_in_dims else None

        # 主分支处理
        x = self.layers0(pos_enc)

        # 条件Gabor分支
        if self.gabor_branch is not None and gabor_feats is not None:
            gabor_out = self.gabor_branch(gabor_feats)
            x = x + gabor_out  # 残差连接

        # 后续处理
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.layers1(x)
        feat = self.fc_feature(x)
        x = self.img_layers(feat)
        img = self.fc_img(x)
        return img
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np
# from scipy.ndimage import gaussian_filter
# from scipy.signal import fftconvolve
# from skimage.feature import match_template
# from model import Sine
# from model import SirenLayer
#
#
# class MultiplicativeActivation(nn.Module):
#     def __init__(self, in_dim, out_dim, alpha, beta=1.0):
#         super(MultiplicativeActivation, self).__init__()
#         self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
#         self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
#         self.linear = torch.nn.Linear(in_dim, out_dim)
#
#     def forward(self, input):
#         norm = (input ** 2).sum(dim=-1, keepdim=True) + (self.mu ** 2).sum(dim=-1, keepdim=True).T - 2 * input @ self.mu.T
#         return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))
#
# class WaveLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
#         super().__init__()
#         self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
#         self.is_first = is_first
#         self.input_dim = input_dim
#         self.w0 = w0
#         self.c = 6
#         self.reset_parameters()
#
#         # 使用乘性激活函数
#         self.activation = MultiplicativeActivation(input_dim, hidden_dim, alpha=6.0 / (hidden_dim + 1))
#
#     def reset_parameters(self):
#         with torch.no_grad():
#             dim = self.input_dim
#             w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
#             self.layer.weight.uniform_(-w_std, w_std)
#             if self.layer.bias is not None:
#                 self.layer.bias.uniform_(-w_std, w_std)
#
#     def forward(self, input):
#         x = self.layer(input)
#         out = self.activation(input)
#         return out
#
#
# class OfficialNerf_wave(nn.Module):
#     def __init__(self, pos_in_dims, D):
#         super(OfficialNerf_wave, self).__init__()
#
#         self.pos_in_dims = pos_in_dims
#
#         self.layers0 = nn.Sequential(
#             WaveLayer(pos_in_dims, D, use_bias=True, w0=30., is_first=True),
#             WaveLayer(D, D, use_bias=True, w0=1., is_first=False),
#             # WaveLayer(D, D, use_bias=True, w0=30., is_first=False),
#             # WaveLayer(D, D, use_bias=True, w0=30., is_first=False),
#         )
#
#         self.layers1 = nn.Sequential(
#             WaveLayer(D + pos_in_dims, D, use_bias=True, w0=1., is_first=False),
#             WaveLayer(D, D, use_bias=True, w0=1., is_first=False),
#             # WaveLayer(D, D, use_bias=True, w0=30., is_first=False),
#             # WaveLayer(D, D, use_bias=True, w0=30., is_first=False),
#         )
#
#         self.fc_feature = nn.Linear(D, D)
#         self.img_layers = WaveLayer(D, D // 2, use_bias=True, w0=1., is_first=False)
#         self.fc_img = nn.Linear(D // 2, 1)
#
#         self.fc_img.bias.data = torch.tensor([0.02]).float()
#
#     def forward(self, pos_enc):
#         x = self.layers0(pos_enc)  # (H, W, N_sample, D)
#         if torch.isnan(x).any(): print("⚠️ NaN after layers0")
#         x = torch.cat([x, pos_enc], dim=-1)  # (H, W, N_sample, D+pos_in_dims)
#         x = self.layers1(x)  # (H, W, N_sample, D)
#         feat = self.fc_feature(x)  # (H, W, N_sample, D)
#         x = self.img_layers(feat)  # (H, W, N_sample, D/2)
#         img = self.fc_img(x)  # (H, W, N_sample, 1)
#         return img