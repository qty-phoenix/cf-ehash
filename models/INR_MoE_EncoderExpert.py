# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This code is the implementation of the Neural Experts model
# It is heavily based on our previous work DiGSimplementation with several significant modifications.
# It was partly based on SIREN and SAL implementation and architecture but with several significant modifications.
# for the original DiGS version see: https://github.com/Chumbyte/DiGS
# for the original SIREN version see: https://github.com/vsitzmann/siren
# for the original SAL version see: https://github.com/matanatz/SAL

import numpy as np
import torch
import torch.nn as nn
import models.managers as managers

from scipy.spatial.distance import cdist
from models.modules import (FullyConnectedNN, ParallelFullyConnectedNN, InputEncoder, tSoftMax, DummyModule)
from sklearn.cluster import KMeans


class EncoderExpertSelector(nn.Module):
    """
    编码器端专家选择器
    基于输入坐标的特征选择合适的编码器专家
    """
    def __init__(self, cfg, in_dim):
        super().__init__()
        self.n_encoder_experts = 2  # 2个编码器专家：精细和粗糙
        self.temperature = cfg.get('encoder_selector_temperature', 1.0)

        # 专家选择网络
        self.selector_net = FullyConnectedNN(
            in_dim, self.n_encoder_experts,
            num_hidden_layers=cfg.get('encoder_selector_n_hidden_layers', 2),
            hidden_features=cfg.get('encoder_selector_hidden_dim', 64),
            outermost_linear=True,
            nonlinearity=cfg.get('encoder_selector_nl', 'relu'),
            init_type=cfg.get('encoder_selector_init', 'default'),
            input_encoding=cfg.get('encoder_selector_input_encoding', 'none'),
            module_name='encoder_expert_selector'
        )

        # Softmax激活
        self.q_activation = tSoftMax(self.temperature, -1, False)

    def forward(self, coords):
        """
        Args:
            coords: (B, N, in_dim) 输入坐标
        Returns:
            q: (B, n_encoder_experts, N) 专家权重
        """
        # coords: (B, N, in_dim)
        B, N, in_dim = coords.shape

        # 展平进行专家选择
        coords_flat = coords.view(-1, in_dim)  # (B*N, in_dim)
        raw_q = self.selector_net(coords_flat)  # (B*N, n_encoder_experts)

        # 激活
        q_flat = self.q_activation(raw_q)  # (B*N, n_encoder_experts)

        # 重塑回原始形状
        q = q_flat.view(B, N, self.n_encoder_experts).transpose(1, 2)  # (B, n_encoder_experts, N)

        return q


class EncoderExpert(nn.Module):
    """
    编码器专家：哈希编码 + 选择性激活
    """
    def __init__(self, cfg, expert_type='fine'):
        """
        Args:
            cfg: 配置
            expert_type: 'fine' 或 'coarse'
        """
        super().__init__()
        self.expert_type = expert_type
        self.input_dim = cfg['in_dim']

        # 哈希编码参数
        hash_cfg = {
            'n_levels': cfg.get('hash_n_levels', 16),
            'n_features_per_level': cfg.get('hash_n_features_per_level', 2),
            'log2_hashmap_size': cfg.get('hash_log2_hashmap_size', 19),
            'base_resolution': cfg.get('hash_base_resolution', 16),
            'finest_resolution': cfg.get('hash_finest_resolution', 512),
            'input_dim': cfg['in_dim']
        }

        # 根据专家类型调整参数
        if expert_type == 'coarse':
            # 粗糙专家：减少层数和特征
            hash_cfg['n_levels'] = max(1, hash_cfg['n_levels'] // 2)
            hash_cfg['n_features_per_level'] = max(1, hash_cfg['n_features_per_level'] // 2)
            # 降低分辨率范围
            hash_cfg['finest_resolution'] = hash_cfg['finest_resolution'] // 4

        from models.modules import HashEncoding
        self.hash_encoder = HashEncoding(**hash_cfg)

        # 输出维度
        self.output_dim = hash_cfg['n_levels'] * hash_cfg['n_features_per_level']

        # 可选：拼接原始坐标
        if cfg.get('hash_cat_coords', True):
            self.output_dim += cfg['in_dim']

        # 粗糙专家的激活层（可选跳过某些层）
        if expert_type == 'coarse':
            self.skip_activation_layers = cfg.get('coarse_skip_activation_layers', 0)
            if self.skip_activation_layers > 0:
                # 创建一个简单的线性变换来模拟跳过激活
                self.activation_skip = nn.Linear(self.output_dim, self.output_dim)
            else:
                self.activation_skip = None
        else:
            self.activation_skip = None

    def forward(self, coords):
        """
        Args:
            coords: (B, N, input_dim)
        Returns:
            encoded: (B, N, output_dim)
        """
        B, N, _ = coords.shape

        # 【修复】直接传入 (B, N, input_dim) 格式，HashEncoding 就是为此设计的
        # 不要展平！否则会导致维度语义错乱，插值计算完全错误
        hash_features = self.hash_encoder(coords)  # (B, N, hash_dim)

        # 可选拼接原始坐标
        if hasattr(self, 'hash_cat_coords') and self.hash_cat_coords:
            encoded = torch.cat([hash_features, coords], dim=-1)
        else:
            encoded = hash_features

        # 粗糙专家的特殊处理
        if self.expert_type == 'coarse' and self.activation_skip is not None:
            # 应用跳过激活的变换
            encoded_flat = encoded.view(B * N, -1)
            encoded_flat = self.activation_skip(encoded_flat)
            encoded = encoded_flat.view(B, N, -1)

        return encoded


class INR_MoE_EncoderExpert(nn.Module):
    """
    编码器端多专家INR模型
    2个编码器专家（精细和粗糙）+ 单个解码器
    """
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']

        self.init_type = cfg['decoder_init_type']
        self.n_encoder_experts = 2  # 2个编码器专家

        # 解码器配置
        self.decoder_input_encoding_module = InputEncoder(cfg, 'none', cfg['decoder_hidden_dim'],
                                                          module_name='decoder_input_encoding_module')

        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim

        # 单个解码器（不是并行的）
        self.decoder = FullyConnectedNN(decoder_first_layer_dim, cfg['out_dim'],
                                       num_hidden_layers=cfg['decoder_n_hidden_layers'],
                                       hidden_features=cfg['decoder_hidden_dim'],
                                       outermost_linear=True,
                                       nonlinearity=cfg['decoder_nl'],
                                       init_type=self.init_type,
                                       input_encoding='none',
                                       freq=cfg['decoder_freqs'],
                                       module_name='decoder')

        # 编码器专家
        self.fine_encoder = EncoderExpert(cfg, expert_type='fine')
        self.coarse_encoder = EncoderExpert(cfg, expert_type='coarse')

        # 编码器专家选择器
        self.encoder_selector = EncoderExpertSelector(cfg, cfg['in_dim'])

        # 专家融合方式
        self.fusion_type = cfg.get('encoder_fusion_type', 'weighted_sum')  # 'weighted_sum' 或 'attention'

        if self.fusion_type == 'attention':
            # 注意力融合
            self.fusion_attention = nn.Linear(self.fine_encoder.output_dim, 1)

        print(f"[INR_MoE_EncoderExpert] 初始化完成:")
        print(f"  - 编码器专家数量: {self.n_encoder_experts}")
        print(f"  - 精细编码器输出维度: {self.fine_encoder.output_dim}")
        print(f"  - 粗糙编码器输出维度: {self.coarse_encoder.output_dim}")
        print(f"  - 融合方式: {self.fusion_type}")

    def forward(self, coords, **kwargs):
        """
        Args:
            coords: (B, N, in_dim) 输入坐标
        Returns:
            prediction: (B, N, out_dim) 最终预测
            encoder_expert_usage: (n_encoder_experts, N) 专家使用统计
        """
        B, N, _ = coords.shape

        # 1. 编码器专家选择
        encoder_q = self.encoder_selector(coords)  # (B, n_encoder_experts, N)

        # 2. 两个编码器专家分别编码
        fine_encoded = self.fine_encoder(coords)      # (B, N, fine_dim)
        coarse_encoded = self.coarse_encoder(coords)  # (B, N, coarse_dim)

        # 确保维度匹配（如果不同，需要对齐）
        fine_dim = fine_encoded.shape[-1]
        coarse_dim = coarse_encoded.shape[-1]

        if fine_dim != coarse_dim:
            # 如果维度不同，使用线性投影对齐
            if not hasattr(self, 'dim_align_coarse'):
                self.dim_align_coarse = nn.Linear(coarse_dim, fine_dim)
            coarse_encoded = self.dim_align_coarse(coarse_encoded.view(B * N, -1)).view(B, N, -1)

        # 3. 融合编码器输出
        if self.fusion_type == 'weighted_sum':
            # 加权求和融合
            # encoder_q: (B, n_encoder_experts, N) -> (B, N, n_encoder_experts)
            q_transposed = encoder_q.transpose(1, 2)

            # 堆叠专家输出: (B, N, n_encoder_experts, dim)
            expert_outputs = torch.stack([fine_encoded, coarse_encoded], dim=2)

            # 加权融合: sum(q * expert_output, dim=2)
            fused_encoded = torch.sum(q_transposed.unsqueeze(-1) * expert_outputs, dim=2)

        elif self.fusion_type == 'attention':
            # 注意力融合
            expert_outputs = torch.stack([fine_encoded, coarse_encoded], dim=1)  # (B, n_encoder_experts, N, dim)

            # 计算注意力权重
            attention_logits = self.fusion_attention(expert_outputs.view(B * self.n_encoder_experts * N, -1))
            attention_weights = torch.softmax(attention_logits.view(B, self.n_encoder_experts, N), dim=1)

            # 加权融合
            fused_encoded = torch.sum(attention_weights.unsqueeze(-1) * expert_outputs, dim=1)

        # 4. 解码器处理
        decoder_input = self.decoder_input_encoding_module(fused_encoded.view(B * N, -1), **kwargs)
        decoder_output = self.decoder(decoder_input)  # (B*N, out_dim)

        # 重塑输出
        prediction = decoder_output.view(B, N, -1)

        # 5. 计算专家使用统计（用于损失函数和可视化）
        encoder_expert_usage = encoder_q.mean(dim=0)  # (n_encoder_experts, N)

        return {
            'prediction': prediction,
            'encoder_expert_usage': encoder_expert_usage,
            'encoder_q': encoder_q,
            'fine_encoded': fine_encoded,
            'coarse_encoded': coarse_encoded,
            'fused_encoded': fused_encoded
        }







