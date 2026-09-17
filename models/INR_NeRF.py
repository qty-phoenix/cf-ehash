"""
基于原始NeRF架构的INR模型
使用位置编码(PE)和MLP网络预测RGB/灰度值
"""

import torch.nn as nn
from models.modules import FullyConnectedNN, InputEncoder


class INR_NeRF(nn.Module):
    """
    基于原始NeRF架构的INR模型
    
    特点:
    - 使用位置编码(PE)对3D坐标进行编码
    - 使用MLP网络预测像素值
    - 支持RGB和灰度图像
    """
    
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        self.init_type = cfg.get('decoder_init_type', 'normal')
        
        # 位置编码模块 - 使用PE编码
        # NeRF默认使用10个频率 (2^0 到 2^9)
        pe_num_freqs = cfg.get('pe_num_freqs', 10)
        cfg['pe_num_freqs'] = pe_num_freqs
        cfg['pe_linear_freqs'] = False  # 使用指数频率 (NeRF风格)
        
        self.decoder_input_encoding_module = InputEncoder(
            cfg, 
            'PE',  # 使用位置编码
            cfg.get('decoder_hidden_dim', 256)
        )
        
        first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        
        # MLP解码器 - NeRF风格
        # 默认配置: 8层, 256维, ReLU激活
        decoder_n_hidden_layers = cfg.get('decoder_n_hidden_layers', 8)
        decoder_hidden_dim = cfg.get('decoder_hidden_dim', 256)
        decoder_nl = cfg.get('decoder_nl', 'relu')  # NeRF使用ReLU
        
        self.decoder = FullyConnectedNN(
            first_layer_dim, 
            cfg['out_dim'],
            num_hidden_layers=decoder_n_hidden_layers,
            hidden_features=decoder_hidden_dim, 
            outermost_linear=True,
            nonlinearity=decoder_nl, 
            init_type=self.init_type,
            input_encoding='PE',
            sphere_init_params=cfg.get('sphere_init_params', [1.6, 0.1]),  
            init_r=cfg.get('decoder_init_r', 0.1)
        )
        
    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        """
        前向传播
        
        Args:
            non_mnfld_pnts: (B, N, 3) 非流形点(像素坐标对应的3D坐标)
            mnfld_pnts: (B, M, 3) 流形点(可选,本任务不使用)
            **kwargs: 其他参数(如dino, img等,本任务不使用)
        
        Returns:
            dict: 包含预测结果
                - nonmanifold_pnts_pred: (B, out_dim, N) 预测的像素值
        """
        # 对3D坐标进行位置编码
        non_mnfld_pnts_encoded = self.decoder_input_encoding_module(non_mnfld_pnts)
        
        # 流形点处理(本任务不使用)
        if mnfld_pnts is not None:
            mnfld_pnts_encoded = self.decoder_input_encoding_module(mnfld_pnts)
            manifold_pnts_pred = self.decoder(mnfld_pnts_encoded)
        else:
            manifold_pnts_pred = None
        
        # 预测非流形点的像素值
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts_encoded)  # (B, N, out_dim)
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred.permute(0, 2, 1),  # (B, out_dim, N)
        }

