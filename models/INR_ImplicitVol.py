"""
基于ImplicitVol架构的INR模型
使用位置编码(PE)和带残差连接的MLP网络预测像素值
参考: ImplicitVol - 3D超声重建的隐式神经表示
"""

import torch.nn as nn
import torch.nn.functional as F
from models.modules import InputEncoder


class ResidualBlock(nn.Module):
    """残差块 - ImplicitVol风格"""
    def __init__(self, hidden_dim, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sine':
            from models.modules import Sine
            self.activation = Sine(30.0, False)
        else:
            self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = out + residual  # 残差连接
        out = self.activation(out)
        return out


class INR_ImplicitVol(nn.Module):
    """
    基于ImplicitVol架构的INR模型
    
    特点:
    - 使用位置编码(PE)对3D坐标进行编码
    - 使用带残差连接的MLP网络（ResNet风格）
    - 支持RGB和灰度图像
    - 更深的网络结构（适合体积重建任务）
    """
    
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        self.init_type = cfg.get('decoder_init_type', 'normal')
        
        # 位置编码模块 - 使用PE编码
        # ImplicitVol使用较多的频率来捕获细节
        pe_num_freqs = cfg.get('pe_num_freqs', 10)
        cfg['pe_num_freqs'] = pe_num_freqs
        cfg['pe_linear_freqs'] = False  # 使用指数频率
        
        self.decoder_input_encoding_module = InputEncoder(
            cfg, 
            'PE',  # 使用位置编码
            cfg.get('decoder_hidden_dim', 256)
        )
        
        first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        
        # MLP解码器 - ImplicitVol风格（带残差连接）
        decoder_n_hidden_layers = cfg.get('decoder_n_hidden_layers', 8)
        decoder_hidden_dim = cfg.get('decoder_hidden_dim', 256)
        decoder_nl = cfg.get('decoder_nl', 'relu')  # ImplicitVol使用ReLU
        
        # 输入层
        self.input_layer = nn.Linear(first_layer_dim, decoder_hidden_dim)
        
        # 残差块（中间层）
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(decoder_hidden_dim, decoder_nl)
            for _ in range(decoder_n_hidden_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(decoder_hidden_dim, cfg['out_dim'])
        
        # 初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        if self.init_type == 'normal':
            # Kaiming Normal初始化（适合ReLU）
            nn.init.kaiming_normal_(self.input_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')
            for block in self.residual_blocks:
                nn.init.kaiming_normal_(block.fc1.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(block.fc2.weight, mode='fan_in', nonlinearity='relu')
        elif self.init_type == 'siren':
            # SIREN初始化（如果使用sine激活）
            # 这里简化处理，实际可以使用更复杂的初始化
            nn.init.kaiming_normal_(self.input_layer.weight)
            nn.init.kaiming_normal_(self.output_layer.weight)
        
        # 偏置初始化为0
        nn.init.zeros_(self.input_layer.bias)
        nn.init.zeros_(self.output_layer.bias)
        for block in self.residual_blocks:
            nn.init.zeros_(block.fc1.bias)
            nn.init.zeros_(block.fc2.bias)
        
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
            manifold_pnts_pred = self._forward_mlp(mnfld_pnts_encoded)
        else:
            manifold_pnts_pred = None
        
        # 预测非流形点的像素值
        nonmanifold_pnts_pred = self._forward_mlp(non_mnfld_pnts_encoded)  # (B, N, out_dim)
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred.permute(0, 2, 1),  # (B, out_dim, N)
        }
    
    def _forward_mlp(self, x):
        """MLP前向传播（带残差连接）"""
        # 输入层
        x = self.input_layer(x)
        x = F.relu(x)
        
        # 残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出层（线性，无激活）
        x = self.output_layer(x)
        
        return x

