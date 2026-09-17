# Text-Guided Neural Experts Model
# 基于文本描述引导的多专家混合模型

import numpy as np
import torch
import torch.nn as nn
import models.managers as managers

from models.modules import (FullyConnectedNN, ParallelFullyConnectedNN, InputEncoder, 
                            tSoftMax, DummyModule)
from models.text_encoder import build_text_encoder


class TextGuidedManager(nn.Module):
    """
    文本引导的Manager网络
    在原始Manager基础上融入文本描述信息
    """
    def __init__(self, cfg, in_dim, text_dim=256, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = cfg['in_dim']
        self.text_dim = text_dim
        self.temperature = float(cfg['manager_softmax_temperature'])  # 确保转换为浮点数
        self.use_text = cfg.get('use_text_guidance', True)
        
        # 文本融合方式
        self.text_fusion = cfg.get('text_fusion_type', 'concat')  # 'concat', 'add', 'gate'
        
        # 根据融合方式确定Manager网络的输入维度
        if self.text_fusion == 'concat':
            manager_input_dim = in_dim + text_dim
        elif self.text_fusion == 'add':
            # 加法融合需要投影文本特征
            manager_input_dim = in_dim
            self.text_projection = nn.Linear(text_dim, in_dim)
        elif self.text_fusion == 'gate':
            # 门控融合
            manager_input_dim = in_dim
            self.text_projection = nn.Linear(text_dim, in_dim)
            self.gate = nn.Sequential(
                nn.Linear(text_dim + in_dim, in_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported text_fusion_type: {self.text_fusion}")
        
        # 激活函数
        self.q_activation_dict = {
            'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
            'sigmoid': nn.Sigmoid(), 
            'none': DummyModule()
        }
        
        # Manager网络
        self.manager_type = cfg['manager_type']
        if self.manager_type == 'standard':
            self.manager_net = FullyConnectedNN(
                manager_input_dim, 
                self.n_experts,
                num_hidden_layers=cfg['manager_n_hidden_layers'],
                hidden_features=cfg['manager_hidden_dim'], 
                outermost_linear=True,
                nonlinearity=cfg['manager_nl'], 
                init_type=cfg['manager_init'],
                input_encoding=cfg['manager_input_encoding'],
                module_name=module_name + '.manager_net'
            )
        elif self.manager_type == 'none':
            self.manager_net = managers.DummyManager(self.n_experts)
        else:
            raise ValueError(f"Unsupported manager_type: {self.manager_type}")
        
        self.q_activation = self.q_activation_dict.get(cfg['manager_q_activation'], DummyModule())
        self.clamp_q = float(cfg['manager_clamp_q'])  # 确保转换为浮点数

    def forward(self, points, text_features=None):
        """
        Args:
            points: (B*n_points, point_dim) 点特征
            text_features: (B, text_dim) 文本特征
        Returns:
            q: (B*n_points, n_experts) 专家权重
            selected_expert_idx: (B*n_points,) 选中的专家索引
            raw_q: (1, n_experts, B*n_points) 原始logits
        """
        # 融合文本特征
        if self.use_text and text_features is not None:
            B = text_features.shape[0]
            n_points_per_batch = points.shape[0] // B
            
            if self.text_fusion == 'concat':
                # 拼接融合：将文本特征扩展到每个点
                text_expanded = text_features.repeat_interleave(n_points_per_batch, dim=0)  # (B*n_points, text_dim)
                manager_input = torch.cat([points, text_expanded], dim=-1)
            
            elif self.text_fusion == 'add':
                # 加法融合
                text_proj = self.text_projection(text_features)  # (B, in_dim)
                text_expanded = text_proj.repeat_interleave(n_points_per_batch, dim=0)
                manager_input = points + text_expanded
            
            elif self.text_fusion == 'gate':
                # 门控融合
                text_proj = self.text_projection(text_features)
                text_expanded = text_proj.repeat_interleave(n_points_per_batch, dim=0)
                text_features_expanded = text_features.repeat_interleave(n_points_per_batch, dim=0)
                gate_input = torch.cat([points, text_features_expanded], dim=-1)
                gate_weights = self.gate(gate_input)  # (B*n_points, in_dim)
                manager_input = gate_weights * points + (1 - gate_weights) * text_expanded
        else:
            manager_input = points
        
        # Manager网络前向传播
        raw_q = self.manager_net(manager_input)  # (B*n_points, n_experts)
        
        q = self.q_activation(raw_q)
        q = torch.clamp(q, self.clamp_q)
        
        selected_expert_idx = torch.argmax(q, 1)
        return q, selected_expert_idx, raw_q.T.unsqueeze(0)


class INR_MoE_TextGuided(nn.Module):
    """
    文本引导的隐式神经表示混合专家模型
    """
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        
        self.init_type = cfg['decoder_init_type']
        self.n_experts = cfg['n_experts']
        self.manager_init = cfg['manager_init']
        self.share_encoder = cfg['shared_encoder']
        self.use_text_guidance = cfg.get('use_text_guidance', True)
        
        # 文本编码器
        if self.use_text_guidance:
            self.text_encoder = build_text_encoder(cfg)
            text_output_dim = cfg.get('text_output_dim', 256)
        else:
            self.text_encoder = None
            text_output_dim = 0
        
        # Decoder输入编码
        self.decoder_input_encoding_module = InputEncoder(
            cfg, cfg['decoder_input_encoding'], 
            cfg['decoder_hidden_dim'],
            module_name='decoder_input_encoding_module'
        )
        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        
        # 并行专家解码器
        self.decoder = ParallelFullyConnectedNN(
            self.n_experts, 
            decoder_first_layer_dim, 
            cfg['out_dim'],
            num_hidden_layers=cfg['decoder_n_hidden_layers'],
            hidden_features=cfg['decoder_hidden_dim'], 
            outermost_linear=True,
            nonlinearity=cfg['decoder_nl'], 
            init_type=self.init_type,
            input_encoding=cfg['decoder_input_encoding'],
            freq=cfg['decoder_freqs'],
            module_name='decoder', 
            cfg=cfg
        )
        
        # Manager条件调节器
        self.manager_conditioning = cfg['manager_conditioning']
        if self.use_text_guidance:
            # 使用文本引导的条件调节器
            self.manager_conditioner = managers.TextGuidedManagerConditioner(
                cfg['manager_conditioning'],
                decoder_first_layer_dim,
                text_output_dim,
                fusion_type=cfg.get('text_conditioner_fusion', 'concat'),
                expert_decoder=self.decoder
            )
        else:
            # 使用原始条件调节器
            self.manager_conditioner = managers.ManagerConditioner(
                cfg['manager_conditioning'],
                decoder_first_layer_dim, 
                self.decoder
            )
        
        # Manager输入编码
        if not self.share_encoder:
            self.manager_input_encoding_module = InputEncoder(
                cfg, cfg['manager_input_encoding'],
                cfg['manager_hidden_dim'],
                module_name='manager_input_encoding_module'
            )
        else:
            self.manager_input_encoding_module = self.decoder_input_encoding_module
        
        manager_first_layer_dim = self.manager_input_encoding_module.first_layer_dim
        
        # 根据条件调节方式调整维度
        if not self.manager_conditioning == 'none':
            manager_first_layer_dim = manager_first_layer_dim + decoder_first_layer_dim
        
        # 如果使用文本引导，进一步调整维度
        if self.use_text_guidance:
            text_conditioner_fusion = cfg.get('text_conditioner_fusion', 'concat')
            if text_conditioner_fusion == 'concat':
                manager_first_layer_dim = manager_first_layer_dim + text_output_dim
        
        # 创建文本引导的Manager网络
        if self.use_text_guidance:
            self.manager_net = TextGuidedManager(
                cfg, 
                manager_first_layer_dim, 
                text_output_dim
            )
        else:
            from models.INR_MoE import Manager
            self.manager_net = Manager(cfg, manager_first_layer_dim)
    
    def forward(self, non_mnfld_pnts, mnfld_pnts=None, text_input=None, **kwargs):
        """
        Args:
            non_mnfld_pnts: (1, n_nm, d) 非流形点（所有采样点）
            mnfld_pnts: (1, n_m, d) 流形点（可选）
            text_input: 文本输入（字符串列表或tokenized ids）
        """
        # 编码文本
        text_features = None
        if self.use_text_guidance and text_input is not None:
            text_features = self.text_encoder(text_input)  # (B, text_dim)
        
        # 编码非流形点
        encoded_non_mnfld_pnts_experts = self.decoder_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        # 流形点处理
        if mnfld_pnts is not None:
            encoded_mnfld_pnts_experts = self.decoder_input_encoding_module(mnfld_pnts, **kwargs)
            manifold_pnts_pred = self.decoder(encoded_mnfld_pnts_experts)  # (1, k, n_m, 1)
            manifold_pnts_pred = manifold_pnts_pred.squeeze(-1)  # (1, k, n_m)
            
            if not self.share_encoder:
                manager_input = self.manager_input_encoding_module(mnfld_pnts, **kwargs)
            else:
                manager_input = encoded_mnfld_pnts_experts
            
            # 应用条件调节（包含文本特征）
            manager_input = self.manager_conditioner(
                encoded_mnfld_pnts_experts, 
                manager_input, 
                text_features=text_features,
                **kwargs
            )
            
            # Manager前向传播
            if self.use_text_guidance:
                mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q = self.manager_net(
                    manager_input.view(-1, manager_input.shape[-1]),
                    text_features=text_features
                )
            else:
                mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q = self.manager_net(
                    manager_input.view(-1, manager_input.shape[-1])
                )
            mnfld_q = mnfld_q.T.unsqueeze(0)  # (1, k, n_m)
            
            selected_manifold_pnts_pred = torch.gather(
                manifold_pnts_pred, dim=-2,
                index=mnfld_selected_expert_idx[None, None, :]
            ).squeeze(-2)
        else:
            manifold_pnts_pred = None
            mnfld_q, mnfld_selected_expert_idx, selected_manifold_pnts_pred = None, None, None
            mnfld_raw_q = None
        
        # 非流形点处理
        nonmanifold_pnts_pred = self.decoder(encoded_non_mnfld_pnts_experts)  # (1, k, n_nm, d)
        
        if not self.share_encoder:
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts, **kwargs)
        else:
            encoded_non_mnfld_pnts_manager = encoded_non_mnfld_pnts_experts
        
        nonmnld_manager_input = encoded_non_mnfld_pnts_manager
        
        # 应用条件调节（包含文本特征）
        nonmnld_manager_input = self.manager_conditioner(
            encoded_non_mnfld_pnts_experts,
            nonmnld_manager_input,
            text_features=text_features,
            **kwargs
        )
        
        nonmnld_manager_input = nonmnld_manager_input.reshape(-1, nonmnld_manager_input.shape[-1])
        if self.use_text_guidance:
            mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q = self.manager_net(
                nonmnld_manager_input,
                text_features=text_features
            )
        else:
            mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q = self.manager_net(
                nonmnld_manager_input
            )
        nonmnfld_q = mout_nonmnfld_q.T.unsqueeze(0)  # (1, k, n_nm)
        
        selected_nonmanifold_pnts_pred = torch.gather(
            nonmanifold_pnts_pred, dim=-3,
            index=nonmnfld_selected_expert_idx[None, None, :, None].repeat(
                [1, 1, 1, nonmanifold_pnts_pred.shape[-1]]
            )
        ).squeeze(-2)
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
            "mnfld_q": mnfld_q,
            "nonmnfld_q": nonmnfld_q,
            "selected_manifold_pnts_pred": selected_manifold_pnts_pred,
            "selected_nonmanifold_pnts_pred": selected_nonmanifold_pnts_pred,
            "mnfld_selected_expert_idx": mnfld_selected_expert_idx,
            "nonmnfld_selected_expert_idx": nonmnfld_selected_expert_idx,
            "mnfld_raw_q": mnfld_raw_q,
            "nonmnfld_raw_q": nonmnfld_raw_q,
            "nonmnld_manager_input": nonmnld_manager_input,
            "mout_nonmnfld_q": mout_nonmnfld_q,
            "text_features": text_features,  # 返回文本特征用于可视化
        }

