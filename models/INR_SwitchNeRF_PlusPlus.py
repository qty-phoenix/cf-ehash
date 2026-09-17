#!/usr/bin/env python3
"""
Switch NeRF++ 模型实现
基于论文: "Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields"
核心组件:
1. 异构哈希专家混合 (HMoHE): 每个专家使用不同分辨率范围的哈希网格
2. 基于哈希的门控网络: 门控网络也使用哈希编码而不是PE
3. 稀疏门控MoE框架: 使用top-k选择专家（训练时仍用加权平均保证梯度）
"""

import torch
import torch.nn as nn
import models.managers as managers
from models.modules import InputEncoder, tSoftMax, DummyModule, FullyConnectedNN, ParallelFullyConnectedNN
from models.modules import HashEncoding
try:
    from models.modules import HeterogeneousMultiExpertHashEncoding
except ImportError:
    # 如果导入失败，说明可能还没有添加到modules.py中
    raise ImportError("HeterogeneousMultiExpertHashEncoding未找到，请确保已添加到models/modules.py中")
import math


class HashBasedManager(nn.Module):
    """
    Switch NeRF++的基于哈希的门控网络
    使用哈希编码而不是PE来编码输入坐标
    """
    def __init__(self, cfg, in_dim, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = in_dim
        self.temperature = float(cfg['manager_softmax_temperature'])
        
        # 哈希编码参数
        hash_n_levels = cfg.get('manager_hash_n_levels', 8)
        hash_n_features_per_level = cfg.get('manager_hash_n_features_per_level', 2)
        hash_log2_hashmap_size = cfg.get('manager_hash_log2_hashmap_size', 14)
        hash_base_resolution = cfg.get('manager_hash_base_resolution', 16)
        hash_finest_resolution = cfg.get('manager_hash_finest_resolution', 1024)
        
        # 基于哈希的输入编码
        self.hash_encoder = HashEncoding(
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features_per_level,
            log2_hashmap_size=hash_log2_hashmap_size,
            base_resolution=hash_base_resolution,
            finest_resolution=hash_finest_resolution,
            input_dim=in_dim
        )
        
        # Manager网络输入维度（哈希编码的输出维度）
        manager_input_dim = hash_n_levels * hash_n_features_per_level
        
        # 激活函数字典
        self.q_activation_dict = {
            'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
            'sigmoid': nn.Sigmoid(), 
            'none': DummyModule()
        }
        
        # Manager网络（MLP）
        self.manager_type_dict = {
            'none': lambda: managers.DummyManager(self.n_experts),
            'standard': lambda: FullyConnectedNN(
                manager_input_dim, self.n_experts,
                num_hidden_layers=cfg['manager_n_hidden_layers'],
                hidden_features=cfg['manager_hidden_dim'], 
                outermost_linear=True,
                nonlinearity=cfg['manager_nl'], 
                init_type=cfg['manager_init'],
                input_encoding=None,  # 输入已经是哈希编码
                module_name=module_name + '.manager_net'
            ),
        }
        
        self.manager_type = cfg['manager_type']
        self.manager_net = self.manager_type_dict.get(self.manager_type, lambda: None)()
        if self.manager_net is None:
            raise ValueError("不支持的manager类型")
        
        self.q_activation = self.q_activation_dict.get(cfg['manager_q_activation'], DummyModule())
        self.clamp_q = float(cfg['manager_clamp_q'])
        
        print(f"[HashBasedManager] 基于哈希的门控网络初始化:")
        print(f"  - Hash encoding: {hash_n_levels} levels, [{hash_base_resolution}, {hash_finest_resolution}]")
        print(f"  - Hash features per level: {hash_n_features_per_level}")
        print(f"  - Manager input dim: {manager_input_dim}")
        print(f"  - Number of experts: {self.n_experts}")
    
    def forward(self, points):
        """
        points: (B, N, input_dim) 输入坐标
        返回: q, selected_expert_idx, raw_q
        """
        # 使用哈希编码编码输入坐标
        hash_features = self.hash_encoder(points)  # (B, N, hash_output_dim)
        
        # Manager网络前向传播
        raw_q = self.manager_net(hash_features)  # (B, N, n_experts)
        
        # 激活和归一化
        q = self.q_activation(raw_q)  # (B, N, n_experts)
        q = torch.clamp(q, self.clamp_q)
        
        # Top-1选择（稀疏门控）
        selected_expert_idx = torch.argmax(q, dim=-1)  # (B, N)
        
        return q, selected_expert_idx, raw_q


class SwitchNeRF_PlusPlus(nn.Module):
    """
    Switch NeRF++ 模型
    主要特点:
    1. 异构哈希专家混合: 每个专家使用不同分辨率范围的哈希网格
    2. 基于哈希的门控网络: 门控网络使用哈希编码而不是PE
    3. 稀疏门控MoE: 使用top-k选择专家（训练时用加权平均保证梯度）
    """
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        
        self.init_type = cfg['decoder_init_type']
        self.n_experts = cfg['n_experts']
        self.share_encoder = cfg.get('shared_encoder', False)
        
        # 异构哈希专家编码参数
        n_levels_per_expert = cfg.get('hash_n_levels_per_expert', 6)
        n_features_per_level = cfg.get('hash_n_features_per_level', 2)
        log2_hashmap_size = cfg.get('hash_log2_hashmap_size', 16)
        
        # 每个专家的分辨率范围（异构设计）
        # 如果未在配置中指定，使用默认值
        base_resolutions = cfg.get('hash_base_resolutions', None)
        finest_resolutions = cfg.get('hash_finest_resolutions', None)
        
        # 异构哈希专家编码模块
        self.heterogeneous_hash_encoder = HeterogeneousMultiExpertHashEncoding(
            n_experts=self.n_experts,
            n_levels_per_expert=n_levels_per_expert,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolutions=base_resolutions,
            finest_resolutions=finest_resolutions,
            input_dim=cfg['in_dim']
        )
        
        # 每个专家的输入维度
        decoder_first_layer_dim = self.heterogeneous_hash_encoder.expert_output_dim
        
        # 解码器网络（每个专家使用独立的FullyConnectedNN）
        # 因为每个专家已经有异构的哈希编码特征，不需要使用ParallelFullyConnectedNN
        self.decoder_experts = nn.ModuleList([
            FullyConnectedNN(
                decoder_first_layer_dim, cfg['out_dim'],
                num_hidden_layers=cfg['decoder_n_hidden_layers'],
                hidden_features=cfg['decoder_hidden_dim'], 
                outermost_linear=True,
                nonlinearity=cfg['decoder_nl'], 
                init_type=self.init_type,
                input_encoding=None,  # 输入已经是哈希编码
                freq=cfg.get('decoder_freqs', 30),
                module_name=f'decoder_expert_{e}'
            )
            for e in range(self.n_experts)
        ])
        
        # 基于哈希的门控网络（不使用PE）
        self.manager_net = HashBasedManager(cfg, cfg['in_dim'], module_name='hash_manager')
        
        print(f"[SwitchNeRF_PlusPlus] 模型初始化完成:")
        print(f"  - 专家数量: {self.n_experts}")
        print(f"  - 每个专家的哈希层级: {n_levels_per_expert}")
        print(f"  - 解码器输入维度: {decoder_first_layer_dim}")
    
    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        """
        Args:
            non_mnfld_pnts: 非流形点 (1, n_nm, d)
            mnfld_pnts: 流形点 (1, n_m, d)，可选
            **kwargs: 其他参数（如dino, img等）
        
        Returns:
            字典包含:
            - nonmanifold_pnts_pred: (1, n_experts, n_nm, out_dim) 所有专家的预测
            - nonmnfld_q: (1, n_experts, n_nm) 专家权重
            - nonmnfld_selected_expert_idx: (n_nm,) 选择的专家索引
            - selected_nonmanifold_pnts_pred: (1, n_nm, out_dim) 选择的专家预测
        """
        # 异构哈希编码非流形点
        # expert_features: (1, n_experts, n_nm, expert_output_dim)
        expert_features = self.heterogeneous_hash_encoder(non_mnfld_pnts)
        
        # 对每个专家分别调用对应的decoder
        B, E, N, D = expert_features.shape  # (1, n_experts, n_nm, feature_dim)
        
        expert_outputs = []
        for e in range(E):
            expert_input = expert_features[:, e, :, :]  # (1, n_nm, feature_dim)
            # 使用对应专家的decoder
            expert_output = self.decoder_experts[e](expert_input)  # (1, n_nm, out_dim)
            expert_outputs.append(expert_output)
        
        # stack: (1, n_experts, n_nm, out_dim)
        nonmanifold_pnts_pred = torch.stack(expert_outputs, dim=1)  # (1, n_experts, n_nm, out_dim)
        
        # 如果out_dim=1，squeeze最后一个维度: (1, n_experts, n_nm)
        if nonmanifold_pnts_pred.shape[-1] == 1:
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.squeeze(-1)  # (1, n_experts, n_nm)
        
        # 确保形状正确
        if nonmanifold_pnts_pred.dim() != 3:
            raise ValueError(f"nonmanifold_pnts_pred形状不正确: {nonmanifold_pnts_pred.shape}")
        
        # 基于哈希的门控网络
        nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q = self.manager_net(non_mnfld_pnts)
        # nonmnfld_q: (1, n_nm, n_experts)
        # nonmnfld_selected_expert_idx: (1, n_nm)
        
        # 转置为 (1, n_experts, n_nm)
        nonmnfld_q = nonmnfld_q.permute(0, 2, 1)  # (1, n_experts, n_nm)
        nonmnfld_selected_expert_idx = nonmnfld_selected_expert_idx.squeeze(0)  # (n_nm,)
        
        # 选择专家预测（用于推断时）
        # nonmanifold_pnts_pred: (1, n_experts, n_nm)
        # nonmnfld_selected_expert_idx: (n_nm,)
        gather_idx = nonmnfld_selected_expert_idx[None, :, None]  # (1, n_nm, 1)
        selected_nonmanifold_pnts_pred = torch.gather(
            nonmanifold_pnts_pred, dim=1,  # 在experts维度上gather
            index=gather_idx
        ).squeeze(1)  # (1, n_nm)
        
        # 流形点处理（如果有）
        if mnfld_pnts is not None:
            mnfld_expert_features = self.heterogeneous_hash_encoder(mnfld_pnts)
            
            mnfld_expert_outputs = []
            for e in range(E):
                mnfld_expert_input = mnfld_expert_features[:, e, :, :]  # (1, n_m, feature_dim)
                # 使用对应专家的decoder
                mnfld_expert_output = self.decoder_experts[e](mnfld_expert_input)  # (1, n_m, out_dim)
                mnfld_expert_outputs.append(mnfld_expert_output)
            
            # stack: (1, n_experts, n_m, out_dim)
            manifold_pnts_pred = torch.stack(mnfld_expert_outputs, dim=1)  # (1, n_experts, n_m, out_dim)
            if manifold_pnts_pred.shape[-1] == 1:
                manifold_pnts_pred = manifold_pnts_pred.squeeze(-1)  # (1, n_experts, n_m)
            
            mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q = self.manager_net(mnfld_pnts)
            mnfld_q = mnfld_q.permute(0, 2, 1)  # (1, n_experts, n_m)
            mnfld_selected_expert_idx = mnfld_selected_expert_idx.squeeze(0)  # (n_m,)
            
            gather_idx_mnfld = mnfld_selected_expert_idx[None, :, None]
            selected_manifold_pnts_pred = torch.gather(
                manifold_pnts_pred, dim=1,
                index=gather_idx_mnfld
            ).squeeze(1)  # (1, n_m)
        else:
            manifold_pnts_pred = None
            mnfld_q = None
            mnfld_selected_expert_idx = None
            selected_manifold_pnts_pred = None
            mnfld_raw_q = None
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,  # (1, n_experts, n_m)
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,  # (1, n_experts, n_nm)
            "mnfld_q": mnfld_q,  # (1, n_experts, n_m)
            "nonmnfld_q": nonmnfld_q,  # (1, n_experts, n_nm)
            "selected_manifold_pnts_pred": selected_manifold_pnts_pred,  # (1, n_m)
            "selected_nonmanifold_pnts_pred": selected_nonmanifold_pnts_pred,  # (1, n_nm)
            "mnfld_selected_expert_idx": mnfld_selected_expert_idx,  # (n_m,)
            "nonmnfld_selected_expert_idx": nonmnfld_selected_expert_idx,  # (n_nm,)
            "mnfld_raw_q": mnfld_raw_q if mnfld_pnts is not None else None,
            "nonmnfld_raw_q": nonmnfld_raw_q,
        }

