#!/usr/bin/env python3
"""
多专家混合哈希编码的INR MoE模型
前N-1层哈希编码共用，最后一层哈希编码使用多专家路由
"""

import torch
import torch.nn as nn
import models.managers as managers
from models.modules import InputEncoder, tSoftMax, DummyModule, FullyConnectedNN, ParallelFullyConnectedNN, MultiExpertHashEncoding
import math


class Manager(nn.Module):
    """Manager网络：负责选择专家"""
    def __init__(self, cfg, in_dim, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = cfg['in_dim']
        self.temperature = float(cfg['manager_softmax_temperature'])
        
        # 激活函数字典
        self.q_activation_dict = {
            'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
            'sigmoid': nn.Sigmoid(), 
            'none': DummyModule()
        }
        
        self.manager_type_dict = {
            'none': lambda: managers.DummyManager(self.n_experts),
            'standard': lambda: FullyConnectedNN(
                in_dim, self.n_experts,
                num_hidden_layers=cfg['manager_n_hidden_layers'],
                hidden_features=cfg['manager_hidden_dim'], 
                outermost_linear=True,
                nonlinearity=cfg['manager_nl'], 
                init_type=cfg['manager_init'],
                input_encoding=cfg['manager_input_encoding'],
                module_name=module_name + '.manager_net'
            ),
        }
        
        self.manager_type = cfg['manager_type']
        self.manager_net = self.manager_type_dict.get(self.manager_type, lambda: None)()
        if self.manager_net is None:
            raise ValueError("不支持的manager类型")
        
        self.q_activation = self.q_activation_dict.get(cfg['manager_q_activation'], DummyModule())
        self.clamp_q = float(cfg['manager_clamp_q'])
        
        # 添加：Auxiliary-Loss-Free Load Balancing - 专家偏置
        # 使用register_buffer作为非可学习参数
        self.use_expert_bias = cfg.get('use_expert_bias_for_load_balancing', False)
        if self.use_expert_bias:
            self.register_buffer('expert_biases', torch.zeros(self.n_experts))
            self.bias_update_rate = cfg.get('expert_bias_update_rate', 0.01)
            print(f"[Manager] 启用专家偏置负载均衡策略 (Auxiliary-Loss-Free)")
            print(f"  - 专家数量: {self.n_experts}")
            print(f"  - 偏置更新率: {self.bias_update_rate}")
    
    def forward(self, points):
        raw_q = self.manager_net(points)
        
        # 添加：在softmax之前添加专家偏置
        if self.use_expert_bias:
            # raw_q: (B, N, n_experts) 或 (N, n_experts)
            # expert_biases: (n_experts,)
            if raw_q.dim() == 3:  # (B, N, n_experts)
                raw_q = raw_q + self.expert_biases.unsqueeze(0).unsqueeze(0)
            elif raw_q.dim() == 2:  # (N, n_experts)
                raw_q = raw_q + self.expert_biases.unsqueeze(0)
            else:
                raise ValueError(f"意外的raw_q形状: {raw_q.shape}")
        
        q = self.q_activation(raw_q)
        q = torch.clamp(q, self.clamp_q)
        selected_expert_idx = torch.argmax(q, 1)
        return q, selected_expert_idx, raw_q.T.unsqueeze(0)
    
    def update_expert_biases(self, expert_usage):
        """
        根据专家使用情况更新偏置（Auxiliary-Loss-Free Load Balancing）
        
        Args:
            expert_usage: 每个专家的使用频率 (n_experts,) 或 (B, n_experts)
        """
        if not self.use_expert_bias:
            return
        
        # 确保是1维tensor: (n_experts,)
        if expert_usage.dim() == 2:  # (B, n_experts)
            expert_usage = expert_usage.mean(dim=0)  # 平均跨batch
        elif expert_usage.dim() > 2:
            expert_usage = expert_usage.flatten()
            if len(expert_usage) != self.n_experts:
                # 如果是多个batch的平均值，取前n_experts个
                expert_usage = expert_usage[:self.n_experts]
        
        # 目标使用频率：每个专家应该使用1/n_experts
        target_usage = 1.0 / self.n_experts
        
        # 根据负载情况更新偏置
        with torch.no_grad():
            for i in range(self.n_experts):
                if expert_usage[i] > target_usage:
                    # 负载过高，降低偏置（减少被选概率）
                    self.expert_biases[i] -= self.bias_update_rate
                else:
                    # 负载过低，增加偏置（增加被选概率）
                    self.expert_biases[i] += self.bias_update_rate


class MultiExpertHashInputEncoder(nn.Module):
    """支持多专家混合哈希编码的输入编码器"""
    def __init__(self, cfg, input_encoding, hidden_dim, module_name=''):
        super().__init__()
        self.input_encoding = input_encoding
        hidden_features = hidden_dim
        in_features = cfg['in_dim']
        self.first_layer_dim = in_features
        self.multi_expert_hash_encoder = None
        
        # 多专家混合哈希编码
        if 'HE_MoE' in self.input_encoding:
            n_levels = cfg.get('hash_n_levels', 12)
            n_experts = cfg.get('n_experts', 4)
            n_features_per_level = cfg.get('hash_n_features_per_level', 2)
            log2_hashmap_size = cfg.get('hash_log2_hashmap_size', 14)
            base_resolution = cfg.get('hash_base_resolution', 16)
            finest_resolution = cfg.get('hash_finest_resolution', 2048)
            shared_levels = cfg.get('hash_shared_levels', 6)  # 前几层共享，默认6层
            
            self.multi_expert_hash_encoder = MultiExpertHashEncoding(
                n_levels=n_levels,
                n_experts=n_experts,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=log2_hashmap_size,
                base_resolution=base_resolution,
                finest_resolution=finest_resolution,
                input_dim=cfg['in_dim'],
                shared_levels=shared_levels
            )
            
            # 前shared_levels层的共用特征维度 + 后expert_levels层每个专家的特征维度
            shared_output_dim = shared_levels * n_features_per_level
            expert_levels = n_levels - shared_levels
            expert_output_dim = expert_levels * n_features_per_level
            self.first_layer_dim = shared_output_dim + expert_output_dim  # 每个专家的总输入维度
            
            print(f"[MultiExpertHashInputEncoder] 混合哈希编码初始化:")
            print(f"  - Levels: {n_levels} (前{shared_levels}层共用，后{expert_levels}层多专家)")
            print(f"  - Experts: {n_experts}")
            print(f"  - Features per level: {n_features_per_level}")
            print(f"  - Hashmap size: 2^{log2_hashmap_size} = {2**log2_hashmap_size}")
            print(f"  - Resolution range: [{base_resolution}, {finest_resolution}]")
            print(f"  - Shared output dim: {shared_output_dim}")
            print(f"  - Expert output dim: {expert_output_dim}")
            print(f"  - First layer dim (per expert): {self.first_layer_dim}")
        else:
            raise ValueError(f"不支持这种编码方式: {input_encoding}，需要使用 'HE_MoE'")
    
    def forward(self, coords, **kwargs):
        """
        coords: (B, N, input_dim) 输入坐标
        返回: 
            encoded_features: (B, n_experts, N, first_layer_dim) 每个专家的编码特征
            shared_features: (B, N, shared_output_dim) 前shared_levels层的共享特征（用于条件化）
        """
        if 'HE_MoE' in self.input_encoding:
            # 获取共用特征和专家特征
            shared_features, expert_features = self.multi_expert_hash_encoder(coords)
            # shared_features: (B, N, shared_output_dim)
            # expert_features: (B, n_experts, N, expert_output_dim)
            
            # 将共用特征广播到每个专家并拼接
            B, N = shared_features.shape[:2]
            n_experts = expert_features.shape[1]
            shared_dim = shared_features.shape[2]
            expert_dim = expert_features.shape[3]
            
            # 广播共用特征: (B, N, shared_dim) -> (B, n_experts, N, shared_dim)
            shared_expanded = shared_features.unsqueeze(1).expand(B, n_experts, N, shared_dim)
            
            # 拼接: (B, n_experts, N, shared_dim + expert_dim)
            encoded = torch.cat([shared_expanded, expert_features], dim=-1)
            
            return encoded, shared_features  # (B, n_experts, N, first_layer_dim), (B, N, shared_output_dim)
        else:
            raise ValueError(f"不支持这种编码方式: {self.input_encoding}")


class INR_MoE_HashHybrid(nn.Module):
    """
    多专家混合哈希编码的INR MoE模型
    - 前N-1层哈希编码共用（单专家）
    - 最后一层哈希编码使用多专家路由
    - Manager网络使用PE编码，不拼接专家特征
    """
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        
        self.init_type = cfg['decoder_init_type']
        self.n_experts = cfg['n_experts']
        self.manager_init = cfg['manager_init']
        self.share_encoder = cfg['shared_encoder']
        
        # 多专家混合哈希编码模块
        self.decoder_input_encoding_module = MultiExpertHashInputEncoder(
            cfg, 'HE_MoE', cfg['decoder_hidden_dim'],
            module_name='decoder_input_encoding_module'
        )
        
        # decoder_first_layer_dim 是每个专家的输入维度
        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        
        # 解码器网络（每个专家独立）
        self.decoder = ParallelFullyConnectedNN(
            self.n_experts, decoder_first_layer_dim, cfg['out_dim'],
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
        
        # Manager条件化（拼接前6层共享哈希编码特征）
        self.manager_conditioning = cfg['manager_conditioning']
        
        # 获取共享特征维度（前shared_levels层的特征）
        shared_levels = cfg.get('hash_shared_levels', 6)
        n_features_per_level = cfg.get('hash_n_features_per_level', 2)
        shared_output_dim = shared_levels * n_features_per_level
        
        # Manager条件化器：根据配置决定是否拼接共享特征
        self.manager_conditioner = managers.ManagerConditioner(
            self.manager_conditioning,  # 可以是 'cat', 'mean', 'max' 等
            shared_output_dim,  # 使用共享特征维度而不是完整特征维度
            self.decoder
        )
        
        # Manager输入编码（只使用PE编码）
        if not self.share_encoder:
            self.manager_input_encoding_module = InputEncoder(
                cfg, cfg['manager_input_encoding'],
                cfg['manager_hidden_dim'],
                module_name='manager_input_encoding_module'
            )
        else:
            # 如果共享编码器，使用解码器的编码模块
            self.manager_input_encoding_module = self.decoder_input_encoding_module
        
        manager_first_layer_dim = self.manager_input_encoding_module.first_layer_dim
        
        # 如果启用条件化，需要增加共享特征的维度
        if self.manager_conditioning != 'none':
            manager_first_layer_dim = manager_first_layer_dim + shared_output_dim
        
        self.manager_net = Manager(cfg, manager_first_layer_dim)
    
    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        """
        Args:
            non_mnfld_pnts: 非流形点 (1, n_nm, d)
            mnfld_pnts: 流形点 (1, n_m, d)，可选
            **kwargs: 其他参数（如dino, img等）
        """
        # 编码非流形点（多专家混合哈希编码）
        # encoded_non_mnfld_pnts_experts: (1, n_experts, n_nm, first_layer_dim)
        # shared_non_mnfld_features: (1, n_nm, shared_output_dim)
        encoded_non_mnfld_pnts_experts, shared_non_mnfld_features = self.decoder_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        # 流形点处理（如果有）
        if mnfld_pnts is not None:
            encoded_mnfld_pnts_experts, shared_mnfld_features = self.decoder_input_encoding_module(mnfld_pnts, **kwargs)
            
            # 解码器处理: (1, n_experts, n_m, first_layer_dim) -> (1, n_experts, n_m, out_dim)
            manifold_pnts_pred = self.decoder(encoded_mnfld_pnts_experts)  # (1, n_experts, n_m, out_dim)
            manifold_pnts_pred = manifold_pnts_pred.squeeze(-1) if manifold_pnts_pred.shape[-1] == 1 else manifold_pnts_pred  # (1, n_experts, n_m)
            
            if not self.share_encoder:
                manager_input = self.manager_input_encoding_module(mnfld_pnts, **kwargs)
            
            # Manager条件化（拼接前6层共享哈希编码特征）
            # shared_mnfld_features: (B, N, shared_dim)
            # manager_input: (B, N, manager_input_dim)
            # ManagerConditioner对于'cat'模式，期望x和manager_input的前两个维度匹配
            # 对于共享特征，我们直接使用 (B, N, shared_dim) 格式
            manager_input = self.manager_conditioner(shared_mnfld_features, manager_input, **kwargs)
            
            # Manager网络前向传播
            mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q = self.manager_net(
                manager_input.view(-1, manager_input.shape[-1])
            )  # (n_m, n_experts), (n_m,), ...
            mnfld_q = mnfld_q.T.unsqueeze(0)  # (1, n_experts, n_m)
            
            # 选择专家预测
            selected_manifold_pnts_pred = torch.gather(
                manifold_pnts_pred, dim=-2,
                index=mnfld_selected_expert_idx[None, None, :]
            ).squeeze(-2)  # (1, n_m)
        else:
            manifold_pnts_pred = None
            mnfld_q, mnfld_selected_expert_idx, selected_manifold_pnts_pred = None, None, None
            mnfld_raw_q = None
        
        # 非流形点处理
        # 问题：ParallelFullyConnectedNN期望输入是 (..., n, d) 格式，即最后两个维度是 (n_points, feature_dim)
        # 但 MultiExpertHashInputEncoder 返回 (1, n_experts, n_nm, feature_dim) - 4维
        # 
        # 解决方案：对每个专家分别调用decoder（避免显存爆炸）
        # 如果一次性reshape所有点，点数会放大n_experts倍，导致显存不足
        
        B, E, N, D = encoded_non_mnfld_pnts_experts.shape  # (1, n_experts, n_nm, feature_dim)
        
        # 使用循环分别处理每个专家，避免显存问题
        # 对每个专家的n_nm个点调用decoder，得到 (1, k, n_nm, out_dim)
        expert_outputs = []
        for e in range(E):
            expert_input = encoded_non_mnfld_pnts_experts[:, e, :, :]  # (1, n_nm, feature_dim)
            expert_output = self.decoder(expert_input)  # (1, k, n_nm, out_dim)
            expert_outputs.append(expert_output)
        
        # stack: (1, n_experts, k, n_nm, out_dim)
        stacked = torch.stack(expert_outputs, dim=1)
        
        # permute为 (1, k, n_experts, n_nm, out_dim)
        nonmanifold_pnts_pred = stacked.permute(0, 2, 1, 3, 4)  # (1, k, n_experts, n_nm, out_dim)
        
        # 保存5维版本用于后续的gather操作
        nonmanifold_pnts_pred_5d = nonmanifold_pnts_pred  # (1, k, n_experts, n_nm, out_dim)
        
        # 对于返回给损失函数的版本，我们需要 (1, k, n_nm, out_dim)
        # 对n_experts维度求平均（保留所有专家信息）
        nonmanifold_pnts_pred = nonmanifold_pnts_pred.mean(dim=2)  # (1, k, n_nm, out_dim)
        
        # 方案3：根据每个点的位置选择对应专家的预测（需要知道点的映射关系）
        # 但目前的实现中，我们不知道每个点对应哪个专家
        
        # 暂时使用方案2（平均），这样可以保留所有专家的信息
        # 现在 nonmanifold_pnts_pred 应该是4维: (1, k, n_nm, out_dim)
        # 确认形状：应该是 (1, k, n_nm, out_dim)，而不是 (1, k, n_experts * n_nm, out_dim)
        # 检查点数：应该是 n_nm，而不是 n_experts * n_nm
        current_points = nonmanifold_pnts_pred.shape[2]
        if current_points != N:
            # 如果点数不正确，说明mean操作没有正确执行，或者之前被reshape了
            # 需要重新检查并修复
            if current_points == self.n_experts * N:
                # 点数被扩展了，需要reshape回正确的形状
                # 重新reshape并取平均
                B, K, total_points, D = nonmanifold_pnts_pred.shape
                nonmanifold_pnts_pred = nonmanifold_pnts_pred.reshape(B, K, self.n_experts, N, D)
                nonmanifold_pnts_pred = nonmanifold_pnts_pred.mean(dim=2)  # (1, k, n_nm, out_dim)
            else:
                raise ValueError(f"点数维度错误：期望 {N}，但实际是 {current_points}，形状={nonmanifold_pnts_pred.shape}")
        
        # 确保形状是 (1, k, n_nm, out_dim)
        assert nonmanifold_pnts_pred.shape == (1, self.n_experts, N, nonmanifold_pnts_pred.shape[-1]), \
            f"nonmanifold_pnts_pred形状不正确: {nonmanifold_pnts_pred.shape}，期望 (1, {self.n_experts}, {N}, {nonmanifold_pnts_pred.shape[-1]})"
        
        # Manager输入编码
        if not self.share_encoder:
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts, **kwargs)
        else:
            # 如果共享编码器，需要使用PE编码而不是混合哈希编码
            # 这里我们需要一个单独的PE编码器
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        nonmnld_manager_input = encoded_non_mnfld_pnts_manager
        
        # Manager条件化（拼接前6层共享哈希编码特征）
        # shared_non_mnfld_features: (B, N, shared_dim)
        # nonmnld_manager_input: (B, N, manager_input_dim)
        # ManagerConditioner对于'cat'模式，期望x和manager_input的前两个维度匹配
        # 对于共享特征，我们直接使用 (B, N, shared_dim) 格式
        nonmnld_manager_input = self.manager_conditioner(shared_non_mnfld_features, nonmnld_manager_input, **kwargs)
        
        # Manager网络前向传播
        nonmnld_manager_input = nonmnld_manager_input.reshape(-1, nonmnld_manager_input.shape[-1])
        mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q = self.manager_net(nonmnld_manager_input)
        nonmnfld_q = mout_nonmnfld_q.T.unsqueeze(0)  # (1, n_experts, n_nm)
        
        # 选择专家预测
        # nonmanifold_pnts_pred 现在应该是4维: (1, k, n_nm, out_dim) - 已经通过mean操作处理过
        
        # 确保索引是1维的
        if nonmnfld_selected_expert_idx.dim() != 1:
            nonmnfld_selected_expert_idx = nonmnfld_selected_expert_idx.squeeze()
        
        # 检查维度：应该是 (1, k, n_nm, out_dim)
        if nonmanifold_pnts_pred.dim() != 4:
            raise ValueError(f"nonmanifold_pnts_pred应该是4维，但实际是{nonmanifold_pnts_pred.dim()}维: {nonmanifold_pnts_pred.shape}")
        
        # 确保点数正确：应该是 n_nm，而不是 n_experts * n_nm
        n_points = nonmanifold_pnts_pred.shape[2]
        idx_length = nonmnfld_selected_expert_idx.shape[0]
        
        # 如果点数不匹配，说明之前的mean操作没有正确执行
        # 检查点数是否是 n_experts * n_nm
        if n_points == self.n_experts * idx_length:
            # 说明点数被扩展了，需要reshape回正确的形状
            # 取平均值或第一个专家的预测
            B, K, total_points, D = nonmanifold_pnts_pred.shape
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.reshape(B, K, self.n_experts, idx_length, D)
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.mean(dim=2)  # (1, k, n_nm, out_dim)
            n_points = nonmanifold_pnts_pred.shape[2]
        
        if idx_length != n_points:
            raise ValueError(f"索引长度 {idx_length} 与点数 {n_points} 不匹配，nonmanifold_pnts_pred.shape={nonmanifold_pnts_pred.shape}")
        
        # 按照标准实现创建索引: (1, 1, n_nm, out_dim)
        # 在dim=1（k维度，即experts维度）上进行gather
        gather_idx = nonmnfld_selected_expert_idx[None, None, :, None].repeat(
            [1, 1, 1, nonmanifold_pnts_pred.shape[-1]]
        ).long()  # (1, 1, n_nm, out_dim)
        
        # 使用 dim=1 在experts维度上进行gather
        selected_nonmanifold_pnts_pred = torch.gather(
            nonmanifold_pnts_pred, dim=1,  # 在k维度（experts维度）上gather
            index=gather_idx
        ).squeeze(1)  # (1, n_nm, out_dim)
        
        # 确保返回的 nonmanifold_pnts_pred 形状正确: (1, k, n_nm, out_dim)
        # 检查点数：应该是 n_nm，而不是 n_experts * n_nm
        if nonmanifold_pnts_pred.shape[2] != N:
            # 如果点数不正确，重新reshape
            if nonmanifold_pnts_pred.shape[2] == self.n_experts * N:
                # 点数被扩展了，需要reshape回正确的形状
                B, K, total_points, D = nonmanifold_pnts_pred.shape
                nonmanifold_pnts_pred = nonmanifold_pnts_pred.reshape(B, K, self.n_experts, N, D)
                nonmanifold_pnts_pred = nonmanifold_pnts_pred.mean(dim=2)  # (1, k, n_nm, out_dim)
            else:
                raise ValueError(f"nonmanifold_pnts_pred的点数维度不正确: {nonmanifold_pnts_pred.shape[2]}，期望 {N} 或 {self.n_experts * N}")
        
        # 最终验证：确保形状是 (1, k, n_nm, out_dim)
        assert nonmanifold_pnts_pred.shape == (1, self.n_experts, N, nonmanifold_pnts_pred.shape[-1]), \
            f"nonmanifold_pnts_pred形状不正确: {nonmanifold_pnts_pred.shape}，期望 (1, {self.n_experts}, {N}, {nonmanifold_pnts_pred.shape[-1]})"
        
        return {
            "manifold_pnts_pred": manifold_pnts_pred,  # (1, n_experts, n_m)
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred,  # (1, n_experts, n_nm, out_dim)
            "mnfld_q": mnfld_q,  # (1, n_experts, n_m)
            "nonmnfld_q": nonmnfld_q,  # (1, n_experts, n_nm)
            "selected_manifold_pnts_pred": selected_manifold_pnts_pred,  # (1, n_m)
            "selected_nonmanifold_pnts_pred": selected_nonmanifold_pnts_pred,  # (1, n_nm)
            "mnfld_selected_expert_idx": mnfld_selected_expert_idx,  # (n_m,)
            "nonmnfld_selected_expert_idx": nonmnfld_selected_expert_idx,  # (n_nm,)
            "mnfld_raw_q": mnfld_raw_q,
            "nonmnfld_raw_q": nonmnfld_raw_q,
            "nonmnld_manager_input": nonmnld_manager_input,
            "mout_nonmnfld_q": mout_nonmnfld_q,
        }

