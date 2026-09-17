#!/usr/bin/env python3
"""
支持KAN网络的INR_MoE实现
将专家网络中的MLP替换为KAN网络
"""

import numpy as np
import torch
import torch.nn as nn
import models.managers as managers
from models.kan_modules_simple import ParallelSimpleKANNetwork, SimpleKANInputEncoder, create_parallel_simple_kan_experts
from models.modules import (FullyConnectedNN, InputEncoder, tSoftMax, DummyModule)
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class Manager(nn.Module):
    def __init__(self, cfg, in_dim, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = cfg['in_dim']
        self.temperature = cfg['manager_softmax_temperature']
        
        # 激活函数字典
        self.q_activation_dict = {'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
                                  'sigmoid': nn.Sigmoid(), 'none': DummyModule()}
        
        # Manager网络类型字典
        self.manager_type_dict = {'none': lambda: managers.DummyManager(self.n_experts),
                                  'standard': lambda: FullyConnectedNN(in_dim, self.n_experts,
                                                   num_hidden_layers=cfg['manager_n_hidden_layers'],
                                                   hidden_features=cfg['manager_hidden_dim'], outermost_linear=True,
                                                   nonlinearity=cfg['manager_nl'], init_type=cfg['manager_init'],
                                                   input_encoding=cfg['manager_input_encoding'],
                                                   module_name=module_name + '.manager_net'),
                                  'kan': lambda: create_parallel_simple_kan_experts(1, in_dim, self.n_experts,
                                                   hidden_dims=[cfg['manager_hidden_dim']] * cfg['manager_n_hidden_layers'],
                                                   num_basis=cfg.get('kan_num_basis', 4),
                                                   activation=cfg.get('kan_activation', 'relu'),
                                                   dropout=cfg.get('kan_dropout', 0.1),
                                                   module_name=module_name + '.manager_net')
                                  }

        self.manager_type = cfg['manager_type']
        self.manager_net = self.manager_type_dict.get(self.manager_type, lambda: None)()
        if self.manager_net is None:
            raise ValueError("Unsupported manager type")

        self.q_activation = self.q_activation_dict.get(cfg['manager_q_activation'], DummyModule())
        self.clamp_q = cfg['manager_clamp_q']

    def forward(self, points):
        if self.manager_type == 'kan':
            # KAN Manager网络
            raw_q = self.manager_net(points)  # (1, 1, n_points, n_experts)
            raw_q = raw_q.squeeze(1).squeeze(-1)  # (1, n_points, n_experts)
        else:
            # 传统Manager网络
            raw_q = self.manager_net(points)

        q = self.q_activation(raw_q)
        q = torch.clamp(q, self.clamp_q)

        selected_expert_idx = torch.argmax(q, 1)
        return q, selected_expert_idx, raw_q.T.unsqueeze(0)


class INR_MoE_KAN(nn.Module):
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        self.init_type = cfg['decoder_init_type']
        self.n_experts = cfg['n_experts']
        self.manager_init = cfg['manager_init']
        self.share_encoder = cfg['shared_encoder']
        
        # KAN配置参数
        self.kan_config = {
            'num_basis': cfg.get('kan_num_basis', 4),
            'activation': cfg.get('kan_activation', 'relu'),
            'dropout': cfg.get('kan_dropout', 0.1)
        }
        
        # 输入编码模块 - 使用传统编码器
        self.decoder_input_encoding_module = InputEncoder(cfg, cfg['decoder_input_encoding'], cfg['decoder_hidden_dim'],
                                                       module_name='decoder_input_encoding_module')
        
        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim

        # 专家网络（Decoder）- 使用KAN网络
        hidden_dims = [cfg['decoder_hidden_dim']] * cfg['decoder_n_hidden_layers']
        self.decoder = create_parallel_simple_kan_experts(
            self.n_experts, decoder_first_layer_dim, cfg['out_dim'],
            hidden_dims=hidden_dims,
            num_basis=self.kan_config['num_basis'],
            activation=self.kan_config['activation'],
            dropout=self.kan_config['dropout'],
            module_name='decoder'
        )

        # Manager网络
        self.manager_conditioning = cfg['manager_conditioning']
        self.manager_conditioner = managers.ManagerConditioner(cfg['manager_conditioning'], decoder_first_layer_dim, self.decoder)
        
        if not self.share_encoder:
            # Manager使用传统编码器
            self.manager_input_encoding_module = InputEncoder(cfg, cfg['manager_input_encoding'],
                                                               cfg['manager_hidden_dim'],
                                                               module_name='manager_input_encoding_module')
        
        manager_first_layer_dim = self.manager_input_encoding_module.first_layer_dim
        manager_first_layer_dim = manager_first_layer_dim + 1024 if cfg['manager_type'] == 'pointnet' else manager_first_layer_dim
        manager_first_layer_dim = manager_first_layer_dim + decoder_first_layer_dim if not self.manager_conditioning == 'none' else manager_first_layer_dim
        
        self.manager_net = Manager(cfg, manager_first_layer_dim)

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        # non_mnfld_pnts: (1,n_nm,d), mnfld_pnts: (1, n_m, d)
        # d is input dim (2 or 3), k is num_experts
        
        encoded_non_mnfld_pnts_experts = self.decoder_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        # Manifold points
        if mnfld_pnts is not None:
            encoded_mnfld_pnts_experts = self.decoder_input_encoding_module(mnfld_pnts, **kwargs)
            manifold_pnts_pred = self.decoder(encoded_mnfld_pnts_experts)  # (1,k,n_m,1)
            
            # 确保输出形状正确 - KAN网络输出形状处理
            if manifold_pnts_pred.shape[0] == self.n_experts:  # (k, n_m, 1)
                manifold_pnts_pred = manifold_pnts_pred.unsqueeze(0)  # (1, k, n_m, 1)
            elif manifold_pnts_pred.dim() == 4 and manifold_pnts_pred.shape[-1] != 1:
                manifold_pnts_pred = manifold_pnts_pred.mean(dim=-1, keepdim=True)  # (1, k, n_m, 1)
            
            manifold_pnts_pred = manifold_pnts_pred.squeeze(-1)  # (1,k,n_m)
            
            if not self.share_encoder:
                manager_input = self.manager_input_encoding_module(mnfld_pnts, **kwargs)

            manager_input = self.manager_conditioner(encoded_mnfld_pnts_experts, manager_input, **kwargs)

            mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q, = self.manager_net(manager_input.view(-1, manager_input.shape[-1])) # (n_m,k), (n_m,)
            mnfld_q = mnfld_q.T.unsqueeze(0) # (1,k,n_m)

            # 修复manifold points的torch.gather操作
            if manifold_pnts_pred.dim() == 3:  # (1, k, n_m)
                expert_indices = mnfld_selected_expert_idx[None, None, :]  # (1, 1, n_m)
                selected_manifold_pnts_pred = torch.gather(manifold_pnts_pred, dim=1, index=expert_indices).squeeze(1)  # (1, n_m)
            else:
                selected_manifold_pnts_pred = manifold_pnts_pred.mean(dim=1)  # (1, n_m)
        else:
            manifold_pnts_pred = None
            mnfld_q, mnfld_selected_expert_idx, selected_manifold_pnts_pred = None, None, None
            mnfld_raw_q = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(encoded_non_mnfld_pnts_experts) # (1,k,n_nm,d)
        
        # 确保输出形状正确 - KAN网络输出形状处理
        if nonmanifold_pnts_pred.shape[0] == self.n_experts:  # (k, n_nm, 1)
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.unsqueeze(0)  # (1, k, n_nm, 1)
        elif nonmanifold_pnts_pred.dim() == 4 and nonmanifold_pnts_pred.shape[-1] != 1:
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.mean(dim=-1, keepdim=True)  # (1, k, n_nm, 1)
        elif nonmanifold_pnts_pred.dim() == 3:  # (1, k, n_nm)
            nonmanifold_pnts_pred = nonmanifold_pnts_pred.unsqueeze(-1)  # (1, k, n_nm, 1)

        if not self.share_encoder:
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts, **kwargs)

        nonmnld_manager_input = encoded_non_mnfld_pnts_manager # (1,n_nm,d)

        nonmnld_manager_input = self.manager_conditioner(encoded_non_mnfld_pnts_experts, nonmnld_manager_input, **kwargs) # (1,n_nm,d)

        nonmnld_manager_input = nonmnld_manager_input.reshape(-1, nonmnld_manager_input.shape[-1])
        mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q, = self.manager_net(nonmnld_manager_input) # (n_nm, k), (n_nm,)
        nonmnfld_q = mout_nonmnfld_q.T.unsqueeze(0) # (1,k,n_nm)

        # 修复torch.gather的维度问题
        if nonmanifold_pnts_pred.dim() == 4:  # (1, k, n_nm, 1)
            expert_indices = nonmnfld_selected_expert_idx[None, None, :, None]  # (1, 1, n_nm, 1)
            expert_indices = expert_indices.expand(1, 1, -1, nonmanifold_pnts_pred.shape[-1])  # (1, 1, n_nm, 1)
            selected_nonmanifold_pnts_pred = torch.gather(nonmanifold_pnts_pred, dim=1, index=expert_indices).squeeze(1).squeeze(-1)  # (1, n_nm)
        else:  # (1, k, n_nm)
            expert_indices = nonmnfld_selected_expert_idx[None, None, :]  # (1, 1, n_nm)
            selected_nonmanifold_pnts_pred = torch.gather(nonmanifold_pnts_pred, dim=1, index=expert_indices).squeeze(1)  # (1, n_nm)

        return {"manifold_pnts_pred": manifold_pnts_pred,                           # (1,k,n_m)
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,                     # (1,k,n_nm)
                "mnfld_q": mnfld_q,                                                 # (1,k,n_m)
                "nonmnfld_q": nonmnfld_q,                                           # (1,k,n_nm)
                "selected_manifold_pnts_pred": selected_manifold_pnts_pred,         # (1,n_m)
                "selected_nonmanifold_pnts_pred": selected_nonmanifold_pnts_pred,   # (1,n_nm)
                "mnfld_selected_expert_idx": mnfld_selected_expert_idx,             # (n_m,)
                "nonmnfld_selected_expert_idx": nonmnfld_selected_expert_idx,       # (n_nm,)
                "mnfld_raw_q": mnfld_raw_q,
                "nonmnfld_raw_q": nonmnfld_raw_q,
                "nonmnld_manager_input": nonmnld_manager_input,
                "mout_nonmnfld_q": mout_nonmnfld_q,
                }


# 兼容性函数
def build_kan_model(cfg_all, loss_cfg):
    """构建KAN模型"""
    model = INR_MoE_KAN(cfg_all)
    return model, loss_cfg








