#!/usr/bin/env python3
"""
基于Simple KAN的INR_MoE实现
使用可学习的B样条激活函数替代传统的Sine/ReLU
"""

import torch
import torch.nn as nn
import models.managers as managers
from models.simple_kan import ParallelSimpleKAN
from models.modules import InputEncoder, tSoftMax, DummyModule, FullyConnectedNN


class Manager(nn.Module):
    """Manager网络：负责选择专家"""
    def __init__(self, cfg, in_dim, module_name=''):
        super().__init__()
        self.n_experts = cfg['n_experts']
        self.point_dim = cfg['in_dim']
        self.temperature = cfg['manager_softmax_temperature']
        
        # 激活函数
        self.q_activation_dict = {
            'softmax': tSoftMax(self.temperature, -1, cfg['manager_softmax_temp_trainable']),
            'sigmoid': nn.Sigmoid(), 
            'none': DummyModule()
        }
        
        # Manager类型
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
        self.clamp_q = cfg['manager_clamp_q']
    
    def forward(self, points):
        raw_q = self.manager_net(points)
        q = self.q_activation(raw_q)
        q = torch.clamp(q, self.clamp_q)
        selected_expert_idx = torch.argmax(q, 1)
        return q, selected_expert_idx, raw_q.T.unsqueeze(0)


class INR_MoE_SimpleKAN(nn.Module):
    """
    使用Simple KAN的INR MoE模型
    将传统的SIREN/MLP专家替换为KAN专家
    """
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        
        self.init_type = cfg.get('decoder_init_type', 'siren')
        self.n_experts = cfg['n_experts']
        self.manager_init = cfg['manager_init']
        self.share_encoder = cfg['shared_encoder']
        
        # Decoder输入编码
        self.decoder_input_encoding_module = InputEncoder(
            cfg, cfg['decoder_input_encoding'], 
            cfg['decoder_hidden_dim'],
            module_name='decoder_input_encoding_module'
        )
        decoder_first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        
        # 使用ParallelSimpleKAN替代ParallelFullyConnectedNN
        kan_grid_size = cfg.get('kan_grid_size', 8)  # B样条网格大小
        self.decoder = ParallelSimpleKAN(
            k=self.n_experts,
            in_features=decoder_first_layer_dim,
            out_features=cfg['out_dim'],
            hidden_features=cfg['decoder_hidden_dim'],
            num_hidden_layers=cfg['decoder_n_hidden_layers'],
            grid_size=kan_grid_size
        )
        
        # Manager conditioning
        self.manager_conditioning = cfg['manager_conditioning']
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
        
        manager_first_layer_dim = self.manager_input_encoding_module.first_layer_dim
        manager_first_layer_dim = (manager_first_layer_dim + 1024 
                                  if cfg['manager_type'] == 'pointnet' 
                                  else manager_first_layer_dim)
        manager_first_layer_dim = (manager_first_layer_dim + decoder_first_layer_dim 
                                  if not self.manager_conditioning == 'none' 
                                  else manager_first_layer_dim)
        
        # Manager网络
        self.manager_net = Manager(cfg, manager_first_layer_dim)
    
    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        """
        前向传播
        non_mnfld_pnts: (1,n_nm,d) - 非流形点（所有像素坐标）
        mnfld_pnts: (1, n_m, d) - 流形点（可选）
        """
        # 编码非流形点
        encoded_non_mnfld_pnts_experts = self.decoder_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        # 流形点处理（如果有）
        if mnfld_pnts is not None:
            encoded_mnfld_pnts_experts = self.decoder_input_encoding_module(mnfld_pnts, **kwargs)
            manifold_pnts_pred = self.decoder(encoded_mnfld_pnts_experts)  # (1,k,n_m,out_dim)
            manifold_pnts_pred = manifold_pnts_pred.squeeze(-1)  # (1,k,n_m)
            
            if not self.share_encoder:
                manager_input = self.manager_input_encoding_module(mnfld_pnts, **kwargs)
            
            manager_input = self.manager_conditioner(encoded_mnfld_pnts_experts, manager_input, **kwargs)
            
            mnfld_q, mnfld_selected_expert_idx, mnfld_raw_q = self.manager_net(
                manager_input.view(-1, manager_input.shape[-1])
            )
            mnfld_q = mnfld_q.T.unsqueeze(0)  # (1,k,n_m)
            
            selected_manifold_pnts_pred = torch.gather(
                manifold_pnts_pred, dim=-2,
                index=mnfld_selected_expert_idx[None, None, :]
            ).squeeze(-2)
        else:
            manifold_pnts_pred = None
            mnfld_q, mnfld_selected_expert_idx, selected_manifold_pnts_pred = None, None, None
            mnfld_raw_q = None
        
        # 非流形点处理
        nonmanifold_pnts_pred = self.decoder(encoded_non_mnfld_pnts_experts)  # (1,k,n_nm,out_dim)
        
        if not self.share_encoder:
            encoded_non_mnfld_pnts_manager = self.manager_input_encoding_module(non_mnfld_pnts, **kwargs)
        
        nonmnld_manager_input = encoded_non_mnfld_pnts_manager
        nonmnld_manager_input = self.manager_conditioner(encoded_non_mnfld_pnts_experts, nonmnld_manager_input, **kwargs)
        
        nonmnld_manager_input = nonmnld_manager_input.reshape(-1, nonmnld_manager_input.shape[-1])
        mout_nonmnfld_q, nonmnfld_selected_expert_idx, nonmnfld_raw_q = self.manager_net(nonmnld_manager_input)
        nonmnfld_q = mout_nonmnfld_q.T.unsqueeze(0)  # (1,k,n_nm)
        
        selected_nonmanifold_pnts_pred = torch.gather(
            nonmanifold_pnts_pred, dim=-3,
            index=nonmnfld_selected_expert_idx[None, None, :, None].repeat([1, 1, 1, nonmanifold_pnts_pred.shape[-1]])
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
        }


