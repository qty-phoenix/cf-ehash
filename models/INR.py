# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>


import torch.nn as nn
from models.modules import FullyConnectedNN, InputEncoder

class INR(nn.Module):
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        self.init_type = cfg['decoder_init_type']

        self.decoder_input_encoding_module = InputEncoder(cfg, cfg['decoder_input_encoding'], cfg['decoder_hidden_dim'])
        # 给了四种编码方式，除了PE，还有直接进行傅里叶编码(随机频率矩阵）DINO编码是一种特征提取网络
        first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        self.decoder = FullyConnectedNN(first_layer_dim, cfg['out_dim'],
                                             num_hidden_layers=cfg['decoder_n_hidden_layers'],
                                             hidden_features=cfg['decoder_hidden_dim'], outermost_linear=True,
                                             nonlinearity=cfg['decoder_nl'], init_type=self.init_type,
                                             input_encoding=cfg['decoder_input_encoding'],
                                             sphere_init_params=cfg['sphere_init_params'],  
                                             init_r=cfg['decoder_init_r'])  # SIREN decoder

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        # non_mnfld_pnts: (1, n_nm, d), mnfld_pnts: (1, n_m, d)

        non_mnfld_pnts = self.decoder_input_encoding_module(non_mnfld_pnts)

        if mnfld_pnts is not None:
            mnfld_pnts = self.decoder_input_encoding_module(mnfld_pnts)
            manifold_pnts_pred = self.decoder(mnfld_pnts) # (1, n_m, 1)
        else:
            manifold_pnts_pred = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts) # (1, n_nm, d)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred.permute(0, 2, 1), # (1, d, n_nm)
                }


