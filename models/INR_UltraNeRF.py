"""
Ultra-NeRF model adapted for the neural-ex coordinate-to-intensity task.
Compatible with INR_NeRF interface — maps 3D coords directly to intensity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import InputEncoder


class INR_UltraNeRF(nn.Module):
    """
    Ultra-NeRF MLP adapted for coordinate-to-intensity mapping.

    Key differences from the full Ultra-NeRF:
    - No ray rendering / physics (attenuation/reflection/scattering/PSF)
    - Outputs 1 channel (intensity) instead of 5 physics channels
    - Uses the same MLP backbone: 8-layer, 128-wide, skip at layer 4
    - Retains Ultra-NeRF's positional encoding (PE, multires=10)
    """

    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all['MODEL']
        self.init_type = cfg.get('decoder_init_type', 'siren')

        pe_num_freqs = cfg.get('pe_num_freqs', 10)
        cfg_pe = dict(cfg)
        cfg_pe['pe_num_freqs'] = pe_num_freqs
        cfg_pe['pe_linear_freqs'] = cfg.get('pe_linear_freqs', False)

        self.decoder_input_encoding_module = InputEncoder(
            cfg_pe, 'PE', cfg.get('decoder_hidden_dim', 128)
        )

        first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        decoder_hidden_dim = cfg.get('decoder_hidden_dim', 128)

        # Ultra-NeRF MLP: 8 layers, 128 hidden, skip at layer 4
        D = cfg.get('decoder_n_hidden_layers', 8)
        W = decoder_hidden_dim
        input_ch = first_layer_dim
        output_ch = cfg['out_dim']  # 1 for grayscale

        self.skips = {4}
        layers = []
        for i in range(D):
            in_features = input_ch if i == 0 else (input_ch + W if (i - 1) in self.skips else W)
            layers.append(nn.Linear(in_features, W))
        self.linears = nn.ModuleList(layers)

        # Output layer
        last_out = input_ch + W if (D - 1) in self.skips else W
        self.output_linear = nn.Linear(last_out, output_ch)

        self.activation = nn.ReLU(inplace=True) if cfg.get('decoder_nl', 'relu') == 'relu' else SineActivation()
        self.out_dim = output_ch

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def decoder(self):
        """Compatibility: expose MLP as 'decoder' for training scripts."""
        return self

    @property
    def manager_net(self):
        """Compatibility: MoE manager (not used, returns empty module)."""
        m = torch.nn.Module()
        return m

    @property
    def manager_input_encoding_module(self):
        """Compatibility: MoE manager encoder (not used, returns empty module)."""
        m = torch.nn.Module()
        return m

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        """
        Args:
            non_mnfld_pnts: (B, N, 3) — 3D coords for each pixel
        Returns:
            dict with 'nonmanifold_pnts_pred': (B, out_dim, N)
        """
        # Positional encoding
        encoded = self.decoder_input_encoding_module(non_mnfld_pnts)  # (B, N, C)

        # MLP forward with skip connections
        x_orig = encoded
        h = encoded
        for i, linear in enumerate(self.linears):
            h = self.activation(linear(h))
            if i in self.skips:
                h = torch.cat([x_orig, h], dim=-1)

        output = self.output_linear(h)  # (B, N, out_dim)

        return {
            "manifold_pnts_pred": None,
            "nonmanifold_pnts_pred": output.permute(0, 2, 1),  # (B, out_dim, N)
        }


class SineActivation(nn.Module):
    """Sine activation like SIREN."""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)
