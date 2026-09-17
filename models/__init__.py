from .INR import INR
from .INR_MoE import INR_MoE
from .INR_MoE_SimpleKAN import INR_MoE_SimpleKAN
from .INR_UltraNeRF import INR_UltraNeRF
from .rgb_losses import RGBImageLoss
from .audio_losses import AudioLoss

# SDF loss requires heavy 3D deps (trimesh, pyrender, etc.).
# Load it lazily so RGB-only tasks don't need those packages.
try:
    from .sdf_losses import SDFShapeLoss
except ImportError:
    SDFShapeLoss = None

import importlib
import sys
import os

__all__ = {
    'inr_sdf': INR,
    'inr_moe_sdf': INR_MoE,
    'inr_rgb': INR,
    'inr_moe_rgb': INR_MoE,  # 支持文本引导（use_text_guidance: true）
    'inr_moe_audio': INR_MoE,
    'inr_audio': INR,
    'inr_moe_simplekan_rgb': INR_MoE_SimpleKAN,
    'inr_ultranerf_rgb': INR_UltraNeRF,
}

__losses__ = {
    'inr_sdf': SDFShapeLoss if SDFShapeLoss is not None else None,
    'inr_moe_sdf': SDFShapeLoss if SDFShapeLoss is not None else None,
    'inr_rgb': RGBImageLoss,
    'inr_moe_rgb': RGBImageLoss,
    'inr_sparsemoe_rgb': RGBImageLoss,
    'inr_sparseidfmoe_rgb': RGBImageLoss,
    'inr_moe_audio': AudioLoss,
    'inr_audio': AudioLoss,
    'inr_moe_simplekan_rgb': RGBImageLoss,
    'inr_ultranerf_rgb': RGBImageLoss,
}

file_name_dict = {
    'inr_sdf': "INR.py",
    'inr_moe_sdf': "INR_MoE.py",
    'inr_rgb': "INR.py",
    'inr_moe_rgb': "INR_MoE.py",
    'inr_moe_audio': "INR_MoE.py",
    'inr_audio': "INR.py",
    'inr_moe_simplekan_rgb': "INR_MoE_SimpleKAN.py",
}

def build_model(cfg, loss_cfg):
    model_cfg = cfg['MODEL']
    model = __all__[model_cfg['model_name']](cfg_all=cfg)
    loss = __losses__[model_cfg['model_name']](cfg=loss_cfg,
                                               model_name=model_cfg['model_name'],
                                               model=model, n_experts=cfg['MODEL']['n_experts'])
    # all 和 loss是两个字典类
    return model, loss

class build_model_from_logdir(object):
    def __init__(self, logdir, cfg, loss_cfg):
        model_cfg = cfg['MODEL']
        pc_model = model_cfg.get('model_name')
        model_instance = __all__[pc_model]
        model_name = model_instance.__name__
        file_name = file_name_dict.get(pc_model)
        spec = importlib.util.spec_from_file_location(model_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = import_model
        spec.loader.exec_module(import_model)
        self.model = model_instance(cfg_all=cfg)

        loss_model = model_cfg.get('model_name')
        loss_instance = __losses__[loss_model]
        loss_name = loss_instance.__name__
        file_name = file_name_dict.get(loss_model)
        spec = importlib.util.spec_from_file_location(loss_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[loss_name] = import_model
        spec.loader.exec_module(import_model)
        self.loss = loss_instance(cfg=loss_cfg, model_name=model_cfg['model_name'],
                                  model=self.model, n_experts=cfg['MODEL']['n_experts'])

    def get(self):
        return self.model, self.loss