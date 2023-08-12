import torch
import mmengine
from mmengine.runner import load_checkpoint
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS

import custom_modules
register_all_modules()

exp_name = 'r0'
n_iter = 15120
ema_id = 2

cfg = mmengine.Config.fromfile(f'configs/{exp_name}.py')
model = MODELS.build(cfg.model)

load_checkpoint(model, f'work_dirs/{exp_name}/iter_{n_iter}.pth')
torch.save(dict(state_dict=model.ema_models[ema_id].state_dict()), f'work_dirs/{exp_name}/best.pth')
