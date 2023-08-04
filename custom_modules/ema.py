import copy
import itertools
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner, ValLoop
from mmdet.registry import MODELS, HOOKS, LOOPS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector


@MODELS.register_module()
class MultiEMADetector(BaseDetector):
    """Keep multiple EMA models during training.
    Modified from mmdet.models.detectors.semi_base.SemiBaseDetector
    """
    def __init__(self,
                 momentums: List[float],
                 detector: ConfigType,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=detector.data_preprocessor,
                         init_cfg=init_cfg)
        self.orig_model = MODELS.build(copy.deepcopy(detector))
        assert len(momentums) > 0
        self.momentums = momentums
        self.ema_models = nn.ModuleList()
        for _ in self.momentums:
            ema_model = MODELS.build(copy.deepcopy(detector))
            self.freeze(ema_model)
            self.ema_models.append(ema_model)

        self.predict_on = 2#'orig_model'

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, *args, **kwargs) -> dict:
        losses = self.orig_model.loss(*args, **kwargs)
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        if isinstance(self.predict_on, str):
            assert self.predict_on == 'orig_model'
            out = self.orig_model(batch_inputs, batch_data_samples, mode='predict')
        elif isinstance(self.predict_on, int):
            assert 0 <= self.predict_on < len(self.momentums)
            out = self.ema_models[self.predict_on](batch_inputs, batch_data_samples, mode='predict')
        else:
            raise ValueError()
        return out

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        return self.orig_model(batch_inputs,
                                batch_data_samples,
                                mode='tensor')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        return self.orig_model.extract_feat(batch_inputs)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Add ema_model and orig_model prefixes to model parameter names."""
        if not any([
                'orig_model' in key or 'ema_model' in key
                for key in state_dict.keys()
        ]):
            keys = list(state_dict.keys())
            for idx, _ in enumerate(self.momentums):
                state_dict.update({f'ema_models.{idx}.' + k: state_dict[k] for k in keys})
            state_dict.update({'orig_model.' + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


@HOOKS.register_module()
class MultiEMAHook(Hook):
    def __init__(self,
                 interval: int = 1,
                 skip_buffers=True) -> None:
        self.interval = interval
        self.skip_buffers = skip_buffers

    def before_train(self, runner: Runner) -> None:
        """To check that ema_model model and orig_model model exist."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'ema_models')
        assert hasattr(model, 'orig_model')
        # only do it at initial stage
        if runner.iter == 0:
            orig_model = model.orig_model
            for ema_model in model.ema_models:
                self.momentum_update(orig_model, ema_model, 1)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ema_model's parameter every self.interval iterations."""
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        orig_model = model.orig_model
        for momentum, ema_model in zip(model.momentums, model.ema_models):
            self.momentum_update(orig_model, ema_model, momentum)

    def momentum_update(self, orig_model, ema_model, momentum) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
        src_params = orig_model.parameters()
        dst_params = ema_model.parameters()
        if not self.skip_buffers:
            src_params = itertools.chain(src_params, orig_model.buffers())
            dst_params = itertools.chain(dst_params, ema_model.buffers())

        # for (src_parm, dst_parm) in zip(src_params, dst_params):
        #     if dst_parm.dtype.is_floating_point:
        #         dst_parm.data.lerp_(src_parm.data, momentum)
        src_params = [p for p in src_params if p.dtype.is_floating_point]
        dst_params = [p for p in dst_params if p.dtype.is_floating_point]
        torch._foreach_lerp_(dst_params, src_params, momentum)


@LOOPS.register_module()
class MultiEMAValLoop(ValLoop):

    def run(self):
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'ema_models')
        assert hasattr(model, 'orig_model')

        saved_predict_on = model.predict_on
        multi_metrics = dict()
        for _predict_on in list(range(len(model.ema_models))) +  ['orig_model']:
            model.predict_on = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if _predict_on == 'orig_model':
                multi_metrics.update(
                    {'/'.join((_predict_on, k)): v
                    for k, v in metrics.items()})
            else:
                multi_metrics.update(
                    {'/'.join((f'ema_model.{_predict_on}', k)): v
                    for k, v in metrics.items()})
        model.predict_on = saved_predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')
