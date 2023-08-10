from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.dense_heads import RTMDetInsHead, RTMDetInsSepBNHead
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import sigmoid_geometric_mean
from mmdet.utils import InstanceList, reduce_mean



@MODELS.register_module()
class RTMDetInsHeadFixes(RTMDetInsHead):
    def __init__(self, *args, exp_on_reg=False, correct_downsample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_on_reg = exp_on_reg
        self.correct_downsample = correct_downsample

    def loss_mask_by_feat(self, mask_feats: Tensor, flatten_kernels: Tensor,
                          sampling_results_list: list,
                          batch_gt_instances: InstanceList) -> Tensor:
        """Fix the implementation flaw of RTMDet-Ins.
        Downsample in the official implementation is wierd.
        """
        batch_pos_mask_logits = []
        pos_gt_masks = []
        for idx, (mask_feat, kernels, sampling_results,
                  gt_instances) in enumerate(
                      zip(mask_feats, flatten_kernels, sampling_results_list,
                          batch_gt_instances)):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(
                mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
            else:
                gt_masks = gt_instances.masks[
                    sampling_results.pos_assigned_gt_inds, :]
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)

        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos
                                                     ])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False).squeeze(0)
        # downsample gt masks
        if not self.correct_downsample:
            pos_gt_masks = pos_gt_masks[:, self.mask_loss_stride //
                                        2::self.mask_loss_stride,
                                        self.mask_loss_stride //
                                        2::self.mask_loss_stride]
        else:
            pos_gt_masks = F.interpolate(
                pos_gt_masks.unsqueeze(0).float(),
                scale_factor=1 / self.mask_loss_stride,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        loss_mask = self.loss_mask(
            batch_pos_mask_logits,
            pos_gt_masks,
            weight=None,
            avg_factor=num_pos)

        return loss_mask

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """The implementation of the reg_dist calculation in the RTMDet series is a mess.
        1. scale, or no scale
        2. exp, F.relu, or do nothing

        There are 2x3=6 combinations, the 4 rtm heads in used 4 different combinations.
        And there's no option to control the behaviour.

        RTMDetHead: (scale, exp), no `exp_on_reg` option
        RTMDetSepBNHead: (no scale, exp), has an `exp_on_reg` option
        RTMDetInsHead: (scale, do nothing), no `exp_on_reg` option
        RTMDetInsSepBNHead: (no scale, F.relu), no `exp_on_reg` option

        As a result, pretrained weights of RTMDetHead and RTMDetInsHead are not compatible.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for kernel_layer in self.kernel_convs:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel(kernel_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            # MODIFIED
            if self.exp_on_reg:
                reg_dist = scale(self.rtm_reg(reg_feat)) * stride[0]
            else:
                reg_dist = scale(self.rtm_reg(reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(
            kernel_preds), mask_feat



@MODELS.register_module()
class RTMDetInsSepBNHeadFixes(RTMDetInsSepBNHead):
    def __init__(self, *args, exp_on_reg=False, correct_downsample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_on_reg = exp_on_reg
        self.correct_downsample = correct_downsample

    def loss_mask_by_feat(self, mask_feats: Tensor, flatten_kernels: Tensor,
                          sampling_results_list: list,
                          batch_gt_instances: InstanceList) -> Tensor:
        """Fix the implementation flaw of RTMDet-Ins.
        Downsample in the official implementation is wierd.
        """
        batch_pos_mask_logits = []
        pos_gt_masks = []
        for idx, (mask_feat, kernels, sampling_results,
                  gt_instances) in enumerate(
                      zip(mask_feats, flatten_kernels, sampling_results_list,
                          batch_gt_instances)):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(
                mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
            else:
                gt_masks = gt_instances.masks[
                    sampling_results.pos_assigned_gt_inds, :]
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)

        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos
                                                     ])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False).squeeze(0)
        # downsample gt masks
        if not self.correct_downsample:
            pos_gt_masks = pos_gt_masks[:, self.mask_loss_stride //
                                        2::self.mask_loss_stride,
                                        self.mask_loss_stride //
                                        2::self.mask_loss_stride]
        else:
            pos_gt_masks = F.interpolate(
                pos_gt_masks.unsqueeze(0).float(),
                scale_factor=1 / self.mask_loss_stride,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        loss_mask = self.loss_mask(
            batch_pos_mask_logits,
            pos_gt_masks,
            weight=None,
            avg_factor=num_pos)

        return loss_mask

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
            else:
                reg_dist = F.relu(self.rtm_reg[idx](reg_feat)) * stride[0]

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(
            kernel_preds), mask_feat
