from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.dense_heads import RTMDetInsHead, RTMDetInsSepBNHead
from mmdet.models.dense_heads.rtmdet_ins_head import MaskFeatModule
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import sigmoid_geometric_mean, multi_apply
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean



@MODELS.register_module()
class RTMDetInsHeadFixes(RTMDetInsHead):
    def __init__(self, *args, exp_on_reg=False, fix_downsample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_on_reg = exp_on_reg
        self.fix_downsample = fix_downsample

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
        if not self.fix_downsample:
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

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     kernel_preds: List[Tensor],
                     mask_feat: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_kernels = torch.cat([
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_gen_params)
            for kernel_pred in kernel_preds
        ], 1)
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        # MODIFIED: convert in data pipeline to speedup training
        # for gt_instances in batch_gt_instances:
        #     gt_instances.masks = gt_instances.masks.to_tensor(
        #         dtype=torch.bool, device=device)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                decoded_bboxes,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                assign_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels,
                                           sampling_results_list,
                                           batch_gt_instances)
        loss = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mask=loss_mask)
        return loss


@MODELS.register_module()
class RTMDetInsSepBNHeadFixes(RTMDetInsSepBNHead):
    def __init__(self,
                 *args,
                 exp_on_reg=False,
                 fix_downsample=False,
                 fix_reg_conv_init=False,
                 **kwargs):
        self.exp_on_reg = exp_on_reg
        self.fix_downsample = fix_downsample
        self.fix_reg_conv_init = fix_reg_conv_init
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                kernel_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            if self.fix_reg_conv_init:
                self.reg_convs.append(reg_convs)
            else:
                self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_kernel.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_gen_params,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=pred_pad_size))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.prior_generator.strides),
            num_prototypes=self.num_prototypes,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)


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
        if not self.fix_downsample:
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
    

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     kernel_preds: List[Tensor],
                     mask_feat: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_kernels = torch.cat([
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_gen_params)
            for kernel_pred in kernel_preds
        ], 1)
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        # MODIFIED: convert in data pipeline to speedup training
        # for gt_instances in batch_gt_instances:
        #     gt_instances.masks = gt_instances.masks.to_tensor(
        #         dtype=torch.bool, device=device)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        losses_cls, losses_bbox,\
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                decoded_bboxes,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                assign_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels,
                                           sampling_results_list,
                                           batch_gt_instances)
        loss = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mask=loss_mask)
        return loss
