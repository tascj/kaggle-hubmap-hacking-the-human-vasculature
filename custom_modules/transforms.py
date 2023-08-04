from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import torch
import numpy as np
from mmcv.image.geometric import cv2_border_modes, cv2_interp_codes
from torchvision.ops import roi_align
from mmcv.transforms import BaseTransform
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks


@TRANSFORMS.register_module()
class CropGtMasks(BaseTransform):

    def __init__(self, roi_size):
        self.roi_size = roi_size

    def transform(self, results):
        bboxes = results['gt_bboxes'].tensor
        masks = results['gt_masks'].to_tensor(torch.float32, 'cpu')

        if masks.size(0) == 0:
            roi_masks = torch.zeros(0,
                                    self.roi_size,
                                    self.roi_size,
                                    dtype=torch.float32)
        else:
            roi_masks = roi_align(
                masks.unsqueeze(1),
                [bbox.reshape(1, 4) for bbox in bboxes],
                self.roi_size,
                sampling_ratio=0,
                aligned=True,
            ).squeeze(1)
        results['gt_masks'] = roi_masks
        return results


@TRANSFORMS.register_module()
class DropGtMasks(BaseTransform):

    def transform(self, results):
        del results['gt_masks']
        return results


@TRANSFORMS.register_module()
class RecomputBBoxes(BaseTransform):

    def __init__(self, roi_size):
        self.roi_size = roi_size

    def transform(self, results):
        results['gt_bboxes'] = results['gt_masks'].get_bboxes(
            type(results['gt_bboxes']))
        return results


@TRANSFORMS.register_module()
class RandomRotateScaleCrop(BaseTransform):
    # TODO: support polygon masks

    def __init__(
        self,
        img_scale,
        scale_range,
        angle_range,
        border_value,
        rotate_prob,
        scale_prob,
        hflip_prob=0.5,
        rot90_prob=1.0,
        shift_mode='auto',
        mask_dtype='u1',
    ):
        super().__init__()
        self.img_scale = img_scale
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.border_value = border_value
        self.rotate_prob = rotate_prob
        self.scale_prob = scale_prob
        self.hflip_prob = hflip_prob
        self.rot90_prob = rot90_prob
        self.shift_mode = shift_mode
        self.mask_dtype = mask_dtype

    def _get_affine_matrix(self, img):

        h, w = img.shape[:2]

        # shift center
        M0 = np.eye(3)
        M0[0, 2] = -w * 0.5
        M0[1, 2] = -h * 0.5

        # hflip
        M1 = np.eye(3)
        if np.random.uniform(0, 1) < self.hflip_prob:
            M1[0, 0] *= -1

        # rot90
        M2 = np.eye(3)
        if np.random.uniform(0, 1) < self.rot90_prob:
            k = np.random.randint(0, 4)
            M2[:2] = cv2.getRotationMatrix2D(center=(0, 0),
                                             angle=k * 90,
                                             scale=1.)

        # scale
        # base scale is img_scale
        scale = min(self.img_scale[0] / w, self.img_scale[1] / h)
        if np.random.uniform(0, 1) < self.scale_prob:
            scale *= np.random.uniform(*self.scale_range)

        # rotate
        if np.random.uniform(0, 1) < self.rotate_prob:
            angle = np.random.uniform(*self.angle_range)
        else:
            angle = 0
        M3 = np.eye(3)
        M3[:2] = cv2.getRotationMatrix2D(center=(0, 0),
                                         scale=scale,
                                         angle=angle)

        # shift to align lefttop to (0, 0)
        M4 = np.eye(3)
        if self.shift_mode == 'auto':
            # get size after rotate&scale
            corners_orig = np.array([
                [0, 0, 1],
                [w, 0, 1],
                [0, h, 1],
                [w, h, 1],
            ])
            _M = M3 @ M2 @ M1 @ M0
            corners_curr = corners_orig @ _M.T
            xmin, ymin = corners_curr[:, :2].min(0)
            xmax, ymax = corners_curr[:, :2].max(0)
        else:
            xmin = -w * scale * 0.5
            ymin = -h * scale * 0.5
            xmax = -xmin
            ymax = -ymin
        M4[0, 2] -= xmin
        M4[1, 2] -= ymin

        # shift(crop)
        margin_w = self.img_scale[0] - (xmax - xmin)
        margin_h = self.img_scale[1] - (ymax - ymin)
        M4[0, 2] += margin_w * np.random.uniform(0, 1)
        M4[1, 2] += margin_h * np.random.uniform(0, 1)

        M = M4 @ M3 @ M2 @ M1 @ M0
        return M[:2]

    def transform(self, results):
        M = self._get_affine_matrix(results['img'])

        img = results['img']
        gt_masks = results['gt_masks']
        assert isinstance(gt_masks, BitmapMasks)
        gt_masks = gt_masks.to_ndarray().astype(self.mask_dtype)

        rot_img = cv2.warpAffine(
            img,
            M,
            self.img_scale,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

        rot_gt_masks = cv2.warpAffine(
            gt_masks.transpose(1, 2, 0),
            M,
            self.img_scale,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if rot_gt_masks.ndim == 2:
            # case when only one mask, (h, w)
            rot_gt_masks = rot_gt_masks[:, :, None]  # (h, w, 1)
        rot_gt_masks = rot_gt_masks.transpose(2, 0, 1)

        # drop oob instances
        valid_inds = rot_gt_masks.any((1, 2))
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]
        rot_gt_masks = rot_gt_masks[valid_inds]

        # new gt_masks and gt_bboxes
        rot_gt_masks = BitmapMasks(rot_gt_masks, rot_img.shape[0],
                                   rot_img.shape[1])
        if self.mask_dtype == 'u1':
            rot_gt_bboxes = rot_gt_masks.get_bboxes(type(results['gt_bboxes']))
        else:
            binary_rot_gt_masks = BitmapMasks(rot_gt_masks.masks > 0.5,
                                              rot_img.shape[0],
                                              rot_img.shape[1])
            rot_gt_bboxes = binary_rot_gt_masks.get_bboxes(
                type(results['gt_bboxes']))

        results['img'] = rot_img
        results['gt_bboxes'] = rot_gt_bboxes
        results['gt_masks'] = rot_gt_masks
        results['img_shape'] = rot_img.shape[:2]
        results['flip'] = None
        results['flip_direction'] = None
        return results
