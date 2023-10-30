# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_final import BaseDetectorFinal

from itertools import zip_longest
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np


from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.0) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


@DETECTORS.register_module()
class TwoStageDetectorDFPN(BaseDetectorFinal):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 neck=None,
                 neck_hsi=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetectorDFPN, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.backbone_hsi = build_backbone(backbone_hsi)

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_hsi = build_neck(neck_hsi)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        k_size = 1
        padding = k_size//2
        self.conv_down = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=k_size, stride=1, padding=padding),
                torch.nn.GroupNorm(32,256),
                torch.nn.LeakyReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(512, 256, kernel_size=k_size, stride=1, padding=padding),
                torch.nn.GroupNorm(32,256),
                torch.nn.LeakyReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(1024, 256, kernel_size=k_size, stride=1, padding=padding),
                torch.nn.GroupNorm(32,256),
                torch.nn.LeakyReLU(inplace=True),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(2048, 256, kernel_size=k_size, stride=1, padding=padding),
                torch.nn.GroupNorm(32,256),
                torch.nn.LeakyReLU(inplace=True),
            )
        ])
        k_size = 5
        padding = k_size//2
        self.conv_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=k_size, stride=1, padding=padding),
            torch.nn.GroupNorm(32,256),
            torch.nn.ReLU(inplace=True),
            #  torch.nn.Conv2d(256, 256, kernel_size=k_size, stride=1, padding=padding),
            #  torch.nn.GroupNorm(32,256),
            #  torch.nn.ReLU(inplace=True),
            #  torch.nn.Conv2d(256, 256, kernel_size=k_size, stride=1, padding=padding),
            #  torch.nn.GroupNorm(32,256),
            #  torch.nn.ReLU(inplace=True),
            )
        k_size = 5
        padding = k_size//2
        self.conv_att = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=k_size, stride=1, padding=padding),
            torch.nn.GroupNorm(32,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid(),
            )
        self.up3 = torch.nn.Upsample(scale_factor=4, mode='nearest')

        self.conv_neck_hsi = torch.nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                torch.nn.GroupNorm(32,256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.GroupNorm(32,256),
                torch.nn.ReLU(inplace=True),
                )
        self.conv_hsi = torch.nn.Sequential(
                torch.nn.Conv2d(30, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.GroupNorm(32,64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
                torch.nn.GroupNorm(32,64),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.GroupNorm(32,128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
                torch.nn.GroupNorm(32,128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                torch.nn.GroupNorm(32,256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                torch.nn.GroupNorm(32,256),
                torch.nn.ReLU(inplace=True),
                )
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img, hsi):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        hsi_0 = self.conv_hsi(hsi)
        x_hsi = self.backbone_hsi(hsi)
        if self.with_neck:
            x_hsi = self.neck_hsi(x_hsi)
        x_hsi_0 = self.conv_neck_hsi( hsi_0 + self.up3(x_hsi[0]))
        x_hsi = (x_hsi_0,) + x_hsi
        x = [self.conv_down[i](x[i]) for i in range(len(x))]
        x_hsi = [F.interpolate(x_hsi[0], size= (x[i].shape[-2],x[i].shape[-1]), mode='bicubic') for i in range(len(x))]

        x_f = [self.conv_fusion(torch.cat([x_hsi[i], x[i]], dim=1 )) for i in range(len(x))]
        x_att = [ self.conv_att(x_f[i]) for i in range(len(x))]
        x = [x_f[i] * x_att[i] for i in range(len(x_f))]

        if self.with_neck:
            x = self.neck(x)
        return x
        #  x = [ x[i] * x_att[i] for i in range(len(x)-1)] + x[-1]

    def forward_dummy(self, img, hsi):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img, hsi)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      hsi,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img, hsi)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, hsi, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img, hsi)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, hsis, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs, hsis)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
