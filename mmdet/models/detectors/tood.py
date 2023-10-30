# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_DFPN import SingleStageDetectorDFPN
from .single_stage_detfusion import SingleStageDetectorDetFusion
from .single_stage_early import SingleStageDetectorEarly
from .single_stage_late import SingleStageDetectorLate
from .single_stage_UACM import SingleStageDetectorUACM

@DETECTORS.register_module()
class TOOD(SingleStageDetector):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOOD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

@DETECTORS.register_module()
class TOODDFPN(SingleStageDetectorDFPN):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 neck,
                 neck_hsi,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOODDFPN, self).__init__(backbone, backbone_hsi, 
                                       neck, neck_hsi, 
                                       bbox_head, train_cfg,
                                       test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

@DETECTORS.register_module()
class TOODDetFusion(SingleStageDetectorDetFusion):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 neck,
                 neck_hsi,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOODDetFusion, self).__init__(backbone, backbone_hsi, neck, neck_hsi, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

@DETECTORS.register_module()
class TOODEarly(SingleStageDetectorEarly):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOODEarly, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

@DETECTORS.register_module()
class TOODLate(SingleStageDetectorLate):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 neck,
                 neck_hsi,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOODLate, self).__init__(backbone, backbone_hsi, neck, neck_hsi, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch

@DETECTORS.register_module()
class TOODUACM(SingleStageDetectorUACM):
    r"""Implementation of `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_."""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 neck,
                 neck_hsi,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TOODUACM, self).__init__(backbone, backbone_hsi, neck, neck_hsi, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
