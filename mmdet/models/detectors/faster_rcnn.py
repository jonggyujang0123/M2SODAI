# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from .two_stage_DFPN import TwoStageDetectorDFPN
from .two_stage_early import TwoStageDetectorEarly
from .two_stage_late import TwoStageDetectorLate
from .two_stage_detfusion import TwoStageDetectorDetFusion
from. two_stage_UACM import TwoStageDetectorUACM
@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

@DETECTORS.register_module()
class FasterRCNNDFPN(TwoStageDetectorDFPN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 neck_hsi=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNDFPN, self).__init__(
            backbone=backbone,
            backbone_hsi=backbone_hsi,
            neck=neck,
            neck_hsi=neck_hsi,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class FasterRCNNEarly(TwoStageDetectorEarly):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNEarly, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class FasterRCNNLate(TwoStageDetectorLate):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 neck_hsi=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNLate, self).__init__(
            backbone=backbone,
            backbone_hsi=backbone_hsi,
            neck=neck,
            neck_hsi=neck_hsi,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)


@DETECTORS.register_module()
class FasterRCNNDetFusion(TwoStageDetectorDetFusion):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 neck_hsi=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNDetFusion, self).__init__(
            backbone=backbone,
            backbone_hsi=backbone_hsi,
            neck=neck,
            neck_hsi=neck_hsi,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)




@DETECTORS.register_module()
class FasterRCNNUACM(TwoStageDetectorUACM):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 backbone_hsi,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 neck_hsi=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNUACM, self).__init__(
            backbone=backbone,
            backbone_hsi=backbone_hsi,
            neck=neck,
            neck_hsi=neck_hsi,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
