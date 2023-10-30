# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from .single_stage_DFPN import SingleStageDetectorDFPN
from .single_stage_detfusion import SingleStageDetectorDetFusion
from .single_stage_early import SingleStageDetectorEarly
from .single_stage_late import SingleStageDetectorLate
from .single_stage_UACM import SingleStageDetectorUACM

@DETECTORS.register_module()
class VFNet(SingleStageDetector):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)


@DETECTORS.register_module()
class VFNetDFPN(SingleStageDetectorDFPN):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

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
        super(VFNetDFPN, self).__init__(backbone, backbone_hsi,
                                         neck, neck_hsi,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)

@DETECTORS.register_module()
class VFNetDetFusion(SingleStageDetectorDetFusion):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

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
        super(VFNetDetFusion, self).__init__(backbone, backbone_hsi,
                                         neck, neck_hsi,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)

@DETECTORS.register_module()
class VFNetEarly(SingleStageDetectorEarly):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VFNetEarly, self).__init__(backbone,
                                         neck,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)


@DETECTORS.register_module()
class VFNetLate(SingleStageDetectorLate):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

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
        super(VFNetLate, self).__init__(backbone, backbone_hsi,
                                         neck, neck_hsi,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)

@DETECTORS.register_module()
class VFNetUACM(SingleStageDetectorUACM):
    """Implementation of `VarifocalNet
    (VFNet).<https://arxiv.org/abs/2008.13367>`_"""

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
        super(VFNetUACM, self).__init__(backbone, backbone_hsi,
                                         neck, neck_hsi,
                                         bbox_head, train_cfg,
                                         test_cfg, pretrained, init_cfg)
