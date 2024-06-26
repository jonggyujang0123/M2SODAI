# model settings
_base_ = [
    '../_base_/datasets/coco_detection_rgb_hsi.py', 
    '../_base_/schedules/schedule_scratch.py', 
    '../_base_/default_runtime.py'
]

# model settings
norm_cfg = dict(type='GN', num_groups =32, requires_grad=True)
model = dict(
    type='FasterRCNNDFPN',
    backbone=dict(
        type='ResNet',
        in_channels=3,
        depth=50,
        num_stages=4,
        zero_init_residual = False,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg= norm_cfg, # dict(type='BN', requires_grad=True), 
        norm_eval=True, 
        style='pytorch',
        init_cfg = None
        ),
    backbone_hsi=dict(
        type='ResNet',
        in_channels=30,
        depth=50,
        num_stages=4,
        zero_init_residual = False,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg= norm_cfg, #dict(type='BN', requires_grad=True), 
        norm_eval=True, 
        style='pytorch',
        init_cfg = None
        ),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048, 4096],
        # in_channels=[256, 256, 256, 256],
        # norm_cfg = norm_cfg,
        norm_cfg = norm_cfg,
        out_channels=256,
        num_outs=5),
    neck_hsi=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        norm_cfg = norm_cfg,
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)
        ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.01),
            #nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
    ))
