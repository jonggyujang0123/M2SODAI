# dataset settings
dataset_type = 'CocoDataset_DMC'
data_root = './data/'
img_norm_cfg = dict(
    mean=[123.6, 116.2, 103.5],std=[58.39, 56.12, 57.3],to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1600, 1600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.75),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('ship', 'floatingmatter')


data = dict(
    #imgs_per_gpu = 2,
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_coco/annotations.json',
        img_prefix=data_root + 'train_coco/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_coco/annotations.json',
        img_prefix=data_root + 'val_coco/',
        pipeline=test_pipeline),
    test=dict(
      type=dataset_type,
      ann_file=data_root + 'test_coco/annotations.json',
      img_prefix=data_root + 'test_coco/',
      pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
