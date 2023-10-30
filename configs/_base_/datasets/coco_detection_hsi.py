# dataset settings
import scipy.io as sio
dataset_type = 'CocoDataset_DMC'
data_root = './data/'

mean_std = sio.loadmat('./data/mean_std.mat')
mean_tmp = mean_std['mean'].reshape([-1]) # (1,127)
std_tmp = mean_std['std'].reshape([-1]) # (1,127)
img_norm_cfg = dict(
mean=mean_tmp.tolist(),std=std_tmp.tolist(),to_rgb=False)

del mean_std
del mean_tmp
del std_tmp
del sio

train_pipeline = [
    dict(type='LoadImageFromHSI'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    #dict(type='MinIoURandomCrop', min_ious=(0.5, 0.7,0.9)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromHSI'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
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
        ann_file=data_root + 'train_coco/annotations_HSI.json',
        img_prefix=data_root + 'train_coco/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_coco/annotations_HSI.json',
        img_prefix=data_root + 'val_coco/',
        pipeline=test_pipeline),
    test = dict(
      type=dataset_type,
      ann_file=data_root + 'test_coco/annotations_HSI.json',
      img_prefix=data_root + 'test_coco/',
        pipeline=test_pipeline,
        separate_eval=True)
    )

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
