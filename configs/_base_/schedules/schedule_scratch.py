optimizer = dict(
        type='SGD', 
        lr= 2e-2, 
        momentum=0.9, 
        weight_decay=0.0001, 
        paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None) #, _delete_=True)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[ 65, 71 ])
runner = dict(type='EpochBasedRunner', max_epochs=73)

