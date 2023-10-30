checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=60,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train',1)]
fp16 = dict(loss_scale=512.)
