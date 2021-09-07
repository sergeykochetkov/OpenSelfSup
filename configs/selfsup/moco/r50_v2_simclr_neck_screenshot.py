_base_ = '../../base.py'
# model settings
feat_dim = 128
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=feat_dim,
    momentum=0.999,
    backbone=dict(
        type='EfficientNetAbbyy',
        blocks_args_list=[
            'r1_k3_s11_e1_i32_o16_se0.25',
            'r2_k3_s22_e6_i16_o24_se0.25',
            'r2_k5_s22_e6_i24_o40_se0.25',
            'r3_k3_s22_e6_i40_o80_se0.25',
            'r3_k5_s11_e6_i80_o112_se0.25',
            'r4_k5_s22_e6_i112_o192_se0.25',
            'r1_k3_s11_e6_i192_o320_se0.25',
        ],
        blocks_args='efficientnet-b0',
        out_indices=(1, 4, 7, 15)),
    neck=dict(
        type='NonLinearNeckSimCLR',  # SimCLR non-linear neck
        in_channels=320,
        hid_channels=320,
        out_channels=feat_dim,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageList',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/image_list.txt'
data_train_root = 'data/cleaned'
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),

    dict(type='RandomHorizontalFlip'),
    dict(type='RandomVerticalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=128,  # total 32*8=256
    workers_per_gpu=16,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 40
find_unused_parameters = False
work_dir='work_dirs/selfsup/moco/r50_v2_simclr_neck_screenshot'

# apex
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)  # grad_clip, coalesce, bucket_size_mb, fp16
resume_from='/home/skochetkov/Documents/OpenSelfSup/work_dirs/selfsup/moco/r50_v2_simclr_neck_screenshot/latest.pth'