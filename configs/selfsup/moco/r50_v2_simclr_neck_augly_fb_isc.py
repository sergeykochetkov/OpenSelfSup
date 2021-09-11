_base_ = '../../base.py'
# model settings
img_size = 224
model = dict(
    type='MOCO',
    pretrained='torchvision://resnet50',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckSimCLR',  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageList',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = '/home/skochetkov/Documents/isc/data/image_list.txt'
data_train_root = '/home/skochetkov/Documents/isc/data/fb-isc-data-training-images'
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=(img_size, img_size)),
    dict(type='AugLy', img_size=img_size,
         src_jpg_path='/home/skochetkov/Documents/isc/data/fb-isc-data-training-images'),
]

# prefetch
prefetch = False

normalization_pipeline = [dict(type='Resize', size=(img_size, img_size)), dict(type='ToTensor'),
                          dict(type='Normalize', **img_norm_cfg)]

data = dict(
    imgs_per_gpu=128,  # 128,  # total 32*8=256
    workers_per_gpu=16,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        normalization_pipeline=normalization_pipeline,
        prefetch=prefetch,
    ))
# optimizer
optimizer = dict(type='SGD', lr=0.1, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 20
#resume_from = '/home/skochetkov/Documents/OpenSelfSup/work_dirs/selfsup/moco/r50_v2_simclr_neck_augly_fb_isc/latest.pth'

# apex
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)  # grad_clip, coalesce, bucket_size_mb, fp16
