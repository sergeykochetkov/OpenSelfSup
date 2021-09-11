_base_ = '../../base.py'
# model settings
img_size = 128
model = dict(
    type='Classification',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=2048,
        num_classes=100000))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = '/home/skochetkov/Documents/isc/data/image_list.txt'
data_train_root = '/home/skochetkov/Documents/isc/data/fb-isc-data-training-images'
data_test_list = '/home/skochetkov/Documents/isc/data/image_list_val.txt'
data_test_root = data_train_root
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_pipeline = [
    dict(type='Resize', size=(img_size, img_size)),
    dict(type='AugLy', img_size=img_size,
         src_jpg_path='/home/skochetkov/Documents/isc/data/fb-isc-data-training-images'),
    dict(type='Resize', size=(img_size, img_size)), dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

# prefetch
prefetch = False
test_pipeline = train_pipeline

data = dict(
    imgs_per_gpu=28,  # total 256
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=0.00001, weight_decay=0.0001)
# learning policy
total_epochs = 10
lr_config = dict(policy='step', step=[total_epochs // 4, 2 * total_epochs // 4, 3 * total_epochs // 4])
checkpoint_config = dict(interval=1)
# runtime settings


work_dir = 'work_dirs/classification/imagenet/r50_fb_isc'

# apex
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)  # grad_clip, coalesce, bucket_size_mb, fp16
