_base_ = '../../base.py'
# model settings
img_size=128
num_classes = 10000
model = dict(
    type='DeepCluster',
    pretrained='torchvision://resnet50',
    with_sobel=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=2,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(type='AvgPoolNeck'),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=2048,
        num_classes=num_classes))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')

data_train_list = '/DATA/SKochetkov/isc/data/image_list.txt'
data_train_root = '/DATA/SKochetkov/isc/data/fb-isc-data-training-images'

dataset_type = 'DeepClusterDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=img_size),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=1.0,
        hue=0.5),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
extract_pipeline = [
    dict(type='Resize', size=img_size),
    dict(type='CenterCrop', size=img_size),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=16,
            dataset=dict(
                type=dataset_type,
                data_source=dict(
                    list_file=data_train_list,
                    root=data_train_root,
                    **data_source_cfg),
                pipeline=extract_pipeline)),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]
# optimizer
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00001,
    nesterov=False,
    paramwise_options={'\Ahead.': dict(momentum=0.)})
# learning policy
lr_config = dict(policy='step', step=[400])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
