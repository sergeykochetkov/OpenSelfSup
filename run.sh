#!/bin/bash

CFG=/home/skochetkov/Documents/OpenSelfSup/configs/selfsup/moco/r50_v2_simclr_neck_augly_fb_isc.py

tools/dist_train.sh $CFG 2

#python tools/train.py $CFG --gpus 0