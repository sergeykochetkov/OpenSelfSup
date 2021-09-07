#!/bin/bash

CFG=/home/skochetkov/Documents/OpenSelfSup/configs/selfsup/moco/effnet_b1_simclr_neck_screenshot.py

tools/dist_train.sh $CFG 2

#python tools/train.py $CFG --gpus 0