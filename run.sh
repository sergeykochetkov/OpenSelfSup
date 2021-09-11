#!/bin/bash

CFG=/home/skochetkov/Documents/OpenSelfSup/configs/classification/imagenet/r50_fb_isc.py

tools/dist_train.sh $CFG 2

#python tools/train.py $CFG --gpus 0