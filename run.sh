#!/bin/bash

CFG=configs/selfsup/deepcluster/r50_fb_isc.py

tools/dist_train.sh $CFG 4

#python tools/train.py $CFG --gpus 0