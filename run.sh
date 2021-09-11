#!/bin/bash

CFG=configs/selfsup/simclr/r50_bs256_ep200_fb_isc.py

tools/dist_train.sh $CFG 4

#python tools/train.py $CFG --gpus 0