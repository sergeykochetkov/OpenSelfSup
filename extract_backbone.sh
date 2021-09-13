#!/bin/bash
WORK_DIR=work_dirs/selfsup/deepcluster/r50_fb_isc/
python tools/extract_backbone_weights.py $WORK_DIR/latest.pth $WORK_DIR/backbone.pth

scp  $WORK_DIR/backbone.pth skochetkov@robotfactory-cuda2.abbyy.unix:/home/skochetkov/