#!/bin/bash

MODELS=(
centernet_hg104_512x512_coco17_tpu-8
centernet_resnet50_v2_512x512_coco17_tpu-8
efficientdet_d0_coco17_tpu-32
faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8
faster_rcnn_resnet50_v1_640x640_coco17_tpu-8
ssd_mobilenet_v2_320x320_coco17_tpu-8
ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8
)

for M in ${MODELS[@]}
do
  echo $M
  cd $M
  python3.7 ../convert.py fp32
  python3.7 ../convert.py fp16
  cd ..
done

