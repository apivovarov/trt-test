#!/bin/bash

MODELS=(
"centernet_hg104_512x512_coco17_tpu-8;512"
"centernet_resnet50_v2_512x512_coco17_tpu-8;512"
"efficientdet_d0_coco17_tpu-32;512"
"faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8;640"
"faster_rcnn_resnet50_v1_640x640_coco17_tpu-8;640"
"ssd_mobilenet_v2_320x320_coco17_tpu-8;320"
"ssd_resnet50_v1_fpn_640x640_coco17_tpu-8;640"
"mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8;1024"
)

for M in ${MODELS[@]}
do
  echo $M
  IFS=";" read -r -a arr <<< "${M}"
  echo ${arr[0]}
  echo ${arr[1]}

  python3.7 time.py "${arr[0]}/saved_model" ${arr[1]}
  python3.7 time.py "${arr[0]}/saved_model_trt_fp32" ${arr[1]}
  python3.7 time.py "${arr[0]}/saved_model_trt_fp16" ${arr[1]}
done
