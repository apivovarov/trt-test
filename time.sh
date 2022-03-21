#!/bin/bash

MODELS=(
#"ssd_mobilenet_v2_320x320_coco17_tpu-8_float_batchN_nms/save_model;320;float32"
#"ssd_mobilenet_v2_2-ml_g4dn;512;uint8"
#"ssd_mobilenet_v2_2;320;uint8"
"ssd_mobilenet_v2_2_trt_fp32;320;uint8"
#"centernet_hourglass_512x512_1-ml_g4dn;512;uint8"
#"centernet_resnet50v1_fpn_512x512_1-ml_g4dn;512;uint8"
#"saved_model_trt_fp32;512;uint8"
#"efficientdet_d0_1-ml_g4dn;512;uint8"
#"faster_rcnn_resnet50_v1_640x640_1-ml_g4dn;640;uint8"
#"faster_rcnn_resnet50_v1_640x640_1;640;uint8"
#"faster_rcnn_inception_resnet_v2_640x640_1-ml_g4dn;640;uint8"
#"mask_rcnn_inception_resnet_v2_1024x1024_1-ml_g4dn;512;uint8"
#"mask_rcnn_inception_resnet_v2_1024x1024_1;512;uint8"
#"resnet_50_classification_1-ml_g4dn;224;float32"
#"aaa/compiled_models/1;640;uint8"
)

batch=1
for M in ${MODELS[@]}
do
  echo $M
  LLL=$(echo $M | tr ";" "\n")
  readarray -t arr <<< "$LLL"
  #IFS=";" read -r -a arr <<< "${M}"
  echo "model: ${arr[0]}"
  echo "input size: ${arr[1]}"
  echo "input dtype: ${arr[2]}"
  echo "input batch: ${batch}"

  python3 time.py ${arr[0]} ${arr[1]} ${arr[2]} ${batch}
  #python3.7 time.py "${arr[0]}/saved_model" ${arr[1]}
  #python3.7 time.py "${arr[0]}/saved_model_trt_fp32" ${arr[1]}
  #python3.7 time.py "${arr[0]}/saved_model_trt_fp16" ${arr[1]}
done
