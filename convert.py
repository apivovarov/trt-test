import tensorflow as tf
import sys
from tensorflow.python.compiler.tensorrt import trt_convert as trt

pr=sys.argv[1]

precision_mode=trt.TrtPrecisionMode.FP32
if pr == "fp16":
    precision_mode=trt.TrtPrecisionMode.FP16

conversion_params = trt.TrtConversionParams(precision_mode=precision_mode)
print("conversion_params:", conversion_params)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="saved_model",
    conversion_params=conversion_params)

converter.convert()
converter.save("saved_model_trt_"+pr)
print("Saved", "saved_model_trt_"+pr)
