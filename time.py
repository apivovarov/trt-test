import sys
import tensorflow as tf
import time
import numpy as np
from tensorflow.python.framework.convert_to_constants \
                    import convert_variables_to_constants_v2

# set_memory_growth(True) for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
for gpu_dev in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_dev, True)

MODEL = sys.argv[1]
img_sz=int(sys.argv[2])
#dtype=tf.uint8
dtype=sys.argv[3]
batch=int(sys.argv[4]) if len(sys.argv) > 4 else 1

m=tf.saved_model.load(MODEL)
f=m.signatures['serving_default']
#ff=f
ff = convert_variables_to_constants_v2(f) #, lower_control_flow=False)
x2=tf.ones(shape=(batch,img_sz,img_sz,3), dtype=dtype)
# Warmup
N = 50
for i in range(N):
  x = tf.constant(x2)
  y = ff(x)

N = 1000
t1 = time.time()
TTT=[]
for i in range(N):
  t10 = time.time()
  x = tf.constant(x2)
  y = ff(x)
  t11 = time.time() - t10
  TTT.append(t11)

ttt = time.time() - t1
print("-----------------------------------", MODEL, "--------------------")
print("exec time (sec):", ttt)
print("AVG (fps)", batch*N/ttt)
p50_time = np.percentile(TTT, 50) * 1000 / batch
print("P50 (ms):", p50_time)
print("P50 (fps):", 1000/p50_time)
#print("GPU Mem usage:", tf.config.experimental.get_memory_info('GPU:0'))
print("--------------------------------------------------------------------------")
#print(TTT)
print()
