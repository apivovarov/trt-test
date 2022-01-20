import sys
import tensorflow as tf
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("TF memory_growth:", tf.config.experimental.get_memory_growth(physical_devices[0]))

#MODEL = "centernet_resnet50_v2_512x512_coco17_tpu-8"
MODEL = sys.argv[1]
img_sz=int(sys.argv[2])
dtype=tf.uint8
batch=int(sys.argv[3]) if len(sys.argv) > 3 else 1

m=tf.saved_model.load(MODEL)
ff=m.signatures['serving_default']
x=tf.ones(shape=(batch,img_sz,img_sz,3), dtype=dtype)
y = ff(x)
y = ff(x)
y = ff(x)
N = 50
t1 = time.time()
for i in range(N):
  y = ff(x)

ttt = time.time() - t1
print("-----------------------------------", MODEL, "--------------------")
print("exec time:", ttt)
print(batch*N/ttt, "fps")
print("GPU Mem usage:", tf.config.experimental.get_memory_info('GPU:0'))
print("--------------------------------------------------------------------------")
