import numpy as np
import tensorflow as tf
import os
import cv2
cp_dir = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down/tflite"
interpreter = tf.lite.Interpreter(
    model_path=os.path.join(cp_dir, "backbone.tflite"))
# interpreter.allocate_tensors()
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
path = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (320, 320))
img = img / 255.
img = img[..., ::-1]
img = img.reshape([-1, 320, 320, 3]).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
for output_detail in output_details:
    pred_maps = interpreter.get_tensor(output_detail['index'])
    pred_maps = pred_maps.reshape([-1])
    print()
    print('-'*100)
    print(pred_maps[:10])
    xxx