import tensorflow as tf
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
cp_dir = "/aidata/anders/data_collection/okay/total/archives/whole/mobilenext_kps"
restored_model = tf.keras.models.load_model(cp_dir)
inputs = tf.constant(0., shape=(1, 320, 320, 3))
preds = restored_model(inputs, training=False)
converter = tf.lite.TFLiteConverter.from_keras_model(restored_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()
model_path = os.path.join(cp_dir, "./tflite/mtfd_FP32.tflite")
with open(model_path, 'wb') as f:
    f.write(tflite_model)