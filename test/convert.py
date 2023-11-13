import tensorflow as tf
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

cp_dir = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down"
restored_model = tf.keras.models.load_model(cp_dir)
inputs = tf.constant(0., shape=(1, 320, 320, 3))
preds = restored_model(inputs, training=False)
backbone = restored_model.backbone.get_layer('mobile_next_net_model')
converter = tf.lite.TFLiteConverter.from_keras_model(backbone)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.float32]
# converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
model_path = os.path.join(cp_dir, "./tflite/backbone.tflite")
with open(model_path, 'wb') as f:
    f.write(tflite_model)