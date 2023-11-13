import numpy as np
import os
import tensorflow as tf
import cv2
from glob import glob
from pprint import pprint
from time import time
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

def flatten_model(nested_model):
    def get_layers(layers):
        layers_flat = []
        for layer in layers:
            try:
                layers_flat.extend(get_layers(layer.layers))
            except AttributeError:
                layers_flat.append(layer)
        return layers_flat

    flat_model = get_layers(nested_model.layers)
    return flat_model

def bn_fusion(value_lists):
    kernel_bias = 0.
    if len(value_lists) == 5:
        kernel_weights= value_lists[0]
        bn_gamma, bn_beta = value_lists[1], value_lists[2]
        bn_moving_mean, bn_moving_variance = value_lists[3], value_lists[4]
    else:
        kernel_weights, kernel_bias = value_lists[0], value_lists[1]
        bn_gamma, bn_beta = value_lists[2], value_lists[3]
        bn_moving_mean, bn_moving_variance = value_lists[4], value_lists[5]
    M = kernel_weights.shape[-1]
    if M == 1:
        fused_weights = (kernel_weights * bn_gamma[None, None, :, None]) / np.sqrt(bn_moving_variance[None, None, :, None] + 0.001)
        
    else:
        fused_weights = (kernel_weights * bn_gamma) / np.sqrt(bn_moving_variance + 0.001)
    fused_bias = bn_beta + (bn_gamma / np.sqrt(bn_moving_variance + 0.001)) * (
            kernel_bias - bn_moving_mean)
    return fused_weights, fused_bias

class TestModel(tf.keras.Model):

    def __init__(self):
        super(TestModel, self).__init__()
        self.stem = tf.keras.layers.Conv2D(32,
                                           kernel_size=3,
                                           strides=2,
                                           use_bias=True,
                                           padding='same',
                                           name='stem')
        self.act1 = tf.keras.layers.Activation(activation='relu', name='act1')
        self.dw1 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                use_bias=True,
                padding='same',
                name='dw_blk0')
        self.conv1 = tf.keras.layers.Conv2D(16,
                                           kernel_size=1,
                                           strides=1,
                                           use_bias=True,
                                           padding='same',
                                           name='conv0_blk0')
        
        self.conv2 = tf.keras.layers.Conv2D(96,
                                           kernel_size=1,
                                           strides=1,
                                           use_bias=True,
                                           padding='same',
                                           name='conv1_blk0')
        

        self.dw2 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=2,
                use_bias=True,
                padding='same',
                name='dw0_blk0') # 4
        

        self.dw3 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                use_bias=True,
                padding='same',
                name='dw0_blk1') # 5


        self.dw4 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=1,
                strides=1,
                use_bias=True,
                padding='same',
                name='dw1_blk1')  # 6
        
        self.dw5 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=1,
                strides=1,
                use_bias=True,
                padding='same',
                name='dw2_blk1') # 7
        
        self.dw6 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                use_bias=True,
                padding='same',
                name='dw3_blk1') # 8
    def __call__(self, x):
        x = self.stem(x)
        x = self.act1(x)
        x = self.dw1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act1(x)


        x = self.dw2(x)
        x = self.dw3(x)
        x = self.act1(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.act1(x)
        x = self.dw6(x)
        x = self.act1(x)
        return x
    
cp_dir = "/aidata/anders/data_collection/okay/total/archives/WF/scale_down"
path = "/aidata/anders/data_collection/okay/demo_test/imgs/frame-000867.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (320, 320))
img = img / 255.
img = img[..., ::-1]

img = img.reshape([-1, 320, 320, 3]).astype(np.float32)
image_inputs = tf.keras.Input(shape=(320, 320, 3), name='image_inputs')
testmodel = TestModel()
fmaps = testmodel(image_inputs)
model = tf.keras.Model(image_inputs, fmaps, name='test')
restored_model = tf.keras.models.load_model(cp_dir)
preds  = restored_model(img, training = False)



model_weights  = model.get_weights()
backbone = restored_model.backbone.get_layer('mobile_next_net_model')
preds = backbone(img, training = False)
for multi_lv_feats in preds:
    multi_lv_feats = multi_lv_feats.numpy()
    print('-' * 100)
    print(multi_lv_feats.shape)
    multi_lv_feats = multi_lv_feats.reshape([-1])
    print(multi_lv_feats[:10])
xxx

backbone_layers = flatten_model(backbone)
backbone_layer0 = backbone_layers[0]
weight0 = bn_fusion(backbone_layer0.get_weights())

model_weights[0] = weight0[0] 
model_weights[1] = weight0[1]
# ------------------------------------------------------------------------
backbone_layer1 = backbone_layers[1]
conv_layers = backbone_layer1.__dict__['conv_layers']

layer1_blk0 = conv_layers[0]
weight1_blk0 = bn_fusion(layer1_blk0.get_weights())
model_weights[2] = weight1_blk0[0]
model_weights[3] = weight1_blk0[1]

layer1_blk1 = conv_layers[1]
weight1_blk1 = bn_fusion(layer1_blk1.get_weights())

model_weights[4] = weight1_blk1[0]
model_weights[5] = weight1_blk1[1]

layer1_blk2 = conv_layers[2]
weight1_blk2 = bn_fusion(layer1_blk2.get_weights())
model_weights[6] = weight1_blk2[0]
model_weights[7] = weight1_blk2[1]
layer1_blk3 = conv_layers[3]
weight1_blk3 = bn_fusion(layer1_blk3.get_weights())
model_weights[8] = weight1_blk3[0]
model_weights[9] = weight1_blk3[1]
# ------------------------------------------------------------------------
backbone_layer2 = backbone_layers[2]
conv_layers = backbone_layer2.__dict__['conv_layers']

layer1_blk1 = conv_layers[0]
weight1_blk1 = bn_fusion(layer1_blk1.get_weights())
model_weights[10] = weight1_blk1[0]
model_weights[11] = weight1_blk1[1]

layer2_blk1 = conv_layers[1]
weight2_blk1 = bn_fusion(layer2_blk1.get_weights())
model_weights[12] = weight2_blk1[0]
model_weights[13] = weight2_blk1[1]

layer3_blk1 = conv_layers[2]
weight3_blk1 = bn_fusion(layer3_blk1.get_weights())
model_weights[14] = weight3_blk1[0]
model_weights[15] = weight3_blk1[1]


layer4_blk1 = conv_layers[3]
weight4_blk1 = bn_fusion(layer4_blk1.get_weights())
model_weights[16] = weight4_blk1[0]
model_weights[17] = weight4_blk1[1]
# ---------------------------------------------------------------

model.set_weights(model_weights)
preds = model(img, training=False)
preds = preds.numpy().astype(np.float32)
print("output shape is {}".format(preds.shape))
preds = np.reshape(preds, [-1])
print(preds[:30])
xxxx
# path = "/home2/anders/proj_c/SNN/test/stem.bin"
# CLpreds = np.fromfile(path, dtype=np.float32)

# N = preds.shape[0]
# # img = img.reshape([-1])
# # N = img.shape[0]
# for i in range(N):
#     mask = np.isclose(preds[i], CLpreds[i], atol = 1e-5)
#     if  mask == False:
#         print('-'*100)
#         print(i)
#         print(mask)
#         print(preds[i], CLpreds[i])
#         xxx
