from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
from src import vgg_arch_train
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras import backend as K
from keras.utils import np_utils
from scipy import misc
from keras import optimizers
import cv2
import os

image_input = Input(shape=(3, 224, 224))

vgg_model = vgg_arch_train.VGGFace(image_input, False, None)  # paramaters:input_tensor,include_top,weights
#vgg_model.load_weights(TH_WEIGHTS_PATH_NO_TOP)

# last_layer = vgg_model.get_layer('pool5').output
#
# x = Flatten(name='flatten')(last_layer)
# x = Dense(hidden_dim, activation='relu', name='fc6')(x)
# x = Dropout(0.5)(x)
# x = Dense(hidden_dim, activation='relu', name='fc7')(x)
# x = Dropout(0.5)(x)
# out = Dense(num_of_persons, activation='softmax', name='fc8')(x)
#
# custom_vgg_model = Model(image_input, out)

vgg_model.load_weights('/home/uia70982/face_reco/rcmalli_vggface_th_weights_th_ordering_notop2.h5')
ops = []
for layer in vgg_model.layers:
   print(layer.__class__.__name__)
   if layer.__class__.__name__ in ['Conv1D', 'Conv2D', 'Conv3D', 'AtrousConvolution2D']:
      print("Getting inside")
      original_w = K.get_value(layer.W)
      converted_w = convert_kernel(original_w)
      ops.append(tf.assign(layer.W, converted_w).op)
      print("converted")

vgg_model.save_weights('rcmalli_vggface_tf_weights_tf_ordering_notop.h5')