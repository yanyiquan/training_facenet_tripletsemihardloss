import tensorflow as tf
import keras
import numpy as np

def face_net_network(name_model):
    if name_model == 'VGG16':
        model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    if name_model == 'DenseNet121':
        model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 3), include_top=False, weights = 'imagenet')
    if name_model == 'mobilenetv2':
        model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    Dense = tf.keras.layers.Dense(128)(model.layers[-4].output)
    norm2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(Dense)
    model = tf.keras.models.Model(inputs=[model.input], outputs=[norm2])
    return model
