import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization


def transform_vgg16(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image


def transform_generic(image):
    image = tf.image.resize(image, (227, 227))
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    mean = [0.485, 0.456, 0.406]
    variance = [0.229, 0.224, 0.225]
    normalization_layer = Normalization(mean=mean, variance=variance)
    image = normalization_layer(image)
    return image
