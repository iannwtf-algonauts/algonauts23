import tensorflow as tf
from src.algonauts.data_processors import image_transforms


def load_vgg16():
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    transform_image = image_transforms.transform_vgg16
    return model, transform_image


def load_from_file(model_filename):
    model = tf.keras.models.load_model(model_filename)
    transform_image = image_transforms.transform_generic
    return model, transform_image
