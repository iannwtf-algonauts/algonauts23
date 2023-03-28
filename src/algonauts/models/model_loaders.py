import tensorflow as tf
import tensorflow_addons as tfa
from src.algonauts.data_processors import image_transforms


def load_vgg16():
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    transform_image = image_transforms.transform_vgg16
    return model, transform_image


def load_from_file(model_filename, transform_image):
    model = tf.keras.models.load_model(model_filename, custin_objects={'F1Score': tfa.metrics.F1Score})
    return model, transform_image
