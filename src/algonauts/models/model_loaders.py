import tensorflow as tf
from src.algonauts.data_processors import image_transforms
from src.algonauts.models.alexnet import create_alexnet_softmax


def load_vgg16():
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    transform_image = image_transforms.transform_vgg16
    return model, transform_image


def load_alexnet(num_classes):
    model = create_alexnet_softmax(num_classes)
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    transform_image = image_transforms.transform_alexnet
    return model, transform_image


def load_from_file(model_filename, transform_image, custom_objects=None):
    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)
    return model, transform_image
