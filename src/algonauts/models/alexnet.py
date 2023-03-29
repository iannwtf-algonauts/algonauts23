from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


def create_alexnet_sigmoid(num_classes):
    """
    Create an AlexNet implementation using keras sequential api, with a sigmoid layer as output for non-mutually
    exclusive multilabel classification problems (eg. COCO)
    classification
    :param num_classes: number of label classes
    :return: keras model object
    """
    model = create_alexnet_headless()
    model.add(Dense(num_classes, activation="sigmoid"))

    return model


def create_alexnet_softmax(num_classes):
    """
    Create an AlexNet implementation using keras sequential api, with a softmax layer as output for mutually exclusive,
    multiclass classification problems (eg. ImageNet)
    :param num_classes: number of label classes
    :return: keras model object
    """
    model = create_alexnet_headless()
    model.add(Dense(num_classes, activation="softmax"))

    return model


def create_alexnet_headless():
    """
    Create an AlexNet implementation using keras sequential api without the output layer
    :return: keras model object
    """
    input_shape = (227, 227, 3)

    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation="relu", input_shape=input_shape, padding="same",
                     kernel_initializer="he_normal", name='conv2d_1'))
    model.add(BatchNormalization(name='conv2d_1_bn'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='conv2d_1_pool'))
    model.add(Conv2D(256, (5, 5), activation="relu", padding="same", kernel_initializer="he_normal", name='conv2d_2'))
    model.add(BatchNormalization(name='conv2d_2_bn'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='conv2d_2_pool'))
    model.add(Conv2D(384, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal", name='conv2d_3'))
    model.add(BatchNormalization(name='conv2d_3_bn'))
    model.add(Conv2D(384, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal", name='conv2d_4'))
    model.add(BatchNormalization(name='conv2d_4_bn'))
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal", name='conv2d_5'))
    model.add(BatchNormalization(name='conv2d_5_bn'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='conv2d_5_pool'))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dropout(0.5))

    return model
