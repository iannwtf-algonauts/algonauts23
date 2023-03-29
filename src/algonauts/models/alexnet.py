from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


def create_alexnet(num_classes):
    """
    Create an AlexNet implementation using keras sequential api
    :param num_classes: number of label classes
    :return: keras model object
    """
    input_shape = (227, 227, 3)

    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation="relu", input_shape=input_shape, padding="same",
                     kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), activation="relu", padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Conv2D(384, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu", kernel_initializer="glorot_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="sigmoid"))

    return model
