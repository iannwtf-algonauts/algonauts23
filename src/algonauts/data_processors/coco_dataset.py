import tensorflow as tf


def create_datasets_from_coco(dataset, num_classes, transform_image, batch_size):
    # Function to encode labels in multi-hot encoding
    def encode_coco_categories(coco_categories, num_classes):
        return tf.reduce_max(tf.one_hot(coco_categories, num_classes), axis=0)

    # Function to set labels to zeros
    def set_labels_as_zeros():
        return tf.zeros(num_classes, tf.float32)

    def preprocess(example, transform_image):
        image = example['image']
        image = transform_image(image)
        coco_categories = example['objects']['label']

        # Check if the image has no objects
        is_empty = tf.equal(tf.size(coco_categories), 0)

        # Use tf.cond to set labels to zeros if the image has no objects or to encode labels if the image has objects
        labels = tf.cond(is_empty, set_labels_as_zeros, lambda: encode_coco_categories(coco_categories, num_classes))
        return image, labels

    train_ds = dataset['train'].map(lambda x: preprocess(x, transform_image)).batch(batch_size).prefetch(
        tf.data.AUTOTUNE)
    val_ds = dataset['validation'].map(lambda x: preprocess(x, transform_image)).batch(batch_size).prefetch(
        tf.data.AUTOTUNE)

    return train_ds, val_ds
