import os
import tensorflow as tf


def load_datasets(dataset, transform_image, batch_size):
    """
    Prepare training, validation and test datasets
    :param dataset: NSDDataset object to get training/test data from
    :param transform_image: image transform to be applied to the images
    :param batch_size: batch size for the tensorflow dataset
    :return: train_ds, val_ds, test_ds
    """
    train_val_imgs_paths = [os.path.join(dataset.train_img_dir, img_name) for img_name in dataset.training_img_list]
    train_paths = [train_val_imgs_paths[i] for i in dataset.idxs_train]
    val_paths = [train_val_imgs_paths[i] for i in dataset.idxs_val]
    test_imgs_paths = [os.path.join(dataset.test_img_dir, img_name) for img_name in dataset.test_img_list]

    train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    test_ds = tf.data.Dataset.from_tensor_slices(test_imgs_paths)

    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def preprocess(ds):
        ds = ds.map(load_image)
        ds = ds.map(transform_image)
        return ds

    train_ds = preprocess(train_ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = preprocess(val_ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = preprocess(test_ds).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds
