import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from src.algonauts.utils.console import HiddenPrints


def slice_model(model, layer_name):
    """
    Slice the model to extract features from a specific layer
    :param model: model to be sliced
    :param layer_name: layer to slice at
    :return: sliced model, or the given model if layer_name not given
    """
    if layer_name is not None:
        return tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    else:
        return model


def train_pca(model, train_dataset):
    """
    Train PCA batch-by-batch for the given model
    :param model: model / sliced model to extract features for training PCA
    :param train_dataset: training dataset to be used in training PCA
    :return:
    """
    pca = IncrementalPCA()
    print("Training PCA")
    for batch in tqdm(train_dataset):
        # Extract features
        with HiddenPrints():
            ft = model.predict(batch)
        # Flatten the features
        ft = ft.reshape(ft.shape[0], -1)
        # Fit PCA for this batch
        pca.partial_fit(ft)
    print("PCA training finished")
    print(f'PCA components: {pca.n_components_}')
    return pca


def extract_and_transform_features(dataset, model, pca):
    """
    For each batch, use model to get dimensionally reduced predictions
    :param dataset: tf dataset to get batches from
    :param model: model to get predictions from
    :param pca: dimensionality reducer
    :return: all features for all batches, vertically stacked
    """
    features = []
    for batch in tqdm(dataset):
        with HiddenPrints():
            ft = model.predict(batch)
        # Flatten the features
        ft = ft.reshape(ft.shape[0], -1)
        # Fit PCA to batch of features
        ft = pca.transform(ft)
        features.append(ft)
    return np.vstack(features)
