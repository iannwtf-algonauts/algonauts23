import numpy as np
from sklearn.decomposition import PCA
from src.algonauts.utils.console import HiddenPrints


def train_pca(model, train_dataset):
    features = []
    for batch in train_dataset:
        # Extract features
        with HiddenPrints():
            ft = model.predict(batch)
        # Flatten the features
        ft = ft.reshape(ft.shape[0], -1)
        # Append to list
        features.append(ft)
    # Combine features from all batches
    features = np.vstack(features)
    n_components = 100
    print('Features shape: ', features.shape, ' PCA Components: ', n_components)
    # Fit PCA to combined features
    pca = PCA(n_components=n_components)
    pca.fit(features)
    # print('plot the explained variance')
    # plt.plot(pca.explained_variance_ratio_)
    # plt.show()
    # print('plot the cumulative explained variance')
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.show()
    return pca
