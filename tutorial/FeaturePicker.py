from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np


class FeaturePicker:
    def __init__(self, batch_size, model, model_layer):
        self.batch_size = batch_size
        self.feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

    def fit_pca(self, dataloader):
        # Define PCA parameters
        pca = IncrementalPCA(n_components=100, batch_size=self.batch_size)

        # Fit PCA to batch
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = self.feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Fit PCA to batch
            pca.partial_fit(ft.detach().cpu().numpy())
        return pca

    def extract_features(self, dataloader, pca):
        features = []
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = self.feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Apply PCA transform
            ft = pca.transform(ft.cpu().detach().numpy())
            features.append(ft)
        return np.vstack(features)
