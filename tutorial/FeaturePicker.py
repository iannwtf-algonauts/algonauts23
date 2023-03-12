from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from tqdm import tqdm
import numpy as np


def get_predictions(dataloader, model, model_layer):
    features = []
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])
    print(f'Collecting features from layer {model_layer}')
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        batch_predictions = ft.detach().cpu().numpy()
        features.append(batch_predictions)
    return np.vstack(features)
