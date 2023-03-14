from torchvision.models.feature_extraction import create_feature_extractor
from src.algonauts.data_processors.torch_dataloader import Dataloader
import torch
from tqdm import tqdm
import numpy as np


class TorchFeatureExtractor:
    def __init__(self, batch_size, random_seed, device):
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device
        self.dataloader = Dataloader(batch_size, random_seed, device)

    def extract_all_features(self, dataset, model, layer_name):
        train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = self.dataloader.get_data_loaders(dataset)
        train_features = self.extract_features(train_imgs_dataloader, model, layer_name)
        val_features = self.extract_features(val_imgs_dataloader, model, layer_name)
        test_features = self.extract_features(test_imgs_dataloader, model, layer_name)
        del train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader
        return train_features, val_features, test_features

    def extract_features(self, dataloader, model, layer_name):
        features = []
        feature_extractor = create_feature_extractor(model, return_nodes=[layer_name])
        print(f'Collecting features from layer {layer_name}')
        for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Extract features
            ft = feature_extractor(d)
            # Flatten the features
            ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
            # Fit PCA to batch
            batch_predictions = ft.detach().cpu().numpy()
            features.append(batch_predictions)
        return np.vstack(features)
