import numpy as np
import torch
from torchvision.models.feature_extraction import get_graph_node_names

import Correlations
import LinearEncoder
from NSDDataset import NSDDataset

data_dir = '../../algonauts_2023_challenge_data'
parent_submission_dir = '../algonauts_2023_challenge_submission'
correlations_dir = '../correlations'

# Pick mps for Apple chip, cuda for nvidia & cpu otherwise
device = 'mps'  # @param ['cpu', 'cuda'] {allow-input: true}
device = torch.device(device)

batch_size = 300
n_components = 256
subj = 1  # @param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}

# Set data directories based on subject
dataset = NSDDataset(data_dir, parent_submission_dir, subj, batch_size)

# Get dataloaders
random_seed = 5
dataloaders = dataset.get_data_loaders(random_seed, device)

# Load pretrained AlexNet
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
model.to(device)  # send the model to the chosen device ('cpu' or 'cuda')
model.eval()  # set the model to evaluation mode, since you are not training it

# See what layers AlexNet has
train_nodes, _ = get_graph_node_names(model)
print(train_nodes)

# layers = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5",
#          "classifier.6"]
# layers = train_nodes
layers = ['features.2', 'features.5', 'features.7', 'features.12', 'classifier.2']
for layer_name in layers:
    print(f'Layer {layer_name}')
    lh_fmri_val_pred, rh_fmri_val_pred = LinearEncoder.predict(batch_size, model, layer_name, dataset, dataloaders, n_components)
    lh_correlation = Correlations.calculate_correlation(lh_fmri_val_pred, dataset.lh_fmri_val)
    rh_correlation = Correlations.calculate_correlation(rh_fmri_val_pred, dataset.rh_fmri_val)
    print(f'Left hemisphere median correlation: {np.median(lh_correlation)}')
    print(f'Right hemisphere median correlation: {np.median(rh_correlation)}')
    Correlations.plot_correlations(dataset.data_dir, lh_correlation, rh_correlation, f'{layer_name}_corr.png')
