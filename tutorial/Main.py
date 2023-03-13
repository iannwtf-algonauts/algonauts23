import numpy as np
import torch
from torchvision.models.feature_extraction import get_graph_node_names

import Correlations
import LinearEncoder
from NSDDataset import NSDDataset
from tutorial.torch_code.TorchFeatureExtractor import TorchFeatureExtractor

data_dir = '../../algonauts_2023_challenge_data'
parent_output_dir = '../output'

# Pick mps for Apple chip, cuda for nvidia & cpu otherwise
device = 'mps'  # @param ['cpu', 'cuda'] {allow-input: true}
device = torch.device(device)

batch_size = 300
random_seed = 5
feature_extractor = TorchFeatureExtractor(batch_size, random_seed, device)

model_name = 'resnet50_torch_pt'

# layers = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5",
#           "classifier.6"]
# layers = train_nodes
# layers = ['features.12']
layers = ['layers.3.2.relu_2']
# subjects = [1]
subjects = [1, 2, 3, 4, 5, 6, 7, 8]
for layer_name in layers:
    print(f'Running for layer {layer_name}')
    for subj in subjects:
        print(f'Running for subject {subj}')

        # Set data directories based on parameters
        output_dir = f'{parent_output_dir}/{model_name}/{layer_name}'
        dataset = NSDDataset(data_dir, output_dir, subj)

        # Load model
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)

        model.to(device)  # send the model to the chosen device ('cpu' or 'cuda')
        model.eval()  # set the model to evaluation mode, since you are not training it

        # See what layers the model has
        train_nodes, _ = get_graph_node_names(model)
        print(train_nodes)

        train_features, val_features, test_features = feature_extractor.extract_all_features(dataset, model, layer_name)

        del model

        # Apply PCA and fit linear encoder to get fMRI predictions
        lh_fmri_val_pred, rh_fmri_val_pred = LinearEncoder.predict(dataset, train_features, val_features, test_features)
        # Calculate correlations for each hemisphere
        lh_correlation = Correlations.calculate_correlation(lh_fmri_val_pred, dataset.lh_fmri_val)
        rh_correlation = Correlations.calculate_correlation(rh_fmri_val_pred, dataset.rh_fmri_val)
        print(f'Left hemisphere median correlation: {np.median(lh_correlation)}')
        print(f'Right hemisphere median correlation: {np.median(rh_correlation)}')
        # Plot and save correlation graph
        plot_file = f'{dataset.var_plot_dir}/correlations.png'
        Correlations.plot_correlations(dataset.data_dir, lh_correlation, rh_correlation, plot_file)
        result_file = open(f'{parent_output_dir}/{model_name}/results.txt', 'a')
        result_file.write(f'Layer: {layer_name} Subject: {subj}\n')
        result_file.write(f'LH Correlation: {np.median(lh_correlation)} RH Correlation: {np.median(rh_correlation)}\n')
        result_file.close()
