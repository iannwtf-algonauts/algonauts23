"""
Run challenge pipeline for Alexnet trained on CIFAR-100
"""

import json
from src.algonauts.data_processors.image_transforms import transform_alexnet
from src.algonauts.evaluators.correlations import find_best_correlations
from src.algonauts.evaluators.predictions import merge_predictions_for_all_subjects
from src.algonauts.models import model_loaders
from src.algonauts.pipelines.tf_pipeline import run_tf_pipeline

# Set experiment parameters
experiment = 'alexnet_cifar100'
batch_size = 300

base_dir = '../../..'
challenge_data_dir = f'{base_dir}/data/algonauts_2023_challenge_data'
exp_output_dir = f'{base_dir}/data/out/{experiment}'

# Alexnet loader
model_filename = f'{base_dir}/data/models/alexnet_cifar100.h5'
model_loader = lambda: model_loaders.load_from_file(model_filename, transform_alexnet)

# Load model and print layers
model, _ = model_loader()
print(*(layer.name for layer in model.layers), sep=' -> ')
del model

# Pick layers to extract features from
layers = ['conv2d_5_pool']
subjects = [1, 2, 3, 4, 5, 6, 7, 8]

# Run the pipeline
run_tf_pipeline(batch_size=batch_size, model_loader=model_loader, layers=layers, subjects=subjects,
                challenge_data_dir=challenge_data_dir,
                exp_output_dir=exp_output_dir)

# Print best layers per ROI
subj = 1
result = find_best_correlations(f'{exp_output_dir}/results.json', subj)
print(json.dumps(result, indent=2))

# Generate merged predictions
merge_predictions_for_all_subjects(subjects, challenge_data_dir, exp_output_dir)
