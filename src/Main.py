from src.algonauts.models.model_loaders import load_vgg16
from src.algonauts.pipelines.tf_pipeline import run_tf_pipeline


"""
Main file for those who want to quickly run the challenge pipeline on a certain model
Here the example used is VGG16 from keras, feel free to change according to needs
"""


# Specify folders for data and output
base_dir = '..'
experiment = 'test_experiment'
challenge_data_dir = f'{base_dir}/data/algonauts_2023_challenge_data'
exp_output_dir = f'{base_dir}/data/out/{experiment}'

# Load model and list layers to pick from
model_loader = lambda: load_vgg16()
model, _ = model_loader()
print(*(layer.name for layer in model.layers), sep=' -> ')
del model

# Configure batch size, pick layers and subjects to run the pipeline for
batch_size = 300
layers = ['block5_pool']
subjects = [1  # 2, 3, 4, 5, 6, 7, 8
            ]

# Run the pipeline with the given configuration
run_tf_pipeline(batch_size=batch_size,
                model_loader=model_loader,
                layers=layers,
                subjects=subjects,
                challenge_data_dir=challenge_data_dir,
                exp_output_dir=exp_output_dir)
