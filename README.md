# Project: Algonauts Challenge 2023

This is a repository for the course project "Algonauts Challenge 2023", of the Winter Semester 2022/2023 course
"Implementing Artificial Neural Networks with Tensorflow" at the Institute of Cognitive Science, University of Osnabrück.

The goal of this project is to train models and experiment with different settings for artificial neural networks (ANNs)
and see how they perform for the Algonauts Challenge 2023.

The project team consists of the following authors:

Andrei Klimenok, Pelin Kömürlüoğlu, Ibrahim Muhip Tezcan

{aklimenok, pkoemuerlueo, itezcan}@uos.de

<!-- TOC -->
* [Project: Algonauts Challenge 2023](#project--algonauts-challenge-2023)
  * [Installation & Setup](#installation--setup)
    * [Challenge Data](#challenge-data)
    * [Dependencies](#dependencies)
  * [Usage](#usage)
    * [Running the pipeline using notebooks](#running-the-pipeline-using-notebooks)
    * [Running the pipeline on Python](#running-the-pipeline-on-python)
    * [Output files](#output-files)
  * [File Structure](#file-structure)
  * [Results](#results)
  * [Contributing](#contributing)
  * [License](#license)
  * [References](#references)
<!-- TOC -->

## Installation & Setup

### Challenge Data

To be able to run the code in the project, you need to have the Algonauts Challange 2023 dataset in your environment.
This dataset contains the following information:
- `roi_masks`:  .npy files containing indices and masks for each ROI
- `test_split`: Stimulus images for each subject in the Natural Scenes Dataset (NSD) test split
- `training_split`: Stimulus images & fMRI responses for each subject in NSD training split

For more information on the dataset and instructions to download it, you can go to the challenge website:
http://algonauts.csail.mit.edu/challenge.html

### Dependencies
To run the main project pipeline that generates predictions for a given model, you need the following dependencies installed:
```
keras
python
scikit-learn
tensorflow
tensorflow-datasets
torch                    (Optional)
torchvision              (Optional)
```
The project contains code with both tensorflow and PyTorch, however most of the work is done on Tensorflow. PyTorch is only needed in order to run the baseline model from the Algonauts Challenge 2023 Tutorial notebook.

Additionally, you need the following to run the notebooks:
```
jupyter
tensorflow-addons
```

### Self-Trained Models (Optional)

If you want to use the self-trained models instead of training them
yourself, you can download the models [here](https://myshare.uni-osnabrueck.de/d/fbcb6c2079184184b3a8/)

The models should be saved in the folder `data/models`

## Usage

### Running the pipeline using notebooks

For generating output (challenge submission files, variance graphs and correlation results) configure and run the
notebook [tf_pipeline.ipynb](notebooks/tf_pipeline.ipynb)

You need to specify the name `experiment` which is used to create output folders.

You also need to select the proper environment. 
- For running locally, you can pick `jupyter_notebook`
- You can also run on Paperspace or Google Colab with some further configuration (ie. mounting drives)

In the pipeline notebook you can either choose to load a pretrained model from keras.applications library (eg. VGG16)
or choose to load a model from file where a model was trained in one of the training notebooks, such as
[train_alexnet_on_coco.ipynb](notebooks/training/train_alexnet_on_coco.ipynb). If loading from a file, the model file should be present, with the same name as the
`experiment`.

There are also notebooks with experiment results that contain runnable code, for example [alexnet_coco.ipynb](notebooks/challenge/alexnet_coco.ipynb)

For training/fine-tuning models, use the notebooks under [notebooks/training](notebooks/training/train_alexnet_on_imagenette.ipynb)

### Running the pipeline on Python

Alternatively, you can run the challenge pipeline directly on Python instead of a notebook.

For this, you need to configure certain parameters and run the method `run_tf_pipeline()` inside
[src/algonauts/pipelines/tf_pipeline.py](src/algonauts/pipelines/tf_pipeline.py)

An example is given in [Main.py](Main.py)

The function requires the parameters below:
```python
run_tf_pipeline(batch_size=batch_size,
                model_loader=model_loader,
                layers=layers,
                subjects=subjects,
                challenge_data_dir=challenge_data_dir,
                exp_output_dir=exp_output_dir)
```

### Output files

Running the pipeline will create the following output files under `data/out/{experiment}` folder:
- `lh_pred_test.npy` and `rh_pred_test.npy` files, which are the predictions saved in the numpy array file format
  - Saved under `{layer_name}/algonauts_2023_challenge_submission` folder for each subject.
  - Contents of `{layer_name}/algonauts_2023_challenge_submission` folder can be zipped & submitted to the challenge
  - [Example](data/out/sample_experiment/layer1/algonauts_2023_challenge_submission/subj01/lh_pred_test.npy)
- `correlations.png`, graph showing correlations of predictions and actual brain data on the validation set
  - Saved under `{layer_name}/variance_graphs/{subject}`
  - [Example](data/out/sample_experiment/layer1/variance_graphs/subj01/correlations.png)
- `results.json` with correlation numbers for each ROI per hemisphere, for each subject and layer
  - [Example](data/out/sample_experiment/results.json)

If working remotely, you can mount something like Google Drive to the `data/out/{experiment}` folder, so that you can
access the output files later.

## File Structure
Below is the basic file structure of the repository:
```
.
|-- LICENSE.md                                                  # License file
|-- Main.py                                                     # Main script to run challenge pipeline
|-- README.md                                                   # Current file
|-- data                                                        # Main data folder
|   `-- out                                                     # Main folder for output files
|       `-- sample_experiment                                   # Output example
|           |-- layer1
|           |   |-- algonauts_2023_challenge_submission
|           |   |   `-- subj01
|           |   |       |-- lh_pred_test.npy                    # Submission file for left hemisphere
|           |   |       `-- rh_pred_test.npy                    # Submission file for right hemisphere
|           |   `-- variance_graphs
|           |       `-- subj01
|           |           `-- correlations.png                    # Graph showing ROI correlations for subject and layer
|           `-- results.json                                    # ROI correlations per subject, hemisphere and layer
|-- notebooks                                                   # Experiments and pipeline templates
|   |-- challenge                                               # Challenge pipelines with experiment results
|   |   |-- alexnet_coco.ipynb
|   |   |-- alexnet_imagenette_finetune_on_coco.ipynb
|   |   |-- alexnet_on_cifar100.ipynb
|   |   |-- alexnet_on_imagenette.ipynb
|   |   `-- alexnet_random_weights.ipynb
|   |-- tf_pipeline.ipynb                                       # Template for running pipeline with tensorflow
|   |-- torch_pipeline.ipynb                                    # Template for running pipeline with torch
|   `-- training                                                # Training pipelines
|       |-- finetune_alexnet_on_coco.ipynb
|       |-- finetune_vgg16_on_coco.ipynb
|       |-- train_alexnet_on_cifar100.ipynb
|       |-- train_alexnet_on_coco.ipynb
|       `-- train_alexnet_on_imagenette.ipynb
`-- src                                                         # Main folder for python source files
    |-- __init__.py
    `-- algonauts
        |-- __init__.py
        |-- data_processors                                     # Any module related to data processing
        |   |-- __init__.py
        |   |-- coco_dataset.py                                 # Prepare a coco dataset
        |   |-- image_transforms.py                             # Image transform functions for models
        |   |-- nsd_dataset.py                                  # Utils for working with nsd dataset
        |   |-- tf_dataloader.py                                # Tensorflow-specific data utils
        |   `-- torch_dataloader.py                             # Torch-specific data utils
        |-- encoders                                            # Anything related to encoding models
        |   |-- __init__.py
        |   `-- linear_encoder.py                               # Linear regressor(s) for predicting brain data
        |-- evaluators                                          # Anything related to evaluating results
        |   |-- __init__.py
        |   |-- correlations.py                                 # For calculating and saving correlations
        |   `-- predictions.py                                  # For getting best predictions and merging
        |-- feature_extractors                                  # Extracting features from model layers
        |   |-- __init__.py
        |   |-- tf_feature_extractor.py                         # Tensorflow feature extraction
        |   `-- torch_feature_extractor.py                      # Torch feature extraction
        |-- models                                              # Custom model definitions and utils to load models
        |   |-- __init__.py
        |   |-- alexnet.py                                      # Alexnet implementation
        |   `-- model_loaders.py                                # Util to load models
        |-- pipelines                                           # Full challenge pipelines to generate predictions
        |   |-- __init__.py
        |   `-- tf_pipeline.py                                  # Tensorflow challenge pipeline
        |-- scripts                                             # Scripts to run pipelines for self-trained model
        |   |-- pipeline_alexnet_cifar100.py
        |   |-- pipeline_alexnet_coco.py
        |   |-- pipeline_alexnet_imagenette.py
        |   |-- pipeline_alexnet_imagenette_finetuned.py
        |   `-- pipeline_alexnet_random_weights.py
        `-- utils                                               # Generic utils
            |-- console.py                                      # Utils for console logs
            `-- file.py                                         # Utils for reading/writing files

```

## Results

The full results are documented in [the project report](https://github.com/iannwtf-algonauts/algonauts23/blob/main/project_submission/Algonauts_Challenge_Report_team-iannwtf.pdf), which can be found in the "project_submission" folder.

Here are some examples from challenge scores on test splits:

| model          | weights    | source | challenge score |
|----------------|------------|--------|-----------------|
| AlexNet        | random     | torch  | 14.6606758043   |
| AlexNet        | random     | self   | 15.1270050698   |
| AlexNet        | cifar100   | self   | 21.3985024228   |
| AlexNet        | imagenette | self   | 22.0083790411   |
| AlexNet        | COCO       | self   | 24.1801036288   |
| AlexNet        | ImageNet   | torch  | 40.1115725758   |
| VGG-16         | ImageNet   | keras  | 40.4690900609   |
| RESNET-50      | ImageNet   | keras  | 43.2214229574   |
| EfficientNetB2 | ImageNet   | keras  | 47.8162146576   |


## Contributing

This repository is for a course project, and we are not actively seeking outside contributions.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for more details.


## References

[The Algonauts Challenge 2023](http://algonauts.csail.mit.edu/challenge.html)