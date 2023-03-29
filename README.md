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
    * [Running the pipeline on Python](#running-the-pipeline-on-python)
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


## Usage

For generating output (challenge submission files, variance graphs and correlation results) configure and run the
notebook `tf_pipeline.ipynb`

You need to specify `experiment_name` which is used to create output folders.

You also need to select the proper environment. 
- For running locally, you can pick `jupyter_notebook`
- You can also run on paperspace or Google Colab with some further configuration

### Running the pipeline on Python

Alternatively, you can run the challenge pipeline directly on Python instead of a notebook.

For this, you need to configure certain parameters and run the method `run_tf_pipeline()` inside
`src/algonauts/pipelines/tf_pipeline.py`

An example is given in `src/algonauts/Main.py`

The function requires the parameters below:
```python
run_tf_pipeline(batch_size=batch_size,
                model_loader=model_loader,
                layers=layers,
                subjects=subjects,
                challenge_data_dir=challenge_data_dir,
                exp_output_dir=exp_output_dir)
```
## File Structure
Below is the basic file structure of the repository:
```
├── README.md  # current file
├── data  # main data folder
│   └── out  # main folder for output files
│       └── sample_experiment  ## output example
│           ├── layer1
│           │   ├── algonauts_2023_challenge_submission
│           │   │   └── subj01
│           │   │       ├── lh_pred_test.npy  # submission file for left hemisphere
│           │   │       └── rh_pred_test.npy  # submission file for right hemisphere
│           │   └── variance_graphs
│           │       └── subj01
│           │           └── correlations.png  # graph showing ROI correlations
│           └── results.json  # correlations for ROIs per subject, hemisphere and layer
├── notebooks  # experiments and pipeline templates
│   ├── finetune_vgg16_on_coco.ipynb  # transfer learning on coco
│   ├── tf_pipeline.ipynb     # template for running pipeline with tensorflow
│   ├── torch_pipeline.ipynb  # template for running pipeline with torch
│   └── train_alexnet_on_coco.ipynb  # training a model from scratch on coco
└── src  # main folder for python source files
    ├── __init__.py
    └── algonauts
        ├── LICENSE.md  # project license for use
        ├── __init__.py
        ├── data_processors  # any module related to data processing
        │   ├── __init__.py
        │   ├── coco_dataset.py      # prepare a coco dataset
        │   ├── image_transforms.py  # image transform functions for models
        │   ├── nsd_dataset.py       # utils for working with nsd dataset
        │   ├── tf_dataloader.py     # tensorflow-specific data utils
        │   └── torch_dataloader.py  # torch-specific data utils
        ├── encoders  # anything related to encoding models
        │   ├── __init__.py
        │   └── linear_encoder.py  # linear regressor(s) for predicting brain data
        ├── evaluators  # anything related to evaluating results
        │   ├── __init__.py
        │   ├── correlations.py  # for calculating and saving correlations
        │   └── predictions.py   # for getting best predictions and merging
        ├── feature_extractors  # extracting features from model layers
        │   ├── __init__.py
        │   ├── tf_feature_extractor.py     # tensorflow feature extraction
        │   └── torch_feature_extractor.py  # torch feature extraction
        ├── models  # custom model definitions and utils to load models
        │   ├── __init__.py
        │   ├── alexnet.py        # alexnet implementation
        │   └── model_loaders.py  # util to load models
        ├── pipelines  # full challenge pipelines to generate predictions
        │   ├── __init__.py
        │   └── tf_pipeline.py  # tensorflow challenge pipeline
        └── utils  # generic utils
            ├── console.py  # utils for console logs
            └── file.py     # utils for reading/writing files
```

## Results

Full results will be documented in the project report.

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