import numpy as np
import os
from sklearn.linear_model import LinearRegression
from src.algonauts.evaluators import correlations as corr
from src.algonauts.data_processors.nsd_dataset import NSDDataset
from src.algonauts.data_processors.tf_dataloader import load_datasets
from src.algonauts.data_processors.reduce_dims import train_pca
from src.algonauts.feature_extractors.tf_feature_extractor import extract_and_transform_features, slice_model


def run_tf_pipeline(batch_size, model_loader, layers, subjects, challenge_data_dir, exp_output_dir):
    """
    Runs the whole pipeline with given parameters, for each layer and subject
    :param batch_size: batch size for loading dataset
    :param model_loader: lambda function that returns model and image transform
    :param layers: layers to run the pipeline for
    :param subjects: subjects to run the pipeline for
    :param challenge_data_dir: folder to the algonauts challenge data
    :param exp_output_dir: output directory to save predictions and correlations
    :return:
    """
    for layer_name in layers:
        print(f'Running for layer {layer_name}')
        for subj in subjects:
            print(f'Running for subject {subj}')

            # Set data directories based on parameters
            output_dir = f'{exp_output_dir}/{layer_name}'
            dataset = NSDDataset(challenge_data_dir, output_dir, subj)

            model, transform_image = model_loader()
            print('Loading datasets...')
            train_ds, val_ds, test_ds = load_datasets(dataset, transform_image, batch_size)
            print('Datasets loaded')

            # Slice model at layer for feature extraction
            model = slice_model(model, layer_name)

            # Train PCA
            print('Training PCA...')
            pca = train_pca(model, train_ds)
            print('PCA over')

            # Extract and transform features
            print('Extracting and transforming features...')
            train_features = extract_and_transform_features(train_ds, model, pca)
            val_features = extract_and_transform_features(val_ds, model, pca)
            test_features = extract_and_transform_features(test_ds, model, pca)
            print('Features extracted and transformed')

            # Delete model to free up memory
            del model, pca

            # Fit regression
            print('Fitting regression...')
            reg_lh = LinearRegression().fit(train_features, dataset.lh_fmri_train)
            reg_rh = LinearRegression().fit(train_features, dataset.rh_fmri_train)
            print('Regression fitted')

            # Use fitted linear regressions to predict the validation and test fMRI data
            print('Predicting fMRI data...')
            lh_fmri_val_pred = reg_lh.predict(val_features)
            lh_fmri_test_pred = reg_lh.predict(test_features)
            rh_fmri_val_pred = reg_rh.predict(val_features)
            rh_fmri_test_pred = reg_rh.predict(test_features)
            print('fMRI data predicted')
            # Calculate correlations for each hemispher
            print('Calculating correlations...')
            lh_correlation = corr.calculate_correlation(lh_fmri_val_pred, dataset.lh_fmri_val)
            rh_correlation = corr.calculate_correlation(rh_fmri_val_pred, dataset.rh_fmri_val)
            print('Correlations calculated')

            corr.plot_and_write_correlations(dataset, lh_correlation, rh_correlation, exp_output_dir, layer_name, subj)
            # Save test predictions
            np.save(os.path.join(dataset.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
            np.save(os.path.join(dataset.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)
