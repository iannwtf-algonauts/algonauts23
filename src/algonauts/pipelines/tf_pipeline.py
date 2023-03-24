from src.algonauts.data_processors.nsd_dataset import NSDDataset
from src.algonauts.data_processors.tf_dataloader import load_datasets
import src.algonauts.feature_extractors.tf_feature_extractor as fe
from src.algonauts.encoders.linear_encoder import predict
import src.algonauts.evaluators.correlations as corr
from src.algonauts.utils.file import save_predictions


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
            model = fe.slice_model(model, layer_name)

            # Train PCA
            pca = fe.train_pca(model, train_ds)

            # Extract and transform features
            print('Extracting and transforming features...')
            train_features = fe.extract_and_transform_features(train_ds, model, pca)
            val_features = fe.extract_and_transform_features(val_ds, model, pca)
            test_features = fe.extract_and_transform_features(test_ds, model, pca)
            print('Features extracted and transformed')

            # Delete model to free up memory
            del model, pca

            pred = predict(dataset, train_features, val_features, test_features)

            # Calculate correlations for each hemisphere
            print('Calculating left hemisphere correlations...')
            lh_correlation = corr.calculate_correlation(pred['val']['left'], dataset.lh_fmri_val)
            print('Calculating right hemisphere correlations...')
            rh_correlation = corr.calculate_correlation(pred['val']['right'], dataset.rh_fmri_val)
            print('Correlations calculated')

            # Plot and write correlations
            corr.plot_and_write_correlations(dataset, lh_correlation, rh_correlation, exp_output_dir, layer_name, subj)

            # Save test predictions
            save_predictions(lh_fmri_test_pred=pred['test']['left'],
                             rh_fmri_test_pred=pred['test']['right'],
                             subject_submission_dir=dataset.subject_submission_dir)
