import numpy as np
import os
from FeaturePicker import FeaturePicker
from sklearn.linear_model import LinearRegression


def predict(batch_size, model, layer_name, dataset, dataloaders):
    folder = f'{dataset.subject_submission_dir}/{layer_name}_{batch_size}'
    print(folder)
    # Create the submission directory if not existing
    if not os.path.isdir(folder):
        os.makedirs(folder)

    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = dataloaders
    feature_picker = FeaturePicker(batch_size, model, layer_name)
    pca = feature_picker.fit_pca(train_imgs_dataloader)
    features_train = feature_picker.extract_features(train_imgs_dataloader, pca)
    features_val = feature_picker.extract_features(val_imgs_dataloader, pca)
    features_test = feature_picker.extract_features(test_imgs_dataloader, pca)
    del model, pca

    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    # Use training fmri data to fit linear regression for each hemisphere
    reg_lh = LinearRegression().fit(features_train, dataset.lh_fmri_train)
    reg_rh = LinearRegression().fit(features_train, dataset.rh_fmri_train)

    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_fmri_val_pred = reg_lh.predict(features_val)
    lh_fmri_test_pred = reg_lh.predict(features_test)
    rh_fmri_val_pred = reg_rh.predict(features_val)
    rh_fmri_test_pred = reg_rh.predict(features_test)

    np.save(os.path.join(folder, 'lh_pred_val.npy'), lh_fmri_val_pred)
    np.save(os.path.join(folder, 'rh_pred_val.npy'), rh_fmri_val_pred)

    np.save(os.path.join(folder, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(folder, 'rh_pred_test.npy'), rh_fmri_test_pred)

    return lh_fmri_val_pred, rh_fmri_val_pred
