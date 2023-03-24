import os

import numpy as np
from sklearn.linear_model import Ridge
from src.algonauts.evaluators import correlations as corr


def predict_and_write(dataset, exp_output_dir, layer_name, subj, features_train, features_val, features_test):
    # Fit linear regressors for each hemisphere
    print("Fitting regression...")
    reg_lh = Ridge().fit(features_train, dataset.lh_fmri_train)
    reg_rh = Ridge().fit(features_train, dataset.rh_fmri_train)
    print("Finished fitting regression.")

    # Use fitted linear regressions to predict the validation and test fMRI data
    print("Predicting fMRI data...")
    lh_fmri_val_pred = reg_lh.predict(features_val)
    lh_fmri_test_pred = reg_lh.predict(features_test)

    rh_fmri_val_pred = reg_rh.predict(features_val)
    rh_fmri_test_pred = reg_rh.predict(features_test)
    print("fMRI prediction finished.")

    # Calculate correlations for each hemisphere
    print('Calculating correlations...')
    lh_correlation = corr.calculate_correlation(lh_fmri_val_pred, dataset.lh_fmri_val)
    rh_correlation = corr.calculate_correlation(rh_fmri_val_pred, dataset.rh_fmri_val)
    print('Correlations calculated')

    corr.plot_and_write_correlations(dataset, lh_correlation, rh_correlation, exp_output_dir, layer_name, subj)
    # Save test predictions
    np.save(os.path.join(dataset.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(dataset.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

    return lh_fmri_test_pred, rh_fmri_test_pred
