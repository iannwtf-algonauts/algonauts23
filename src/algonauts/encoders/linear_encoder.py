import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def predict(dataset, features_train, features_val, features_test):
    # Print training feature shape
    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    # Print validation feature shape
    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    # Print test feature shape
    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    # Prepare data processing pipeline
    pca_l = PCA()
    ridge_l = Ridge()
    pipe_l = Pipeline([
                     ("reduce_dims", pca_l),
                     ("lin_reg", ridge_l)
                     ])

    pca_r = PCA()
    ridge_r = Ridge()
    pipe_r = Pipeline([
                     ("reduce_dims", pca_r),
                     ("lin_reg", ridge_r)
                     ])

    # Grid search parameters. More values can be added to the arrays to do the search.
    # Setting the best values found for subject 8
    param_grid = dict(reduce_dims__n_components=[150],  # Tried params: [50, 100, 150]
                      lin_reg__alpha=[1],               # Tried params: np.logspace(-5, 5, 10)
                      lin_reg__fit_intercept=[False])   # Tried params: [True, False]

    print('Starting grid search for left hemisphere')
    grid_l = GridSearchCV(pipe_l, param_grid=param_grid, cv=10, verbose=True)
    grid_l.fit(features_train, dataset.lh_fmri_train)
    print('Grid search finished for left hemisphere')
    print(f'Best params: {grid_l.best_params_}')

    print('Starting grid search for right hemisphere')
    grid_r = GridSearchCV(pipe_r, param_grid=param_grid, cv=10, verbose=True)
    grid_r.fit(features_train, dataset.rh_fmri_train)
    print('Grid search finished for right hemisphere')
    print(f'Best params: {grid_r.best_params_}')

    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_fmri_val_pred = grid_l.predict(features_val)
    lh_fmri_test_pred = grid_l.predict(features_test)
    del pca_l, ridge_l, grid_l
    rh_fmri_val_pred = grid_r.predict(features_val)
    rh_fmri_test_pred = grid_r.predict(features_test)
    del pca_r, ridge_r, grid_r

    # Save test predictions
    np.save(os.path.join(dataset.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(dataset.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

    return lh_fmri_val_pred, rh_fmri_val_pred
