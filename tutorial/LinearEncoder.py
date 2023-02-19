import numpy as np
import os

from sklearn.decomposition import PCA
from FeaturePicker import get_predictions
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def predict(model, layer_name, dataset, dataloaders):
    train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader = dataloaders
    # Extract training features
    features_train = get_predictions(train_imgs_dataloader, model, layer_name)
    print('\nTraining images features:')
    print(features_train.shape)
    print('(Training stimulus images × PCA features)')

    # Prepare data processing pipeline
    pipe_l = Pipeline([
                     ("reduce_dims", PCA()),
                     ("lin_reg", Ridge())
                     ])

    pipe_r = Pipeline([
                     ("reduce_dims", PCA()),
                     ("lin_reg", Ridge())
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

    # Extract validation features
    features_val = get_predictions(val_imgs_dataloader, model, layer_name)
    print('\nValidation images features:')
    print(features_val.shape)
    print('(Validation stimulus images × PCA features)')

    # Extract test features
    features_test = get_predictions(test_imgs_dataloader, model, layer_name)
    print('\nTest images features:')
    print(features_test.shape)
    print('(Test stimulus images × PCA features)')

    # Use fitted linear regressions to predict the validation and test fMRI data
    lh_fmri_val_pred = grid_l.predict(features_val)
    lh_fmri_test_pred = grid_l.predict(features_test)
    rh_fmri_val_pred = grid_r.predict(features_val)
    rh_fmri_test_pred = grid_r.predict(features_test)

    # Save test predictions
    np.save(os.path.join(dataset.subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(dataset.subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)

    return lh_fmri_val_pred, rh_fmri_val_pred
