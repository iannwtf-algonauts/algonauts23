from sklearn.linear_model import Ridge


def predict(dataset, features_train, features_val, features_test):
    """
    Use a linear encoder to predict fmri data from given features
    :param dataset: NSDDataset object, used to get fmri data
    :param features_train: training features
    :param features_val: validation features
    :param features_test: test features
    :return: predictions as a dictionary {val:{left, right}, test:{left, right}}
    """
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

    return {'val': {'left': lh_fmri_val_pred, 'right': rh_fmri_val_pred},
            'test': {'left': lh_fmri_test_pred, 'right': rh_fmri_test_pred}}
