import json
import numpy as np
import os


def save_predictions(lh_fmri_test_pred, rh_fmri_test_pred, subject_submission_dir):
    lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
    rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)

    # Create folder if it does not exist
    if not os.path.isdir(subject_submission_dir):
        os.makedirs(subject_submission_dir)

    # Save prediction arrays as files
    np.save(os.path.join(subject_submission_dir, 'lh_pred_test.npy'), lh_fmri_test_pred)
    np.save(os.path.join(subject_submission_dir, 'rh_pred_test.npy'), rh_fmri_test_pred)


def read_json_file(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        data = {}
    return data


def write_json_file(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=2)
