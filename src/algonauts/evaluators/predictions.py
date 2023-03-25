import os
import numpy as np

from src.algonauts.evaluators.correlations import find_best_correlations
from src.algonauts.utils.file import save_predictions


def merge_predictions_for_all_subjects(subjects, data_dir, exp_output_dir):
    """
    Merge the best predictions for each ROI, to create a merged prediction file
    :param subjects: list of subjects (eg. [1, 2])
    :param data_dir: folder containing challenge data
    :param exp_output_dir: folder containing output submission data
    :return:
    """
    for subj in subjects:
        print(f'Merging predictions for subject {subj}')
        roi_indices = load_roi_indices(data_dir, subj)
        merge_predictions_per_subject(subj, roi_indices, exp_output_dir)
        print(f'{subj} merged.')


def merge_predictions_per_subject(subj, roi_indices, exp_output_dir):
    """
    Merge the best predictions for each ROI for the given subject, to create a merged prediction file
    Prediction data for each ROI will be taken from the best layer
    Will first get predictions from the best overall layer, then write over the ROIs
    :param subj: subject (eg. 1)
    :param roi_indices: dictionary of format {'LH': {roi_name: roi_idxs}, 'RH': {roi_name: roi_idx}}
    :param exp_output_dir: folder to fetch prediction files for the layers
    :return: final version of predictions
    """
    # Define which ROIs we are interested in
    rois_to_use = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1', 'FBA-2', 'mTL-bodies', 'OFA',
                   'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2',
                   'mfs-words', 'mTL-words']

    # Get dictionary of layers with the best correlations per ROI
    best_correlations = find_best_correlations(f'{exp_output_dir}/results.json', subj)

    # Get names of all layers that we will use
    layers = set([hemi_val['layer'] for hemi_info in best_correlations.values() for hemi_val in hemi_info.values()])

    # Load predictions for the layers
    layer_predictions = get_layer_predictions(exp_output_dir, layers, subj)
    lh_best_overall_layer = best_correlations['LH']['All vertices']['layer']
    rh_best_overall_layer = best_correlations['RH']['All vertices']['layer']

    lh_default_predictions = layer_predictions[lh_best_overall_layer]['LH']
    rh_default_predictions = layer_predictions[rh_best_overall_layer]['RH']

    # Merge predictions
    for roi in rois_to_use:
        lh_idx = roi_indices['LH'][roi]
        rh_idx = roi_indices['RH'][roi]

        lh_best_layer = best_correlations['LH'][roi]['layer']
        rh_best_layer = best_correlations['RH'][roi]['layer']

        if lh_best_layer is not None:
            lh_roi_best_predictions = layer_predictions[lh_best_layer]['LH'][:, lh_idx]
            lh_default_predictions[:, lh_idx] = lh_roi_best_predictions

        if rh_best_layer is not None:
            rh_roi_best_predictions = layer_predictions[rh_best_layer]['RH'][:, rh_idx]
            rh_default_predictions[:, rh_idx] = rh_roi_best_predictions

    subj = f'subj{format(subj, "02")}'
    subject_submission_dir = f'{exp_output_dir}/best/{subj}'
    print(f'Saving merged predictions to {subject_submission_dir}')
    save_predictions(lh_default_predictions, rh_default_predictions, subject_submission_dir)


def get_layer_predictions(exp_output_dir, layers, subj):
    """
    Map layer names to array of predictions, per each hemisphere
    :param exp_output_dir: directory to fetch the prediction files from
    :param layers: layer names
    :param subj: subject (eg. 1)
    :return: Dictionary of format {'layer_name': {'LH': lh_predictions, 'RH': rh_predictions}}
    """
    subj = f'subj{format(subj, "02")}'
    layer_predictions = {}
    for layer in layers:
        if layer is not None:
            layer_prediction_dir = f'{exp_output_dir}/{layer}/algonauts_2023_challenge_submission/{subj}'
            lh_pred = np.load(f'{layer_prediction_dir}/lh_pred_test.npy')
            print(f'LH pred shape: {lh_pred.shape}')
            rh_pred = np.load(f'{layer_prediction_dir}/rh_pred_test.npy')
            print(f'RH pred shape: {rh_pred.shape}')
            layer_predictions[layer] = {'LH': lh_pred, 'RH': rh_pred}
    return layer_predictions


def load_roi_indices(data_dir, subj):
    """
    Get the ROI indices per each hemisphere for the given subject
    :param data_dir: folder containing challenge data, used to fetch ROI mappings
    :param subj: subject (eg. 1)
    :return: dictionary with format {'LH': {'roi_name': roi_idx}, 'RH': {'roi_name': roi_idx}}
    """
    subj = f'subj{format(subj, "02")}'
    roi_data_dir = f'{data_dir}/{subj}'
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(roi_data_dir, 'roi_masks', r),
                                     allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
                              'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
                              'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
                              'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
                              'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
                              'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
                              'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(roi_data_dir, 'roi_masks',
                                                      lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(roi_data_dir, 'roi_masks',
                                                      rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    roi_indices = {'LH': {}, 'RH': {}}
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                roi_indices['LH'][r2[1]] = lh_roi_idx
                roi_indices['RH'][r2[1]] = rh_roi_idx
                print(f'{r2[1]}')
    roi_names.append('All vertices')
    print(f'roi names: {roi_names}')
    return roi_indices
