import os
import numpy as np

from src.algonauts.evaluators.correlations import find_best_correlations
from src.algonauts.utils.file import save_predictions


def merge_predictions_for_all_subjects(subjects, data_dir, exp_output_dir):
    """
    Merge predictions from the best layers for each ROI, to create merged prediction files for given subjects
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
    Merge predictions from the best layers for each ROI for the given subject, to create a merged prediction file
    Prediction data for each ROI will be taken from the best layer
    Will first get predictions from the best layer for all vertices, then write over the ROIs
    :param subj: subject (eg. 1)
    :param roi_indices: dictionary of format {'LH': {roi_name: roi_idxs}, 'RH': {roi_name: roi_idx}}
    :param exp_output_dir: folder to fetch prediction files for the layers
    :return:
    """
    # Get list of ROI names
    roi_names = [roi_name for roi_name in roi_indices['LH'].keys()]

    # Get dictionary of layers with the best correlations per ROI
    best_correlations = find_best_correlations(f'{exp_output_dir}/results.json', subj)

    # Get names of all layers inside the best correlations map
    layers = set([hemi_val['layer'] for hemi_info in best_correlations.values() for hemi_val in hemi_info.values()])

    # Load predictions for the layers
    layer_predictions = get_layer_predictions(exp_output_dir, layers, subj)

    # Get predictions of the best layer for all vertices and set it as default predictions
    lh_best_overall_layer = best_correlations['LH']['All vertices']['layer']
    rh_best_overall_layer = best_correlations['RH']['All vertices']['layer']

    lh_default_predictions = layer_predictions[lh_best_overall_layer]['LH']
    rh_default_predictions = layer_predictions[rh_best_overall_layer]['RH']

    # Merge predictions
    for roi in roi_names:
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
    subject_submission_dir = f'{exp_output_dir}/best_layers_merged/{subj}'
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
    Get a map of "roi_name: roi_indices" for each ROI in each hemisphere for the given subject
    :param data_dir: folder containing challenge data, used to fetch ROI mappings
    :param subj: subject (eg. 1)
    :return: dictionary with format {'LH': {'roi_name': roi_idx}, 'RH': {'roi_name': roi_idx}}
    """
    subj = f'subj{format(subj, "02")}'
    roi_data_dir = f'{data_dir}/{subj}'
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_streams.npy', 'mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(roi_data_dir, 'roi_masks', r),
                                     allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.streams_challenge_space.npy', 'lh.prf-visualrois_challenge_space.npy',
                              'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
                              'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
                              ]
    rh_challenge_roi_files = ['rh.streams_challenge_space.npy', 'rh.prf-visualrois_challenge_space.npy',
                              'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
                              'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
                              ]
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(roi_data_dir, 'roi_masks',
                                                      lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(roi_data_dir, 'roi_masks',
                                                      rh_challenge_roi_files[r])))

    # Create map of roi_name: roi_indices for each ROI in each hemisphere
    roi_indices = {'LH': {}, 'RH': {}}
    for roi_type_id in range(len(lh_challenge_rois)):
        for roi_id, roi_name in roi_name_maps[roi_type_id].items():
            if roi_id != 0:  # zeros indicate to vertices falling outside the ROI of interest
                lh_roi_idx = np.where(lh_challenge_rois[roi_type_id] == roi_id)[0]
                rh_roi_idx = np.where(rh_challenge_rois[roi_type_id] == roi_id)[0]
                roi_indices['LH'][roi_name] = lh_roi_idx
                roi_indices['RH'][roi_name] = rh_roi_idx
                print(f'{roi_name}')
    return roi_indices
