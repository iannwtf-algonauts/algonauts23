import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr as corr
import json


def calculate_correlation(prediction, truth):
    # Empty correlation array with shape of prediction
    correlation = np.zeros(prediction.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(prediction.shape[1])):
        correlation[v] = corr(prediction[:, v], truth[:, v])[0]

    return correlation

def add_data_to_json(json_data, layer_name, subject, lh_median, rh_median, lh_median_roi_correlation, rh_median_roi_correlation):
    if not json_data:
        json_data = {
            "Layers": []
        }

    layer_found = False
    for layer in json_data["Layers"]:
        if layer["layer_name"] == layer_name:
            layer_found = True
            
            subject_found = False
            for subject_data in layer["Subjects"]:
                if subject_data["subject"] == subject:
                    subject_found = True
                    subject_data["LH_median_correlation"] = lh_median
                    subject_data["RH_median_correlation"] = rh_median
                    subject_data["LH_median_roi_correlation"] = lh_median_roi_correlation
                    subject_data["RH_median_roi_correlation"] = rh_median_roi_correlation
                    break
            
            if not subject_found:
                subject_data = {
                    "subject": subject,
                    "LH_median_correlation": lh_median,
                    "RH_median_correlation": rh_median,
                    "LH_median_roi_correlation": lh_median_roi_correlation,
                    "RH_median_roi_correlation": rh_median_roi_correlation
                }
                layer["Subjects"].append(subject_data)
            
            break

    if not layer_found:
        new_layer = {
            "layer_name": layer_name,
            "Subjects": [
                {
                    "subject": subject,
                    "LH_correlation": lh_median,
                    "RH_correlation": rh_median,
                    "LH_median_roi_correlation": lh_median_roi_correlation,
                    "RH_median_roi_correlation": rh_median_roi_correlation
                }
            ]
        }
        json_data["Layers"].append(new_layer)

    return json_data

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




def plot_and_write_correlations(nsd_dataset, lh_correlation, rh_correlation, output_dir, layer_name, subj):
    plot_filename = f'{nsd_dataset.var_plot_dir}/correlations.png'

    # Print and write median results
    layer_and_subject_str = f'Layer: {layer_name} Subject: {subj}\n'
    lh_median = np.median(lh_correlation)
    rh_median = np.median(rh_correlation)
    results_str = f'LH Correlation: {lh_median} RH Correlation: {rh_median}\n'

    print(layer_and_subject_str)
    print(results_str)

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
                         'mapping_floc-faces.npy', 'mapping_floc-places.npy',
                         'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(nsd_dataset.data_dir, 'roi_masks', r),
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
        lh_challenge_rois.append(np.load(os.path.join(nsd_dataset.data_dir, 'roi_masks',
                                                      lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(nsd_dataset.data_dir, 'roi_masks',
                                                      rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0:  # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    def get_median_correlatoins_per_roi(roi_correlation, roi_names):
        median_roi_correlations = {}
        for i in range(len(roi_names)):
           median_corr = np.median(roi_correlation[i])
           if np.isnan(median_corr):
               median_roi_correlations[roi_names[i]] = None
           else:
              median_roi_correlations[roi_names[i]] = median_corr
        return median_roi_correlations
    
    lh_median_roi_correlation = get_median_correlatoins_per_roi(lh_roi_correlation, roi_names)
    rh_median_roi_correlation = get_median_correlatoins_per_roi(rh_roi_correlation, roi_names) 
    # Print and write ROI correlations
    print(f'LH median roi correlation: \n{lh_median_roi_correlation}\n')
    print(f'RH median roi correlation: \n{rh_median_roi_correlation}\n')
    # Write results to json file
    json_filename = f'{output_dir}/results.json'
    json_data = read_json_file(json_filename)
    amended_json_data = add_data_to_json(json_data, layer_name, subj, lh_median, rh_median, lh_median_roi_correlation, rh_median_roi_correlation)
    print(amended_json_data)
    print(json.dumps(amended_json_data, indent=2))
    write_json_file(json_filename, amended_json_data)


    plottable_values_lh = [0 if value is None else value for value in list(lh_median_roi_correlation.values())]
    plottable_values_rh = [0 if value is None else value for value in list(rh_median_roi_correlation.values())]
    plottable_keys = list(lh_median_roi_correlation.keys())

    # Plot ROI correlations
    plt.figure(figsize=(18, 6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width / 2, plottable_values_lh, width, label='Left Hemisphere')
    plt.bar(x + width / 2, plottable_values_rh, width,
            label='Right Hemishpere')
    plt.xlim(left=min(x) - .5, right=max(x) + .5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=plottable_keys, rotation=60)
    plt.ylabel('Median Pearson\'s $r$')
    plt.legend(frameon=True, loc=1)
    plt.savefig(plot_filename)
    plt.show()
