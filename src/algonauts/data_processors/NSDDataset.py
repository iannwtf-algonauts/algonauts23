import os
import numpy as np
from sklearn.model_selection import train_test_split

submission_dir = 'algonauts_2023_challenge_submission'
var_plot_dir = 'variance_graphs'


class NSDDataset:
    def __init__(self, data_dir, output_dir, subj):
        self.subj = format(subj, '02')

        # Configure data folders
        self.data_dir = os.path.join(data_dir, 'subj' + self.subj)
        self.train_img_dir = os.path.join(self.data_dir, 'training_split', 'training_images')
        self.test_img_dir = os.path.join(self.data_dir, 'test_split', 'test_images')
        # Configure output folders
        self.subject_submission_dir = os.path.join(output_dir, submission_dir, 'subj' + self.subj)
        self.var_plot_dir = os.path.join(output_dir, var_plot_dir, 'subj' + self.subj)
        # Get training and test image lists
        self.training_img_list, self.test_img_list = self.get_training_and_test_image_lists()

        # Create the output directories if not existing
        if not os.path.isdir(self.subject_submission_dir):
            os.makedirs(self.subject_submission_dir)
        if not os.path.isdir(self.var_plot_dir):
            os.makedirs(self.var_plot_dir)

        # Get fMRI data
        lh_fmri, rh_fmri = self.get_full_fmri_data()
        # Get randomized partition indices
        self.idxs_train, self.idxs_val, self.idxs_test = self.get_random_partition_indices(random_seed=5)
        # Get train and validation fMRI data with randomized indices
        self.lh_fmri_train = lh_fmri[self.idxs_train]
        self.lh_fmri_val = lh_fmri[self.idxs_val]
        self.rh_fmri_train = rh_fmri[self.idxs_train]
        self.rh_fmri_val = rh_fmri[self.idxs_val]

        del lh_fmri, rh_fmri

    def get_training_and_test_image_lists(self):
        """
        Get list of filenames for training and test images
        :return: two lists, for training and test images, respectively
        """
        # Create lists will all training and test image file names, sorted
        training_img_list = os.listdir(self.train_img_dir)
        training_img_list.sort()
        test_img_list = os.listdir(self.test_img_dir)
        test_img_list.sort()
        print('Training images: ' + str(len(training_img_list)))
        print('Test images: ' + str(len(test_img_list)))

        return training_img_list, test_img_list

    def get_full_fmri_data(self):
        """
        Get fMRI data as numpy arrays for one subject
        :return: Tuple of left and right hemisphere fMRI data for the subject (lh_fmri, rh_fmri)
        """
        fmri_dir = os.path.join(self.data_dir, 'training_split', 'training_fmri')
        lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
        rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

        print('LH training fMRI data shape:')
        print(lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')

        print('\nRH training fMRI data shape:')
        print(rh_fmri.shape)
        print('(Training stimulus images × RH vertices)')

        return lh_fmri, rh_fmri

    def get_random_partition_indices(self, random_seed):
        """
        Get indices for training, validation and test datasets.
        90% of the data will be training and 10% for validation.
        Test indices are not changed.
        :param random_seed: seed for the randomizer
        :return: training, validation and test indices, respectively
        """
        training_img_list, test_img_list = self.get_training_and_test_image_lists()
        idxs_test = np.arange(len(test_img_list))

        idxs = np.arange(len(training_img_list))
        idxs_train, idxs_val = train_test_split(idxs, test_size=0.1, random_state=random_seed)

        print('Training stimulus images: ' + format(len(idxs_train)))
        print('\nValidation stimulus images: ' + format(len(idxs_val)))
        print('\nTest stimulus images: ' + format(len(idxs_test)))

        return idxs_train, idxs_val, idxs_test
