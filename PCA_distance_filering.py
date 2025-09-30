import numpy as np
import h5py
from sklearn.decomposition import PCA
from data_visualization import scatter_3d_plot

# with h5py.File('Regoliths.h5', 'r') as hf:

#     all_labels = np.array(hf['labels'])

#     spectra = np.array(hf['spectra'][()])
#     all_conditions = np.array(hf['conditions'][()])
#     wavelen = np.array(hf['calibration'][()])

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0] - 25
    spectra_last_idx = np.where(wavelen == 850)[0][0] + 25

    labels = np.array(hf['labels_one_hot'])
    spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    # all_spectra = apply_multiprocessing_along_axis(correct_baseline, all_spectra) #baseline correction

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    spectra = spectra[c]
    all_labels = labels[c]
    all_conditions = conditions[c][:, [1,2,3]]

labels = all_labels.shape[1]
conditions = all_conditions.shape[1]

pca_inliers_list = []
inlier_labels_list = []
inlier_conditions_list = []

for i in range(conditions):
    condition_idx = list(np.where(all_conditions[:, i] == 1)[0])

    for j in range(labels):
        labels_idx = list(np.where(all_labels[:, j] == 1)[0])

        common_idx = list(set(list(filter(lambda x: x in condition_idx, labels_idx))))

        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(spectra[common_idx, :])

        center = np.mean(pca_data, axis=0)

        dists = np.linalg.norm(pca_data - center, axis=1)

        # threshold = np.percentile(dists, 80)
        mean_dist = np.mean(dists)
        IQR = np.percentile(dists, 75) - np.percentile(dists, 25)
        iqr_thresh = mean_dist + 1.5 * IQR

        filtered_idx = list(filter(lambda x: dists[x] < iqr_thresh, list(range(len(dists))))) # and dists[x] < mean_dist

        pca_inliers = pca_data[filtered_idx, :]
        pca_outliers = pca_data[list(filter(lambda x: x not in filtered_idx, list(range(len(dists))))), :]

        # scatter_3d_plot(np.concatenate((pca_data[filtered_idx, :], pca_outliers), axis=0),
        #                 ['sample_'+str(j) for _ in range(len(filtered_idx))]+['samepl_'+str(j)+'_outliers' for _ in range(pca_outliers.shape[0])],
        #                 xyz_labels=['PCA 1', 'PCA 2', 'PCA 3'])

        labels_pca = all_labels[common_idx][filtered_idx]
        condition_idx_pca = all_conditions[common_idx][filtered_idx]
        spectra_pca = spectra[common_idx][filtered_idx]

        pca_inliers_list.append(spectra_pca)
        inlier_labels_list.append(labels_pca)
        inlier_conditions_list.append(condition_idx_pca)

pca_inliers_np = np.concatenate(pca_inliers_list, axis=0)
inlier_labels_np = np.concatenate(inlier_labels_list, axis=0)
inlier_conditions_np = np.concatenate(inlier_conditions_list, axis=0)

print(pca_inliers_np.shape, inlier_labels_np.shape, inlier_conditions_np.shape)

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/pca_inliers_spectra.h5', 'w') as hf:
    hf.create_dataset('spectra', data=pca_inliers_np)
    hf.create_dataset('labels', data=inlier_labels_np)
    hf.create_dataset('conditions', data=inlier_conditions_np)
    hf.create_dataset('calibration', data=wavelen)
