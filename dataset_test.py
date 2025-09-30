from modules import SpectraDataset
import h5py
import numpy as np
import time

with h5py.File('/home/antonipa57/PycharmProjects/Spectra_transfer/spectra_transfer_code/CAVE_alike/PCA_inliers.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 800)[0][0]

    all_labels = np.array(hf['labels'])
    indices = list(np.where(all_labels[:, 13] == 1)[0])
    indices.extend(list(np.where(all_labels[:, 27] == 1)[0]))
    indices_ = list(filter(lambda x: x not in indices, list(range(all_labels.shape[0]))))
    print(f'{len(indices_)} spectra included')

    spectra = np.array(hf['spectra'][indices_, spectra_0_idx:spectra_last_idx+1])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])
    excluded_spectra = all_spectra[indices_, spectra_0_idx:spectra_last_idx+1]

    all_conditions = np.array(hf['conditions'][()])
    conditions = all_conditions[indices_]
    excluded_conditions = all_conditions[indices]

    labels = all_labels[indices_]
    excluded_labels = all_labels[indices]

ds = SpectraDataset(spectra, conditions, labels, batch_size=50)
ds.random_select_samples(shuffle_data_pairs=True)
k = 0
for i in ds:
    x_0, x_1, y_0, y_1 = i
    k += 1
    print(x_0.shape, x_1.shape, y_0.shape, y_1.shape,y_0[0], y_1[0])

print(k)

# for e in range(epochs):
#     for i in ds:
#         x_0, x_1, y_0, y_1 = i
#
#          ...
#
#         ds.random_select_samples()