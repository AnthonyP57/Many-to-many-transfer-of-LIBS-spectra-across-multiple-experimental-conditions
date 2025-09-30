import pycaltransfer.caltransfer as caltransfer
import h5py
import numpy as np
from modules import SpectraDataset, ClassDataset, PDS_test_spectra, train_test_spectra_idx, train_test_spectra_samples, PDSDataset
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib
from torch.utils.data import  DataLoader
from CNN_conc_baseline import CNN
import json
from tqdm import tqdm
from sklearn.metrics import r2_score
import random
from data_visualization import xy_plot


with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]
    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    earth_cond = np.where(conditions[:, 2] == 1)[0] # vac50: 0 , vac100: 1, earth50: 2, earth100: 3
    mars_cond = np.where(conditions[:, 1] == 1)[0]

    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx + 1])
    mars_spectra = all_spectra[mars_cond]

    mars_labels = np.array(hf['labels_one_hot'][mars_cond])

    mars_spectra_ = []
    for i in range(mars_labels.shape[1]):
        all_idx = np.where(mars_labels[:, i] == 1)[0]
        random_10 = random.sample(list(all_idx), 20)
        mars_spectra_.append(mars_spectra[random_10])

    mars_spectra = np.array(mars_spectra_).reshape(mars_labels.shape[1] * 20, -1)

    earth_spectra = all_spectra[earth_cond]
    earth_labels = np.array(hf['labels_one_hot'][earth_cond])

    earth_spectra_ = []
    for i in range(earth_labels.shape[1]):
        all_idx = np.where(earth_labels[:, i] == 1)[0]
        random_10 = random.sample(list(all_idx), 20)
        earth_spectra_.append(earth_spectra[random_10])

    earth_spectra = np.array(earth_spectra_).reshape(earth_labels.shape[1] * 20, -1)

std = StandardScaler().fit(all_spectra)
mars_spectra = std.transform(mars_spectra)
earth_spectra = std.transform(earth_spectra)

segment_size = 20

ref_spectra = earth_spectra
target_spectra = mars_spectra
test_spectra = earth_spectra

F, a = caltransfer.pds_pls_transfer_fit(target_spectra, ref_spectra, max_ncp = 10, ww = 10)
standardized_spectra = test_spectra.dot(F) + a

standardized_spectra = std.inverse_transform(standardized_spectra)
earth_spectra = std.inverse_transform(earth_spectra)
mars_spectra = std.inverse_transform(mars_spectra)

xy_plot([wavelen[spectra_0_idx:spectra_last_idx+1] for i in range(3)], [standardized_spectra[250], earth_spectra[250], mars_spectra[250]], labels=['transformed', 'source', 'target'])
