import h5py
import numpy as np
import os
from spectra_normalization import to_onehot
import json

main_folder = 'C:\\data\\ceitec usb backup\\LIBS_karol'
atmospheres = os.listdir(main_folder)

data_np = []
atm_=[]
label_=[]
perc_=[]

for i, a in enumerate(atmospheres):
    perc = os.listdir(os.path.join(main_folder, a))
    for j, p in enumerate(perc):
        samples = os.listdir(os.path.join(main_folder, a, p))
        for k, s in enumerate(samples):
            sample_path = os.path.join(main_folder, a, p, s)
            data = np.loadtxt(sample_path, dtype=np.float32)
            calibration =  data[:, 0].flatten()
            data = data[:, 1:].transpose()
            n_samples = data.shape[0]
            data_np.append(data)
            atm_.append([a.upper()]*n_samples)
            label_.append([s.upper().split('.')[:-1]]*n_samples)
            perc_.append([p.upper()]*n_samples)


data_np = np.concatenate(data_np, axis=0)

atm_one_hot, atm_dict = to_onehot(np.array(atm_))
label_one_hot, label_dict = to_onehot(np.array(label_))
perc_one_hot, perc_dict = to_onehot(np.array(perc_))

with open('new_data\\atm_dict.json', 'w') as f:
    json.dump(atm_dict, f)

with open('new_data\\label_dict.json', 'w') as f:
    json.dump(label_dict, f)

with open('new_data\\perc_dict.json', 'w') as f:
    json.dump(perc_dict, f)

with h5py.File('new_data\\spectra.h5', 'w') as hdf:
    hdf.create_dataset('spectra', data=data_np)
    hdf.create_dataset('calibration', data=calibration)

    hdf.create_dataset('atmosphere', data=np.array(atm_).flatten().tolist())
    hdf.create_dataset('atm_one_hot', data=atm_one_hot)

    hdf.create_dataset('labels', data=np.array(label_).flatten().tolist())
    hdf.create_dataset('labels_one_hot', data=label_one_hot)

    hdf.create_dataset('energy', data=np.array(perc_).flatten().tolist())
    hdf.create_dataset('energy_one_hot', data=perc_one_hot)