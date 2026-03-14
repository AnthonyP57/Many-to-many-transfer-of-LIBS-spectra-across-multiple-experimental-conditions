import h5py
import numpy as np
import os
from libs_transfer.prepare_data.spectra_normalization import to_onehot
import json

def data_to_h5(in_path, out_folder='./'):
    """
    Transforms the data into a h5 file. The data is the default data format obtained from the LIBS Discovery system.

    e.g. (tab separated)

    ```text
    [nm]      [a.u.]
    200.000 -72.060547
    200.020 107.521484
    ...
    ```

    Args:
        in_path (str): path to the data
        out_folder (str): path to the output folder

    Returns:
        None    
    """
    os.makedirs(out_folder, exist_ok=True)
    atmospheres = os.listdir(in_path)

    data_np = []
    atm_=[]
    label_=[]
    perc_=[]

    for i, a in enumerate(atmospheres):
        perc = os.listdir(os.path.join(in_path, a))
        for j, p in enumerate(perc):
            samples = os.listdir(os.path.join(in_path, a, p))
            for k, s in enumerate(samples):
                sample_path = os.path.join(in_path, a, p, s)
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

    with open(f'{out_folder}/atm_dict.json', 'w') as f:
        json.dump(atm_dict, f)

    with open(f'{out_folder}/label_dict.json', 'w') as f:
        json.dump(label_dict, f)

    with open(f'{out_folder}/perc_dict.json', 'w') as f:
        json.dump(perc_dict, f)

    with h5py.File(f'{out_folder}/spectra.h5', 'w') as hdf:
        hdf.create_dataset('spectra', data=data_np)
        hdf.create_dataset('calibration', data=calibration)

        hdf.create_dataset('atmosphere', data=np.array(atm_).flatten().tolist())
        hdf.create_dataset('atm_one_hot', data=atm_one_hot)

        hdf.create_dataset('labels', data=np.array(label_).flatten().tolist())
        hdf.create_dataset('labels_one_hot', data=label_one_hot)

        hdf.create_dataset('energy', data=np.array(perc_).flatten().tolist())
        hdf.create_dataset('energy_one_hot', data=perc_one_hot)
