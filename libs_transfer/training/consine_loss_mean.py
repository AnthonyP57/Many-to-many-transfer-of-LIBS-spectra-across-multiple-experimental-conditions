import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import spatial

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

    all_spectra = np.maximum(all_spectra, 0)  # relu

    all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)  # total emissivity normalization
    emis_std = StandardScaler().fit(all_spectra)

    all_spectra = emis_std.transform(all_spectra)

    # test_labels = [1, 7, 17, 40]

    # test_labels_idx = np.where(np.isin(np.argmax(labels, axis=1), test_labels))[0]

    v100_idx = np.where(np.argmax(conditions, axis=1) == 0)[0]
    v100_1 = [i for i in np.where(np.argmax(labels, axis=1) == 1)[0] if i in v100_idx]
    v100_7 = [i for i in np.where(np.argmax(labels, axis=1) == 7)[0] if i in v100_idx]
    v100_17 = [i for i in np.where(np.argmax(labels, axis=1) == 17)[0] if i in v100_idx]
    v100_40 = [i for i in np.where(np.argmax(labels, axis=1) == 40)[0] if i in v100_idx]

    e50_idx = np.where(np.argmax(conditions, axis=1) == 1)[0]
    e50_1 = [i for i in np.where(np.argmax(labels, axis=1) == 1)[0] if i in e50_idx]
    e50_7 = [i for i in np.where(np.argmax(labels, axis=1) == 7)[0] if i in e50_idx]
    e50_17 = [i for i in np.where(np.argmax(labels, axis=1) == 17)[0] if i in e50_idx]
    e50_40 = [i for i in np.where(np.argmax(labels, axis=1) == 40)[0] if i in e50_idx]
    
    e100_idx = np.where(np.argmax(conditions, axis=1) == 2)[0]
    e100_1 = [i for i in np.where(np.argmax(labels, axis=1) == 1)[0] if i in e100_idx]
    e100_7 = [i for i in np.where(np.argmax(labels, axis=1) == 7)[0] if i in e100_idx]
    e100_17 = [i for i in np.where(np.argmax(labels, axis=1) == 17)[0] if i in e100_idx]
    e100_40 = [i for i in np.where(np.argmax(labels, axis=1) == 40)[0] if i in e100_idx]

    assert np.array_equal(labels[v100_idx], labels[e50_idx]) and np.array_equal(labels[v100_idx], labels[e100_idx]), "labels are not equal"

    v100_1 = np.mean(all_spectra[v100_1], axis=0)
    v100_7 = np.mean(all_spectra[v100_7], axis=0)
    v100_17 = np.mean(all_spectra[v100_17], axis=0)
    v100_40 = np.mean(all_spectra[v100_40], axis=0)

    e50_1 = np.mean(all_spectra[e50_1], axis=0)
    e50_7 = np.mean(all_spectra[e50_7], axis=0)
    e50_17 = np.mean(all_spectra[e50_17], axis=0)
    e50_40 = np.mean(all_spectra[e50_40], axis=0)

    e100_1 = np.mean(all_spectra[e100_1], axis=0)
    e100_7 = np.mean(all_spectra[e100_7], axis=0)
    e100_17 = np.mean(all_spectra[e100_17], axis=0)
    e100_40 = np.mean(all_spectra[e100_40], axis=0)

    all_spectra = np.vstack([
        v100_1,
        v100_7,
        v100_17,
        v100_40,

        v100_1,
        v100_7,
        v100_17,
        v100_40,

        e50_1,
        e50_7,
        e50_17,
        e50_40,

        e50_1,
        e50_7,
        e50_17,
        e50_40,

        e100_1,
        e100_7,
        e100_17,
        e100_40,

        e100_1,
        e100_7,
        e100_17,
        e100_40
    ])


pca = PCA(n_components=15)
pca.fit(all_spectra)
all_spectra = pca.transform(all_spectra)

def cosine_similarity_mean_spectra(pred):
    pred = pca.transform(pred)
    cos_sim=[]
    for p, a in zip(pred, all_spectra[:len(pred)]):
        cos_sim.append(1 - spatial.distance.cosine(p, a))
    return np.mean(cos_sim)
