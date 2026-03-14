import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
import json
from preprocessing_LIBS import correct_baseline, apply_multiprocessing_along_axis

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    # spectra_0_idx = np.where(wavelen == 250)[0][0]
    # spectra_last_idx = np.where(wavelen == 850)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'])#[:, spectra_0_idx:spectra_last_idx+1])

    all_spectra = apply_multiprocessing_along_axis(correct_baseline, all_spectra)

    all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

wavelen_vals = {
"Al_wavelen" : [309.2, 309.35],
"Si_wavelen" : [288.1, 288.25],
"Ca_wavelen" : [393.1, 393.6],
"Fe_wavelen" : [438.25, 438.45],
"K_wavelen" : [769.7, 770.1],
"Mg_wavelen" : [285.1, 285.3],
"Mn_wavelen" : [402.4, 404], #not done
"P_wavelen" : [253, 254], #not done
"Ti_wavelen" : [375, 377], #not done
"Na_wavelen" : [330, 331], #not done
"Cr_wavelen" : [520, 521.5], #not done
"Ni_wavelen" : [361.8, 361.95],
}

with open('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/label_dict.json', 'r') as fp:
    reg_conditions_dict = json.load(fp)

conc_df = pd.read_excel(r'/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/Regoliths/Concentrations.xlsx')
conc_df = conc_df.dropna().transpose()
conc_df.columns = [a.upper() for a in conc_df.iloc[0].tolist()]
conc_df = conc_df.iloc[1:]
conc_df.index = [a.upper() for a in conc_df.index]
elements = conc_df.columns.tolist()

sample_id = range(labels.shape[1])
condition_id = range(conditions.shape[1])

integrals=[[] for _ in range(len(condition_id))]
concentrations = [[] for _ in range(len(condition_id))]
stds=[[] for _ in range(len(condition_id))]
conditions_=[]
samples_=[]

for c_i, c in enumerate(condition_id):
    c_idx = np.where(np.argmax(conditions, axis=1) == c)[0]

    for s in sample_id:
        s_idx = np.where(np.argmax(labels, axis=1) == s)[0]
        cs_idx = [i for i in c_idx if i in s_idx]
        fig, ax = plt.subplots(1,len(wavelen_vals), figsize=(3*len(wavelen_vals), 6), sharey=True)

        for i, (k, v) in enumerate(wavelen_vals.items()):

            idx = np.where((wavelen >= v[0]) & (wavelen <= v[1]))[0]
            _wavelen = wavelen[idx]
            _spectra = all_spectra[cs_idx][:, idx]

            _spectra_mean = np.mean(_spectra, axis=0, keepdims=True)
            _spectra_std = np.std(_spectra, axis=0, keepdims=True)

            _mean_interpol = InterpolatedUnivariateSpline(_wavelen, _spectra_mean, k=3)
            _mean_interpol_integral = _mean_interpol.integral(_wavelen[0], _wavelen[-1])
            _std_interpol = InterpolatedUnivariateSpline(_wavelen, _spectra_std, k=3)
            _std_interpol_integral = _std_interpol.integral(_wavelen[0], _wavelen[-1])

            ax[i].plot(_wavelen, _spectra_mean.reshape(-1), label='mean', color='orange')
            ax[i].fill_between(_wavelen, _spectra_mean.reshape(-1) - _spectra_std.reshape(-1), _spectra_mean.reshape(-1) + _spectra_std.reshape(-1), alpha=0.3, label='std', color='violet')
            ax[i].fill_between(_wavelen, 0, _spectra_mean.reshape(-1), alpha=0.2, label='integral', color='orange')
            ax[i].set_title(k)
            ax[i].text(0.1, 0.97, f"Integral: {_mean_interpol_integral:.2f}", transform=ax[i].transAxes, fontsize=8.5, ha='left', va='top')
            ax[i].set_xlim([wavelen[idx][0], wavelen[idx][-1]])

            integrals[c_i].append(_mean_interpol_integral)
            stds[c_i].append(_std_interpol_integral)

        conditions_.append(c)
        samples_.append(s)

        sample_label = reg_conditions_dict[str(s)]
        cond_conc = np.array([conc_df.loc[sample_label].to_list() if sample_label in conc_df.index else [0] * conc_df.shape[1]])
        concentrations[c_i].append(cond_conc)
    
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.5)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.savefig(f"/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/img/{c}-{s}.png")
        plt.close()

integrals = pd.DataFrame(np.array(integrals).reshape(-1, len(wavelen_vals)), columns=wavelen_vals.keys())
integrals['condition'] = conditions_
integrals['sample'] = samples_
integrals.to_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/integrals.csv', index=False)

stds = pd.DataFrame(np.array(stds).reshape(-1, len(wavelen_vals)), columns=wavelen_vals.keys())
stds['condition'] = conditions_
stds['sample'] = samples_
stds.to_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/stds.csv', index=False)

concentrations = pd.DataFrame(np.array(concentrations).reshape(-1, len(elements)), columns=elements)
concentrations['condition'] = conditions_
concentrations['sample'] = samples_
concentrations.to_csv('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/integrals/concentrations.csv', index=False)