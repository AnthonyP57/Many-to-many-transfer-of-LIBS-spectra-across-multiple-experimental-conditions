import h5py
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import matplotlib.ticker as ticker

sample = 7

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 380)[0][0]
    spectra_last_idx = np.where(wavelen == 450)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0]
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

condition_dict = {'vac100': 0, 'e50': 1, 'e100': 2}
x = np.linspace(380, 400, all_spectra.shape[1])

sample_label = np.where(np.argmax(labels, axis=1) == sample)[0]
vac_sample_atm = np.where(np.argmax(conditions, axis=1) == condition_dict['vac100'])[0]
e50_sample_atm = np.where(np.argmax(conditions, axis=1) == condition_dict['e50'])[0]
e100_sample_atm = np.where(np.argmax(conditions, axis=1) == condition_dict['e100'])[0]

vac100_idx = np.intersect1d(sample_label, vac_sample_atm)
e50_idx = np.intersect1d(sample_label, e50_sample_atm)
e100_idx = np.intersect1d(sample_label, e100_sample_atm)

vac_mean = all_spectra[vac100_idx]
vac_mean = np.maximum(vac_mean, 0)
vac_std = np.std(vac_mean, axis=0)
vac_mean = np.mean(vac_mean, axis=0)

vac_lower = np.maximum(vac_mean - vac_std, 0)
vac_upper = vac_mean + vac_std

e50_mean = all_spectra[e50_idx]
e50_mean = np.maximum(e50_mean, 0)
e50_std = np.std(e50_mean, axis=0)
e50_mean = np.mean(e50_mean, axis=0) + 5e4

e50_lower = np.maximum(e50_mean - e50_std, 5e4)
e50_upper = e50_mean + e50_std

e100_mean = all_spectra[e100_idx]
e100_mean = np.maximum(e100_mean, 0)
e100_std = np.std(e100_mean, axis=0)
e100_mean = np.mean(e100_mean, axis=0) + 2.5e5

e100_lower = np.maximum(e100_mean - e100_std, 2.5e5)
e100_upper = e100_mean + e100_std

vac_ylim = (-1e3, np.max(vac_upper)*1.1)
e50_ylim = (5e4, 5e4 + np.max(e50_upper - 5e4)*1.1)
e100_ylim = (2.5e5, 2.5e5 + np.max(e100_upper - 2.5e5)*1.1)

fig = plt.figure(figsize=(8, 6))
bax = brokenaxes(
    ylims=[vac_ylim, e50_ylim, e100_ylim],
    hspace=0.1,
    despine=False,  # keep spines so we can remove diagonal lines manually
    d=0.0           # disables the diagonal break marks
)

bax.plot(x, vac_mean, 'r-', label='vacuum', linewidth=1)
bax.fill_between(x, vac_lower, vac_upper, color='violet', alpha=0.5)

bax.plot(x, e50_mean, 'b-', label='Earth 50%', linewidth=1)
bax.fill_between(x, e50_lower, e50_upper, color='cyan', alpha=0.4)

bax.plot(x, e100_mean, 'g-', label='Earth 100%', linewidth=1)
bax.fill_between(x, e100_lower, e100_upper, color='seagreen', alpha=0.4)

bax.set_xlabel("wavelength (nm)", labelpad=20)

fig.text(0.04, 0.5, 'intensity (—)', va='center', rotation='vertical')

formatter = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "'"))
for ax in bax.axs:
    y0 = ax.get_ylim()[0]
    fmt = ticker.FuncFormatter(lambda x, pos, y0=y0: f"{x - y0:,.0f}".replace(",", "'"))
    ax.yaxis.set_major_formatter(fmt)

ax0 = bax.axs[2]
ax0.set_yticks([0, 2.5e4])
ax0.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", "'"))
)

bax.legend(loc='upper left', frameon=False)
bax.set_xlim(380, 400)
plt.tight_layout()
# plt.savefig("spectrum_atms.png", dpi=600)
plt.show()
