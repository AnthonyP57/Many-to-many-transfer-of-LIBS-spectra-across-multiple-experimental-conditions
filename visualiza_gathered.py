import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker
import difflib

# with open('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/std_scaler.pkl', 'rb') as fp:
#     scaler = pickle.load(fp)

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/metrics/42-4.h5', 'r') as hf:
    spectra = np.array(hf['data'])
    labels = np.array(hf['labels'])

# print(labels, spectra.shape)
# print(set(labels))
# print(labels[2002])
# print(np.where(np.array(labels) == b'e100_vac100-1')[0])

sample = b'e50_vac100-7' # 1,7,17,40

if sample.decode('utf-8') not in [i.decode('utf-8') for i in set(labels)]:
    closest_matches = difflib.get_close_matches(sample.decode('utf-8'), [i.decode('utf-8') for i in set(labels)], n=1, cutoff=0)
    if closest_matches:
        print(f"{sample.decode('utf-8')} is not in the list. The most similar element is: {closest_matches[0]}")
    sample = closest_matches[0].encode('utf-8')

# spectra = scaler.inverse_transform(np.concat([np.zeros((spectra.shape[0], 25)), spectra, np.zeros((spectra.shape[0], 25))], axis=1))[:, 25:-25]
predicted = spectra[np.where(labels == sample)[0]]
x = np.linspace(250, 850, spectra.shape[1])

with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    # all_spectra = apply_multiprocessing_along_axis(correct_baseline, all_spectra) #baseline correction

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]] # vac100, e50, e100

condition_dict = {'vac100': 0, 'e50': 1, 'e100': 2}

org_sample_atm = np.where(np.argmax(conditions, axis=1) == condition_dict[sample.decode('utf-8').split('-')[0].split('_')[1]])[0]
org_sample_label = np.where(np.argmax(labels, axis=1) == int(sample.decode('utf-8').split('-')[-1]))[0]
idx = np.intersect1d(org_sample_atm, org_sample_label)

# original = spectra[2001:2003]
# predicted = spectra[2002:2004]

source_sample_atm = np.where(np.argmax(conditions, axis=1) == condition_dict[sample.decode('utf-8').split('-')[0].split('_')[0]])[0]

source_idx = np.intersect1d(source_sample_atm, org_sample_label)

source = all_spectra[source_idx]
source = source * (source > 0)
source_mean = np.mean(source, axis=0)
source_std = np.std(source, axis=0)

source_upper = source_mean + source_std
source_lower = source_mean - source_std

original = all_spectra[idx]
original = original * (original > 0) # relu
predicted = predicted * (predicted > 0)

original_mean = np.mean(original, axis=0)
org_std = np.std(original, axis=0)
org_lower = np.maximum(original_mean - org_std, 0)
org_upper = original_mean + org_std

predicted_mean = np.mean(predicted, axis=0)
pred_std = np.std(predicted, axis=0)
pred_lower = np.maximum(predicted_mean - pred_std, 0)
pred_upper = predicted_mean + pred_std

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, original_mean, 'r-', label='target', linewidth=1)
ax.plot(x, predicted_mean, 'b-', label='predicted', linewidth=1)
ax.plot(x, source_mean, 'g-', label='source', linewidth=1)

ax.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
ax.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
ax.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("intensity (—)")
ax.set_xlim(250, 850)
ax.set_ylim(-1000, max(max(org_upper), max(pred_upper), max(source_upper))*1.1)
ax.legend(loc='upper left', frameon=False)

formatter = ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", "'"))
ax.yaxis.set_major_formatter(formatter)

# Maximal error region
axins1 = inset_axes(
    ax, width="25%", height="30%",
    bbox_to_anchor=(0.35, 0.6, 1, 1),
    bbox_transform=ax.transAxes, loc='lower left'
)
axins1.plot(x, original_mean, 'r-', linewidth=1)
axins1.plot(x, predicted_mean, 'b-', linewidth=1)
axins1.plot(x, source_mean, 'g-', linewidth=1)

axins1.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
axins1.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
axins1.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

axins1.set_xlim(392, 398)
axins1.set_ylim(0, 40_000)
axins1.set_title("maximal error", fontsize=9)
axins1.tick_params(labelsize=7)
axins1.yaxis.set_major_formatter(formatter)
mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="gray", lw=0.7)

# Median error region
axins2 = inset_axes(
    ax, width="25%", height="30%",
    bbox_to_anchor=(0.7, 0.6, 1, 1),
    bbox_transform=ax.transAxes, loc='lower left'
)
axins2.plot(x, original_mean, 'r-', linewidth=1)
axins2.plot(x, predicted_mean, 'b-', linewidth=1)
axins2.plot(x, source_mean, 'g-', linewidth=1)

axins2.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
axins2.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
axins2.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

axins2.set_xlim(630, 670)
axins2.set_ylim(0, 10_000)
axins2.set_title("median error", fontsize=9)
axins2.tick_params(labelsize=7)
axins2.yaxis.set_major_formatter(formatter)
mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="gray", lw=0.7)

# plt.tight_layout()
# plt.savefig("git_code/ACVAE/spectra_e50_v100_7_acvae.png", dpi=1000)
plt.show()
