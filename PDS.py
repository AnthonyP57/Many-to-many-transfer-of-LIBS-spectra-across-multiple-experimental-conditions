import torch
import torch.nn as nn

class PDSModel(nn.Module):
    def __init__(self):
        super(PDSModel, self).__init__()

    def forward(self, ref_segment, target_segment, test_segment, device):
        ref_t = torch.as_tensor(ref_segment, dtype=torch.float32, device=device)
        target_t = torch.as_tensor(target_segment, dtype=torch.float32, device=device)
        test_t = torch.as_tensor(test_segment, dtype=torch.float32, device=device)

        # Compute the least squares solution (X'X)^-1 * X'y
        # (ref_t.T @ ref_t) is the covariance matrix of the reference data
        XTX = ref_t.T @ ref_t

        # Compute the pseudo-inverse to avoid singular matrix errors
        XTX_inv = torch.linalg.pinv(XTX)

        # Calculate the weights (coefficients) for linear regression: (X'X)^-1 * X' * y
        weights = XTX_inv @ ref_t.T @ target_t

        # Apply the weights to the test segment to standardize it
        standardized_segment = test_t @ weights

        # Return the standardized segment and the weights (both moved to CPU)
        return standardized_segment.cpu(), weights.cpu().numpy()

def piecewise_direct_standardization_pytorch(ref_spectra, target_spectra, test_spectra, segment_size):
    # Choose the correct device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    n_samples, n_wavelengths = ref_spectra.shape
    standardized_spectra = np.zeros_like(test_spectra)
    model = PDSModel().to(device)

    # Iterate through the spectra in segments
    for i in range(0, n_wavelengths, segment_size):
        start = i
        end = min(i + segment_size, n_wavelengths)

        ref_segment = ref_spectra[:, start:end]
        target_segment = target_spectra[:, start:end]
        test_segment = test_spectra[:, start:end]

        standardized_segment, weights = model(ref_segment, target_segment, test_segment, device)

        standardized_spectra[:, start:end] = standardized_segment.detach().numpy()

    return standardized_spectra, weights

if __name__ == '__main__':
    import numpy as np
    import h5py
    import random
    from sklearn.preprocessing import StandardScaler
    # from modules import train_test_spectra_samples
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    import matplotlib.ticker as ticker

    test_idx = 7

    with h5py.File('/home/anthonyp57/VSCode_projects/Spectra_transfer/git_code/ACVAE/new_data/spectra.h5', 'r') as hf:
        wavelen = np.array(hf['calibration'][()])
        spectra_0_idx = np.where(wavelen == 250)[0][0]
        spectra_last_idx = np.where(wavelen == 850)[0][0]
        all_atm = np.array(hf['atm_one_hot'][()])
        all_ene = np.array(hf['energy_one_hot'][()])
        # labels = np.array(hf['labels_one_hot'])

        conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])

        # train_idx, test_idx = train_test_spectra_samples(conditions, labels) # divide within samples -> test samples idx: [1, 7, 17, 40]

        test_samples = [1, 7, 17, 40]

        earth_cond = np.where(conditions[:, 2] == 1)[0] # vac50: 0 , vac100: 1, earth50: 2, earth100: 3
        mars_cond = np.where(conditions[:, 1] == 1)[0] # target

        all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx + 1])
        mars_spectra = all_spectra[mars_cond]

        mars_labels = np.array(hf['labels_one_hot'][mars_cond])

        mars_spectra_ = []
        for i in range(mars_labels.shape[1]):
            if i not in test_samples:
                all_idx = np.where(mars_labels[:, i] == 1)[0]
                # random_10 = random.sample(list(all_idx), 10)
                mars_spectra_.append(mars_spectra[all_idx])
            else:
                if i == test_idx:
                    all_idx = np.where(mars_labels[:, i] == 1)[0]
                    test_mars = mars_spectra[all_idx]

        mars_spectra = np.array(mars_spectra_).reshape((mars_labels.shape[1] - len(test_samples)) * 100, -1)

        earth_spectra = all_spectra[earth_cond]
        earth_labels = np.array(hf['labels_one_hot'][earth_cond])

        earth_spectra_ = []
        for i in range(earth_labels.shape[1]):
            if i not in test_samples:
                all_idx = np.where(earth_labels[:, i] == 1)[0]
                # random_10 = random.sample(list(all_idx), 10)
                earth_spectra_.append(earth_spectra[all_idx])
            else:
                if i == test_idx:
                    all_idx = np.where(earth_labels[:, i] == 1)[0]
                    test_earth = earth_spectra[all_idx]

        earth_spectra = np.array(earth_spectra_).reshape((earth_labels.shape[1]- len(test_samples)) * 100, -1)

    std = StandardScaler().fit(all_spectra)
    mars_spectra = std.transform(mars_spectra)
    earth_spectra = std.transform(earth_spectra)
    test_earth = std.transform(test_earth)
    test_mars = std.transform(test_mars)

    segment_size = 20

    ref_spectra = earth_spectra
    target_spectra = mars_spectra
    test_spectra = test_earth

    standardized_spectra, weights = piecewise_direct_standardization_pytorch(ref_spectra, target_spectra, test_spectra, segment_size)
    predicted = std.inverse_transform(standardized_spectra)
    source = std.inverse_transform(test_earth)
    target = std.inverse_transform(test_mars)

    x = wavelen[spectra_0_idx:spectra_last_idx+1]

    source = source * (source > 0)
    source_mean = np.mean(source, axis=0)
    source_std = np.std(source, axis=0)

    source_upper = source_mean + source_std
    source_lower = source_mean - source_std

    original = target
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
    # ax.plot(x, source_mean, 'g-', label='source', linewidth=1)

    ax.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
    ax.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
    # ax.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("intensity (—)")
    ax.set_xlim(250, 850)
    ax.set_ylim(-1000, max(max(org_upper), max(pred_upper))*1.1)
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
    # axins1.plot(x, source_mean, 'g-', linewidth=1)

    axins1.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
    axins1.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
    # axins1.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

    axins1.set_xlim(392, 398)
    axins1.set_ylim(0, 90_000)
    axins1.set_title("maximal error", fontsize=9)
    axins1.tick_params(labelsize=7)
    axins1.yaxis.set_major_formatter(formatter)
    mark_inset(ax, axins1, loc1=1, loc2=3, fc="none", ec="gray", lw=0.7)

    # Median error region
    axins2 = inset_axes(
        ax, width="25%", height="30%",
        bbox_to_anchor=(0.7, 0.6, 1, 1),
        bbox_transform=ax.transAxes, loc='lower left'
    )
    axins2.plot(x, original_mean, 'r-', linewidth=1)
    axins2.plot(x, predicted_mean, 'b-', linewidth=1)
    # axins2.plot(x, source_mean, 'g-', linewidth=1)

    axins2.fill_between(x, org_lower, org_upper, color='violet', alpha=0.4)
    axins2.fill_between(x, pred_lower, pred_upper, color='cyan', alpha=0.3)
    # axins2.fill_between(x, source_lower, source_upper, color='seagreen', alpha=0.3)

    axins2.set_xlim(630, 670)
    axins2.set_ylim(0, 10_000)
    axins2.set_title("median error", fontsize=9)
    axins2.tick_params(labelsize=7)
    axins2.yaxis.set_major_formatter(formatter)
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="gray", lw=0.7)

    # plt.tight_layout()
    plt.savefig("./git_code/ACVAE/spectra_e50_v100_7_pds.png", dpi=1000)
    plt.show()

