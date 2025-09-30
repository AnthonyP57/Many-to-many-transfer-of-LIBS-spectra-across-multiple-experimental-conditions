import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from data_visualization import xy_plot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data
with h5py.File('new_data\\spectra.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]

    labels = np.array(hf['labels_one_hot'])
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])

    wavelen = wavelen[spectra_0_idx:spectra_last_idx+1]

    all_atm = np.array(hf['atm_one_hot'][()])
    all_ene = np.array(hf['energy_one_hot'][()])

    conditions = np.array([np.kron(row1, row2) for row1, row2 in zip(all_atm, all_ene)])
    # c_ = np.where(np.argmax(conditions, axis=1) == 1)[0]
    # c__ = np.where(np.argmax(conditions, axis=1) == 3)[0]
    # c_ = np.concatenate([c_, c__])
    # all_spectra = all_spectra[c_]
    # conditions = conditions[c_][:,[1,3]]

    c = np.where(np.argmax(conditions, axis=1) != 0)[0] #no vacuum 50%
    all_spectra = all_spectra[c]
    labels = labels[c]
    conditions = conditions[c][:, [1,2,3]]

    c = np.where(np.argmax(conditions, axis=1) == 2)[0]
    all_spectra = all_spectra[c]

n_components = 15  # Choose the number of principal components
pca = PCA(n_components=n_components)
pca.fit(all_spectra)

all_spectra = pca.transform(all_spectra)

# Define the custom range for k (clusters)
min_k = 2
max_k = 11

k_values = range(min_k, max_k + 1)
sil_scores = []
inertia_values = []

# Prepare for subplot layout
n_plots = len(k_values)
n_cols = 5  # Number of columns in the plot grid
n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate number of rows required

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
fig.suptitle('Silhouette plots for various k after PCA', fontsize=16)

# Iterate through each k value and generate the silhouette plot
for idx, k in enumerate(k_values):
    ax = axes[idx // n_cols, idx % n_cols]  # Determine subplot position

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(all_spectra)

    # Overall silhouette score
    score = silhouette_score(all_spectra, labels)
    sil_scores.append(score)

    # Inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

    # Silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(all_spectra, labels)

    # Plot silhouette scores for each cluster
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for spacing between plots

    ax.axvline(x=score, color="red", linestyle="--")
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(all_spectra) + (k + 1) * 10])
    ax.set_title(f'k={k}')
    ax.set_yticks([])
    ax.set_xticks([])

# Adjust spacing between subplots
for i in range(n_plots, n_rows * n_cols):
    fig.delaxes(axes.flat[i])

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

inertia_values = [i / max(inertia_values) for i in inertia_values]

# Plot overall silhouette score and inertia vs. number of clusters (k) using xy_plot
xy_plot(
    [[i for i in k_values], [i for i in k_values]],
    [sil_scores, inertia_values],
    xy_labels=('k', 'Score'),
    title='k vs. silhouette score and inertia after PCA',
    labels=['silhouette score', 'inertia']
)

input_k = int(input("Enter k value: "))

kmeans = KMeans(n_clusters=input_k, random_state=42)
labels_ = kmeans.fit_predict(all_spectra)
labels_ = labels_.reshape(-1)

# Generate sample data
X = all_spectra[:, :3]  # Assuming you have 3 features

# Create subplot for original data
fig = plt.figure(figsize=(14, 14))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
for i in np.unique(labels):
    ax1.scatter(X[labels_==i, 0], X[labels_==i, 1], X[labels_==i, 2], label=f"Cluster {i}")
ax1.set_title("Original Data")
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_zlabel("PC 3")
plt.savefig('pca_scores.png')
# plt.show()


all_spectra = pca.inverse_transform(all_spectra)

cluster_means=[np.mean(all_spectra[labels_==i],axis=0) for i in range(input_k)]

fig, axs = plt.subplots(input_k, 1, figsize=(32, 24))

for i in range(input_k):
    ax = axs[i]
    _spec = all_spectra[labels_==i]
    # _spec = _spec - cluster_means[i]
    for s in _spec:
        ax.plot(wavelen, s, color='orange', alpha=0.1, label='spectra')

    ax.plot(wavelen, cluster_means[i], color='red', label='cluster mean')
    ax.set_title(f'Cluster {i+1}', fontsize=8)

# plt.savefig('spectra_diffs.png')