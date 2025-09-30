import random

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score
from data_visualization import xy_plot
import matplotlib.pyplot as plt
from match_clusters import map_labels

# Load data
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

    # all_spectra = all_spectra / np.sum(all_spectra, axis=1, keepdims=True)  # total emissivity normalization
    # emis_std = StandardScaler().fit(all_spectra)

    # all_spectra = emis_std.transform(all_spectra)

    test_labels = [1, 7, 17, 40]

    test_labels_idx = np.where(np.isin(np.argmax(labels, axis=1), test_labels))[0]

    v100_idx = np.where(np.argmax(conditions, axis=1) == 0)[0]
    v100_idx = [i for i in v100_idx if i in test_labels_idx]
    e50_idx = np.where(np.argmax(conditions, axis=1) == 1)[0]
    e50_idx = [i for i in e50_idx if i in test_labels_idx]
    e100_idx = np.where(np.argmax(conditions, axis=1) == 2)[0]
    e100_idx = [i for i in e100_idx if i in test_labels_idx]

    assert np.array_equal(labels[v100_idx], labels[e50_idx]) and np.array_equal(labels[v100_idx], labels[e100_idx]), "labels are not equal"

    v100 = all_spectra[v100_idx]
    e50 = all_spectra[e50_idx]
    e100 = all_spectra[e100_idx]

    all_spectra = np.concatenate([
        v100,
        v100,
        e50,
        e50,
        e100,
        e100
    ], axis=0)

# Apply PCA
n_components = 10  # Choose the number of principal components
pca = PCA(n_components=n_components)
pca.fit(all_spectra)
reduced_spectra = pca.transform(all_spectra)

# Define the custom range for k (clusters)
min_k = 2
max_k = 21

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
    labels = kmeans.fit_predict(reduced_spectra)

    # Overall silhouette score
    score = silhouette_score(reduced_spectra, labels)
    sil_scores.append(score)

    # Inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

    # Silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(reduced_spectra, labels)

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
    ax.set_ylim([0, len(reduced_spectra) + (k + 1) * 10])
    ax.set_title(f'k={k}')
    ax.set_yticks([])
    ax.set_xticks([])

# Adjust spacing between subplots
for i in range(n_plots, n_rows * n_cols):
    fig.delaxes(axes.flat[i])

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('silhouette_plots_test.png')

inertia_values = [i / max(inertia_values) for i in inertia_values]

# Plot overall silhouette score and inertia vs. number of clusters (k) using xy_plot
xy_plot(
    [[i for i in k_values], [i for i in k_values]],
    [sil_scores, inertia_values],
    xy_labels=('k', 'Score'),
    title='k vs. silhouette score and inertia after PCA',
    labels=['silhouette score', 'inertia'],
    show=False,
    save_to='k_vs_silhouette_score_and_inertia_after_PCA_test.png'
)

input_k = int(input("Enter k value: "))

kmeans = KMeans(n_clusters=input_k, random_state=42)
labels_ = kmeans.fit_predict(reduced_spectra)
labels_ = labels_.reshape(-1, 1)

with h5py.File('transformed_data_EtM.h5', 'r') as hf:
    tr_spectra = np.array(hf['spectra'][()])

idx = random.sample(range(reduced_spectra.shape[0]), tr_spectra.shape[0])
reduced_spectra = reduced_spectra[idx]
labels_ = labels_[idx]

# pca.fit(tr_spectra)
tr_spectra = pca.transform(tr_spectra)
tr_labels = kmeans.fit_predict(tr_spectra)
tr_labels = tr_labels.reshape(-1, 1)

best_match = map_labels(labels_.reshape(-1,), tr_labels.reshape(-1,))
score = accuracy_score(labels_.reshape(-1,), tr_labels.reshape(-1,))
print(f"Accuracy score: {score}")

import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X = reduced_spectra[:, :3]  # Assuming you have 3 features
labels = labels_.reshape(-1, )

# Create subplot for original data
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for i in np.unique(labels):
    ax1.scatter(X[labels==i, 0], X[labels==i, 1], X[labels==i, 2], label=f"Cluster {i}")
ax1.set_title("Original Data")
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_zlabel("PC 3")
ax1.legend()

# Perform KMeans clustering on transformed data
X_transformed = tr_spectra[:, :3]  # Assuming you have 3 features
labels_transformed = best_match#.reshape(-1, )

# Create subplot for transformed data
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i in np.unique(labels_transformed):
    ax2.scatter(X_transformed[labels_transformed==i, 0], X_transformed[labels_transformed==i, 1], X_transformed[labels_transformed==i, 2], label=f"Cluster {i}")
ax2.set_title("Transformed Data")
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_zlabel("PC 3")
ax2.legend()

plt.savefig(f"kmeans_test.png")