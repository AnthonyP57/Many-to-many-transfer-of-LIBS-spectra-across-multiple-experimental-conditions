import random
import h5py
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score
from sympy import reduced
from matplotlib.animation import FuncAnimation
from data_visualization import xy_plot
import matplotlib.pyplot as plt
from match_clusters import map_labels

# Load data
with h5py.File('C:\\Users\\Antoni\\PycharmProjects\\Spectra_transfer\\PCA_inliers_mean.h5', 'r') as hf:
    wavelen = np.array(hf['calibration'][()])
    spectra_0_idx = np.where(wavelen == 250)[0][0]
    spectra_last_idx = np.where(wavelen == 850)[0][0]
    conditions = np.array(hf['conditions'][()])
    earth_cond = np.where(conditions[:, 1] == 1)[0]
    mars_spectra = np.array(hf['spectra'][earth_cond, spectra_0_idx:spectra_last_idx+1])#[:, 25: -25]
    all_spectra = np.array(hf['spectra'][:, spectra_0_idx:spectra_last_idx+1])#[:, 25: -25]
    mars_labels = np.array(hf['labels'][earth_cond])
    #indices of first 10 spectra from each one-hot label
    mars_spectra_ = []
    for i in range(mars_labels.shape[1]):
        all_idx = np.where(mars_labels[:, i] == 1)[0]
        random_10 = random.sample(list(all_idx), 20)
        mars_spectra_.append(mars_spectra[random_10])

    mars_spectra = np.array(mars_spectra_).reshape(mars_labels.shape[1]*20, -1)

print(mars_spectra.shape)
# Apply PCA
n_components = 3  # Choose the number of principal components
pca = TSNE(n_components=n_components)
reduced_spectra = pca.fit_transform(mars_spectra)

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
labels_ = kmeans.fit_predict(reduced_spectra)
labels_ = labels_.reshape(-1, 1)

with h5py.File('transformed_data_EtM.h5', 'r') as hf:
    tr_spectra = np.array(hf['spectra'][()])
    mars_labels = np.array(hf['labels'][()])
    mars_spectra_ = []
    for i in range(mars_labels.shape[1]):
        all_idx = np.where(mars_labels[:, i] == 1)[0]
        random_10 = random.sample(list(all_idx), 20)
        mars_spectra_.append(tr_spectra[random_10])

    tr_spectra = np.array(mars_spectra_).reshape(mars_labels.shape[1] * 20, -1)
print(tr_spectra.shape)
# idx = random.sample(range(reduced_spectra.shape[0]), tr_spectra.shape[0])
# reduced_spectra = reduced_spectra[idx]
# labels_ = labels_[idx]

# pca.fit(tr_spectra)
tr_spectra = pca.fit_transform(tr_spectra)
tr_labels = kmeans.fit_predict(tr_spectra)
tr_labels = tr_labels.reshape(-1, 1)

best_match = map_labels(labels_.reshape(-1,), tr_labels.reshape(-1,))
score_ = accuracy_score(labels_.reshape(-1,), tr_labels.reshape(-1,))
score = accuracy_score(labels_.reshape(-1,), best_match.reshape(-1,))
print(f"Accuracy score: {score:.3f} ; previous score: {score_:.3f}")

import matplotlib.pyplot as plt
import numpy as np

X = reduced_spectra[:, :3]  # Assuming you have 3 features
labels = labels_.reshape(-1, )

# Create subplot for original data
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for i in np.unique(labels):
    ax1.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], label=f"Cluster {i}")
ax1.set_title("Original Data")
ax1.set_xlabel("PC 1")
ax1.set_ylabel("PC 2")
ax1.set_zlabel("PC 3")
ax1.legend()

# Perform KMeans clustering on transformed data
X_transformed = tr_spectra[:, :3]  # Assuming you have 3 features
labels_transformed = best_match  # Assuming labels are pre-defined

# Create subplot for transformed data
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
for i in np.unique(labels_transformed):
    ax2.scatter(X_transformed[labels_transformed == i, 0], X_transformed[labels_transformed == i, 1], X_transformed[labels_transformed == i, 2], label=f"Cluster {i}")
ax2.set_title("Transformed Data")
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_zlabel("PC 3")
ax2.legend()

# Function to update the view angle for both subplots
def update(num):
    ax1.view_init(elev=30, azim=num)  # Rotate the first subplot
    ax2.view_init(elev=30, azim=num)  # Rotate the second subplot

# Animate the plot
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 0.25), interval=25)
# ani.save('3d_rotation.gif', writer='pillow', fps=40)  # You can adjust fps as needed

# Show the animation
plt.show()

# Accuracy score: 0.360 ; previous score: 0.255