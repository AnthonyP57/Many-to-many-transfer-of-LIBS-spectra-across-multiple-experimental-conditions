import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from data_visualization import xy_plot
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

# Load data
conc_df = pd.read_excel('/home/antonipa57/PycharmProjects/Spectra_transfer/Regoliths/Concentrations.xlsx')
conc_df = conc_df.dropna()
conc_df = conc_df.transpose()
samples = list(conc_df.index[1:])
minerals = list(conc_df.iloc[0].transpose())
conc_df = conc_df.iloc[1:].reset_index(drop=True)

conc_df = np.array(conc_df)

std = StandardScaler()
conc_df = std.fit_transform(conc_df)

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
    labels = kmeans.fit_predict(conc_df)

    # Overall silhouette score
    score = silhouette_score(conc_df, labels)
    sil_scores.append(score)

    # Inertia (within-cluster sum of squares)
    inertia_values.append(kmeans.inertia_)

    # Silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(conc_df, labels)

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
    ax.set_ylim([0, len(conc_df) + (k + 1) * 10])
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
labels = kmeans.fit_predict(conc_df)

sample_names = {m: int(c) for m, c in zip(samples, labels)}

print(sample_names)

with open('kmeans_conc_sample_name_to_cluster.json', 'w') as fp:
    json.dump(sample_names, fp)