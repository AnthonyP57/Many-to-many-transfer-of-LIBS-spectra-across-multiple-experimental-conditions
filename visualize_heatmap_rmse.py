import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example Data for two different RMSE metrics
data1 = {
    'Vacuum 100%': {'Vacuum 100%': None, 'Earth 50%': 3.09, 'Earth 100%': 3.17},
    'Earth 50%':    {'Vacuum 100%': 3.27, 'Earth 50%': None, 'Earth 100%': 3.30},
    'Earth 100%':   {'Vacuum 100%': 2.68, 'Earth 50%': 2.72, 'Earth 100%': None},
}
data2 = {
    'Vacuum 100%': {'Vacuum 100%': None, 'Earth 50%': 1.97, 'Earth 100%': 2.53},
    'Earth 50%':    {'Vacuum 100%': 1.68, 'Earth 50%': None, 'Earth 100%': 2.69},
    'Earth 100%':   {'Vacuum 100%': 1.76, 'Earth 50%': 2.32, 'Earth 100%': None},
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)

for i, (df, title) in enumerate(zip([df1, df2], ['Metric A', 'Metric B'])):
    ax = axes[i]
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=False,
        annot_kws={"size": 18},
        ax=ax,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    # Subplot label (a) and (b)
    ax.text(-0.25, 1.05, f'{chr(97 + i)})', transform=ax.transAxes,
            fontsize=18, fontweight='normal', va='top', ha='right')
    # Center ticks on heatmap cells
    ax.set_xticks(np.arange(len(df.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df.index)) + 0.5)
    # Tick labels: horizontal x, vertical y
    ax.set_xticklabels(df.columns, rotation=0, ha='center', fontsize=12.5)
    ax.set_yticklabels(df.index, rotation=90, va='center', fontsize=12.5)
    # Move ticks outside
    ax.tick_params(axis='x', which='both', direction='out', pad=10)
    ax.tick_params(axis='y', which='both', direction='out', pad=10)
    ax.set_xlabel('')
    ax.set_ylabel('')

# Shared labels
fig.text(0.5, 0.04, 'Target Condition', ha='center', fontsize=18)
fig.text(0.04, 0.5, 'Source Condition', va='center', rotation='vertical', fontsize=18)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()
