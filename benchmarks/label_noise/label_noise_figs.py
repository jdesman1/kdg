import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from kdg.utils import plot_2dsim
from kdg.utils import (
    generate_gaussian_parity,
    generate_ellipse,
    generate_spirals,
    generate_sinewave,
    generate_polynomial,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 2D simulation plotting
n_samples = 1e4
X, y = {}, {}
X["gxor"], y["gxor"] = generate_gaussian_parity(n_samples)
X["spiral"], y["spiral"] = generate_spirals(n_samples)
X["circle"], y["circle"] = generate_ellipse(n_samples)
X["sine"], y["sine"] = generate_sinewave(n_samples)
X["poly"], y["poly"] = generate_polynomial(n_samples, a=[1, 3])

# Heatmap plotting
func_names = ["GXOR", "Spirals", "Ellipse", "Sinewave", "Polynomial"]
func_labels = ["Gaussian XOR", "Spiral", "Ellipse", "Sinewave", "Polynomial"]
p0s = np.arange(0, 0.5, 0.05)
p1s = np.arange(0, 0.5, 0.05)
p0_labels = ["{:.1f}".format(p0) for p0 in p0s]
p0_labels = ["" if i % 2 != 0 else p0_labels[i] for i in range(len(p0_labels))]
p1_labels = ["{:.1f}".format(p1) for p1 in p1s]
p1_labels = ["" if i % 2 != 0 else p1_labels[i] for i in range(len(p1_labels))]

fig, ax = plt.subplots(3, 5, figsize=(12, 6))

for i, key in enumerate(X.keys()):
    plot_2dsim(X[key], y[key], ax=ax[0, i])
    ax[0, i].set_aspect("equal")
    ax[0, i].set_title(func_labels[i])

fig.text(0.09, 0.77, "Simulation Data", va="center", rotation="vertical", fontsize=11)
# KDF plotting
for i, f in enumerate(func_names):
    fname = "data/" + f + "_mapping_kdf.pkl"
    with open(fname, "rb") as fp:
        mapping = pickle.load(fp)

    hm = sns.heatmap(
        mapping,
        xticklabels=p1_labels,
        yticklabels=p0_labels,
        cmap="RdBu_r",
        ax=ax[1, i],
        center=0,
        cbar=False,
        cbar_kws={"shrink": 0.7},
        vmax=0.3,
        vmin=-0.3,
    )

    if i == len(func_names) - 1:
        # [x, y, width, height]
        cbar_ax = fig.add_axes([0.90, 0.38, 0.01, 0.23])
        cbar_ax.tick_params(labelsize=8)
        sns.heatmap(
            mapping,
            xticklabels=p1_labels,
            yticklabels=p0_labels,
            cmap="RdBu_r",
            ax=ax[1, i],
            center=0,
            cbar=True,
            vmax=0.3,
            vmin=-0.3,
            cbar_ax=cbar_ax,
        )

    ax[1, i].set_aspect("equal")
    ax[1, i].set_yticklabels(ax[1, i].get_yticklabels(), rotation=0)
    # ax[1, i].set_ylabel("p0")
    # ax[1, i].set_xlabel("p1")

fig.text(
    0.09, 0.50, "KDF - RF Noise Error", va="center", rotation="vertical", fontsize=11
)

# KDN plotting
for i, f in enumerate(func_names):
    fname = "data/" + f + "_mapping_kdn.pkl"
    with open(fname, "rb") as fp:
        mapping = pickle.load(fp)

    sns.heatmap(
        mapping,
        xticklabels=p1_labels,
        yticklabels=p0_labels,
        cmap="RdBu_r",
        ax=ax[2, i],
        center=0,
        cbar=False,
        vmax=0.3,
        vmin=-0.3,
    )

    if i == len(func_names) - 1:
        # [x, y, width, height]
        cbar_ax = fig.add_axes([0.90, 0.11, 0.01, 0.23])
        cbar_ax.tick_params(labelsize=8)
        sns.heatmap(
            mapping,
            xticklabels=p1_labels,
            yticklabels=p0_labels,
            cmap="RdBu_r",
            ax=ax[2, i],
            center=0,
            cbar=True,
            vmax=0.3,
            vmin=-0.3,
            cbar_ax=cbar_ax,
        )

    ax[2, i].set_aspect("equal")
    ax[2, i].set_yticklabels(ax[2, i].get_yticklabels(), rotation=0)

fig.text(
    0.09, 0.21, "KDN - DN Noise Error", va="center", rotation="vertical", fontsize=11
)

# plt.tight_layout()
# plt.show()
plt.savefig("plots/label_noise_heatmaps.pdf", transparent=True, bbox_inches="tight")
