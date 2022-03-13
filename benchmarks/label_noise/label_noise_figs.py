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

fig, ax = plt.subplots(3, 5, figsize=(12, 6))

for i, key in enumerate(X.keys()):
    plot_2dsim(X[key], y[key], ax=ax[0, i])
    ax[0, i].set_aspect("equal")
    ax[0, i].get_xaxis().set_ticks([])
    ax[0, i].get_yaxis().set_ticks([])
    ax[0, i].set_title(func_labels[i])

    if i == 0:
        ax[0, i].set_ylabel("Simulation Data")


# KDF plotting
for i, f in enumerate(func_names):
    fname = "data/" + f + "_mapping_kdf.pkl"
    with open(fname, "rb") as fp:
        mapping = pickle.load(fp)

    sns.heatmap(
        mapping,
        xticklabels=False,
        yticklabels=False,
        cmap="RdBu_r",
        ax=ax[1, i],
        center=0,
        cbar=False,
        cbar_kws={"shrink": 0.7},
    )
    ax[1, i].set_aspect("equal")

    if i == 0:
        ax[1, i].set_ylabel("KDF - RF Noise Error")

# KDN plotting
for i, f in enumerate(func_names):
    fname = "data/" + f + "_mapping_kdn.pkl"
    with open(fname, "rb") as fp:
        mapping = pickle.load(fp)

    sns.heatmap(
        mapping,
        xticklabels=False,
        yticklabels=False,
        cmap="RdBu_r",
        ax=ax[2, i],
        center=0,
        cbar=False,
    )
    ax[2, i].set_aspect("equal")

    if i == 0:
        ax[2, i].set_ylabel("KDN - DN Noise Error")

plt.tight_layout()
plt.savefig("plots/label_noise_heatmaps.pdf")
