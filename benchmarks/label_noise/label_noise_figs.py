import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Create KDF figure
with open("data/xor_heatmap_kdf.pkl", "rb") as f:
    mapping = pickle.load(f)

# Setup some plotting parameters
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
p0s = np.arange(0, 0.5, 0.05)
p1s = np.arange(0, 0.5, 0.05)
p0_labels = ["{:.2f}".format(p0) for p0 in p0s]
p1_labels = ["{:.2f}".format(p1) for p1 in p1s]

# Make every other label invisible
for j, p1 in enumerate(p1s):
    if j % 2 == 0:
        p1_labels[j] = ""
        p0_labels[j] = ""

sns.heatmap(
    mapping,
    xticklabels=p1_labels,
    yticklabels=p0_labels,
    cmap="RdBu_r",
    ax=ax,
    cbar=True,
    cbar_kws={"shrink": 0.7},
    vmin=-0.1,
    vmax=0.1,
)
ax.set_xlabel("p1")
ax.set_ylabel("p0")
ax.set_aspect("equal", "box")
plt.title("Gaussian XOR Label Noise: KDF - RF Errors")
plt.tight_layout()
plt.savefig("plots/label_noise_kdf.pdf")

# Create KDN figure
with open("data/xor_heatmap_kdn.pkl", "rb") as f:
    mapping = pickle.load(f)

# Setup some plotting parameters
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
p0s = np.arange(0, 0.5, 0.05)
p1s = np.arange(0, 0.5, 0.05)
p0_labels = ["{:.2f}".format(p0) for p0 in p0s]
p1_labels = ["{:.2f}".format(p1) for p1 in p1s]

# Make every other label invisible
for j, p1 in enumerate(p1s):
    if j % 2 == 0:
        p1_labels[j] = ""
        p0_labels[j] = ""

sns.heatmap(
    mapping,
    xticklabels=p1_labels,
    yticklabels=p0_labels,
    cmap="RdBu_r",
    ax=ax,
    cbar=True,
    cbar_kws={"shrink": 0.7},
    vmin=-0.2,
    vmax=0.2,
)
ax.set_xlabel("p1")
ax.set_ylabel("p0")
ax.set_aspect("equal", "box")
plt.title("Gaussian XOR Label Noise: KDN - DN Errors")
plt.tight_layout()
plt.savefig("plots/label_noise_kdn.pdf")
