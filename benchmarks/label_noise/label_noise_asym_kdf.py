import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from kdg.utils import generate_gaussian_parity
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf


def label_noise_gp(n_samples, p0=0.0, p1=0.0, n_estimators=500):
    """Single label noise trial with proportion p0 and p1 of flipped
    labels for class 0 and class 1, respectively."""
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.5)
    X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)

    # Generate noise and flip labels
    y0_idx = np.where(y == 0)[0]
    y1_idx = np.where(y == 1)[0]
    n_noise0 = np.int32(np.round(len(y0_idx) * p0))
    n_noise1 = np.int32(np.round(len(y1_idx) * p1))
    noise_indices0 = random.sample(list(y0_idx), n_noise0)
    noise_indices1 = random.sample(list(y1_idx), n_noise1)
    y[noise_indices0] = 1 - y[noise_indices0]
    y[noise_indices1] = 1 - y[noise_indices1]

    model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    model_kdf.fit(X, y)
    error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)

    model_rf = rf(n_estimators=n_estimators)
    model_rf.fit(X, y)
    error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
    return error_kdf, error_rf


### Run the experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_estimators = 100
n_samples = 1000
p0s = np.arange(0, 0.5, 0.05)
p1s = np.arange(0, 0.5, 0.05)

err_kdf = []
err_rf = []
p0_list = []
p1_list = []
reps_list = []

for p0 in p0s:
    for p1 in p1s:
        print("Doing p0 = {:.2f}, p1 = {:.2f}".format(p0, p1))
        for ii in range(reps):
            err_kdf_i, err_rf_i = label_noise_gp(
                n_samples=n_samples, p0=p0, p1=p1, n_estimators=n_estimators
            )
            err_kdf.append(err_kdf_i)
            err_rf.append(err_rf_i)
            reps_list.append(ii)
            p0_list.append(p0)
            p1_list.append(p1)
            print("KDF error = {:.4f}, RF error = {:.4f}".format(err_kdf_i, err_rf_i))

# Construct DataFrame
df["reps"] = reps_list
df["p0"] = p0_list
df["p1"] = p1_list
df["error_kdf"] = err_kdf
df["error_rf"] = err_rf

# Heatmap params
mapping = np.zeros((len(p0s), len(p1s)))

# Setup some plotting parameters
styles = ["-", "--", "-.", ":", ".", ","]
alphas = np.arange(1, 0, -0.2)
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

for j, p0 in enumerate(p0s):
    err_kdf_med = []
    err_kdf_25_quantile = []
    err_kdf_75_quantile = []
    err_rf_med = []
    err_rf_25_quantile = []
    err_rf_75_quantile = []
    for p1 in p1s:
        df_p0_p1 = df[(df["p0"] == p0) & (df["p1"] == p1)]
        err_kdf_med.append(np.median(df_p0_p1["error_kdf"]))
        err_kdf_25_quantile.append(np.quantile(df_p0_p1["error_kdf"], 0.25))
        err_kdf_75_quantile.append(np.quantile(df_p0_p1["error_kdf"], 0.75))
        err_rf_med.append(np.median(df_p0_p1["error_rf"]))
        err_rf_25_quantile.append(np.quantile(df_p0_p1["error_rf"], 0.25))
        err_rf_75_quantile.append(np.quantile(df_p0_p1["error_rf"], 0.75))

    delta_med = np.array(err_kdf_med) - np.array(err_rf_med)
    mapping[j, :] = delta_med

    ax.plot(p1s, delta_med, label="p0 = {:.2f}".format(p0))

# Finish plotting figure 1
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

ax.set_xlabel("P1")
ax.set_ylabel("KDF Error - RF Error")
ax.legend(frameon=False, bbox_to_anchor=(1.05, 1.0), loc="upper left")
plt.axhline(y=0.0, color="r", linestyle="--")
ax.text(1.05, -0.01, "KDF Wins", transform=ax.get_yaxis_transform())
ax.text(1.05, 0.01, "RF Wins", transform=ax.get_yaxis_transform())
plt.tight_layout()
plt.title("KDF - RF Errors: Gaussian XOR")
plt.savefig("xor_kdf.pdf")
# plt.show()

# Plot figure 2
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
p0_labels = ["{:.2f}".format(p0) for p0 in p0s]
p1_labels = ["{:.2f}".format(p1) for p1 in p1s]
sns.heatmap(
    mapping,
    xticklabels=p1_labels,
    yticklabels=p0_labels,
    cmap="RdBu_r",
    ax=ax,
    cbar=True,
    vmin=-0.1,
    vmax=0.1,
)
ax.set_xlabel("P1")
ax.set_ylabel("P0")
plt.title("KDF - RF Errors: Gaussian XOR")
plt.tight_layout()
plt.savefig("xor_heatmap_kdf.pdf")
# plt.show()
