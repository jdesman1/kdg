import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from kdg.utils import (
    generate_gaussian_parity,
    generate_ellipse,
    generate_spirals,
    generate_sinewave,
    generate_polynomial,
)
from kdg import kdf, kdn
from tensorflow import keras
from tensorflow.keras import layers
import pickle

# network architecture
def getNN():
    compile_kwargs = {
        "loss": "binary_crossentropy",
        "optimizer": keras.optimizers.Adam(3e-4),
    }
    network_base = keras.Sequential()
    network_base.add(layers.Dense(5, activation="relu", input_shape=(2,)))
    network_base.add(layers.Dense(5, activation="relu"))
    network_base.add(layers.Dense(units=2, activation="softmax"))
    network_base.compile(**compile_kwargs)
    return network_base


def label_noise_trial(func, n_samples, p0=0.0, p1=0.0):
    """Single label noise trial with proportion p0 and p1 of flipped
    labels for class 0 and class 1, respectively.
    Also allows for different data generation functions."""
    if func == generate_gaussian_parity:
        X, y = func(n_samples, cluster_std=0.5)
        X_val, y_val = func(1000, cluster_std=0.5)
        X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
    elif func == generate_polynomial:
        X, y = func(n_samples, a=[1, 3])
        X_val, y_val = func(1000, a=[1, 3])
        X_test, y_test = func(1000, a=[1, 3])
    else:
        X, y = func(n_samples)
        X_val, y_val = func(1000)
        X_test, y_test = func(1000)

    # Generate noise and flip labels
    y0_idx = np.where(y == 0)[0]
    y1_idx = np.where(y == 1)[0]
    n_noise0 = np.int32(np.round(len(y0_idx) * p0))
    n_noise1 = np.int32(np.round(len(y1_idx) * p1))
    noise_indices0 = random.sample(list(y0_idx), n_noise0)
    noise_indices1 = random.sample(list(y1_idx), n_noise1)
    y[noise_indices0] = 1 - y[noise_indices0]
    y[noise_indices1] = 1 - y[noise_indices1]

    # NN params
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=False
    )
    fit_kwargs = {
        "epochs": 200,
        "batch_size": 64,
        "verbose": False,
        "validation_data": (X_val, keras.utils.to_categorical(y_val)),
        "callbacks": [callback],
    }

    # Train vanilla NN
    nn = getNN()
    nn.fit(X, keras.utils.to_categorical(y), **fit_kwargs)
    proba_nn = nn.predict(X_test)
    error_nn = 1 - np.mean(np.argmax(nn.predict(X_test), axis=1) == y_test)

    model_kdn = kdn(nn)
    model_kdn.fit(X, y)
    error_kdn = 1 - np.mean(model_kdn.predict(X_test) == y_test)
    return error_kdn, error_nn


### Run the experiment with varying proportion of label noise
df = pd.DataFrame()
reps = 10
n_samples = 1000
p0s = np.arange(0, 0.5, 0.05)
p1s = np.arange(0, 0.5, 0.05)

err_kdn = []
err_nn = []
p0_list = []
p1_list = []
reps_list = []

for p0 in p0s:
    for p1 in p1s:
        print("Doing p0 = {:.2f}, p1 = {:.2f}".format(p0, p1))
        for ii in range(reps):
            err_kdn_i, err_nn_i = label_noise_trial(
                generate_gaussian_parity,
                n_samples=n_samples,
                p0=p0,
                p1=p1,
            )
            err_kdn.append(err_kdn_i)
            err_nn.append(err_nn_i)
            reps_list.append(ii)
            p0_list.append(p0)
            p1_list.append(p1)
            print("KDF error = {:.4f}, NN error = {:.4f}".format(err_kdn_i, err_nn_i))

# Construct DataFrame
df["reps"] = reps_list
df["p0"] = p0_list
df["p1"] = p1_list
df["error_kdn"] = err_kdn
df["error_nn"] = err_nn

# Heatmap params
mapping = np.zeros((len(p0s), len(p1s)))

# Setup some plotting parameters
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
        err_kdf_med.append(np.median(df_p0_p1["error_kdn"]))
        err_kdf_25_quantile.append(np.quantile(df_p0_p1["error_kdn"], 0.25))
        err_kdf_75_quantile.append(np.quantile(df_p0_p1["error_kdn"], 0.75))
        err_rf_med.append(np.median(df_p0_p1["error_nn"]))
        err_rf_25_quantile.append(np.quantile(df_p0_p1["error_nn"], 0.25))
        err_rf_75_quantile.append(np.quantile(df_p0_p1["error_nn"], 0.75))

    delta_med = np.array(err_kdf_med) - np.array(err_rf_med)
    mapping[j, :] = delta_med

# Save data
with open("data/xor_heatmap_kdn.pkl", "wb") as f:
    pickle.dump(mapping, f)

# Finish plotting figure 1
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

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
plt.title("KDN - NN Errors: Gaussian XOR")
plt.tight_layout()
plt.savefig("xor_heatmap_kdn.pdf")
