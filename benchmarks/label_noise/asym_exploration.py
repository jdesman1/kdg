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
from kdg.utils import plot_2dsim
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf


def gen_contaminated_data(func, n_samples, p0, p1):
    """Generate contaminated data with conditional label noise"""
    if func == generate_gaussian_parity:
        X, y = func(n_samples, cluster_std=0.25)
        X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.25)
    elif func == generate_polynomial:
        X, y = func(n_samples, a=[1, 3])
        X_test, y_test = func(1000, a=[1, 3])
    else:
        X, y = func(n_samples)
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

    return X, y, X_test, y_test


# Data generation functions
funcs = [
    generate_gaussian_parity,
    generate_ellipse,
    generate_spirals,
    generate_sinewave,
    generate_polynomial,
]
func_names = ["GXOR", "Ellipse", "Spirals", "Sinewave", "Polynomial"]

# Noise and model parameters
p0 = 0
p1s = np.arange(0, 0.5, 0.1)
n_samples = 1000
n_estimators = 100

# Posterior setup
p = np.arange(-2, 2, step=0.1)
q = np.arange(-2, 2, step=0.1)
xx, yy = np.meshgrid(p, q)

grid_samples = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

# Asymmetric posteriors
for i in range(len(funcs)):
    # Figure
    fig, ax = plt.subplots(3, len(p1s), figsize=(15, 8))
    sns.set_context("talk")
    for idx, p1 in enumerate(p1s):
        # Generate data where p0 =/= p1
        X, y, X_test, y_test = gen_contaminated_data(funcs[i], n_samples, p0, p1)
        plot_2dsim(X, y, ax=ax[0, idx])
        ax[0, idx].set_title("p1 = {:.1f}".format(p1))

        # Fit models
        model_kdf = kdf(kwargs={"n_estimators": n_estimators})
        model_kdf.fit(X, y)
        error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)
        res_kdf = np.int32(model_kdf.predict(X_test) == y_test)

        model_rf = rf(n_estimators=n_estimators)
        model_rf.fit(X, y)
        error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
        res_rf = np.int32(model_rf.predict(X_test) == y_test)

        # KDF posterior
        pdf = model_kdf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[1, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[1, idx].set_title("Error = {:.3f}".format(error_kdf))

        # RF Posterior
        pdf = model_rf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[2, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[2, idx].set_title("Error = {:.3f}".format(error_rf))
    plt.tight_layout()
    plt.savefig("plots/" + func_names[i] + "_posteriors_asym_kdf.pdf")

# Symmetric posteriors
for i in range(len(funcs)):
    # Figure
    fig, ax = plt.subplots(3, len(p1s), figsize=(15, 8))
    sns.set_context("talk")
    for idx, p1 in enumerate(p1s):
        # Generate data where p0=p1
        X, y, X_test, y_test = gen_contaminated_data(funcs[i], n_samples, p1, p1)
        plot_2dsim(X, y, ax=ax[0, idx])
        ax[0, idx].set_title("p0 = p1 = {:.1f}".format(p1))

        # Fit models
        model_kdf = kdf(kwargs={"n_estimators": n_estimators})
        model_kdf.fit(X, y)
        error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)
        res_kdf = np.int32(model_kdf.predict(X_test) == y_test)

        model_rf = rf(n_estimators=n_estimators)
        model_rf.fit(X, y)
        error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
        res_rf = np.int32(model_rf.predict(X_test) == y_test)

        # KDF posterior
        pdf = model_kdf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[1, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[1, idx].set_title("Error = {:.3f}".format(error_kdf))

        # RF Posterior
        pdf = model_rf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[2, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[2, idx].set_title("Error = {:.3f}".format(error_rf))
    plt.tight_layout()
    plt.savefig("plots/" + func_names[i] + "_posteriors_sym_kdf.pdf")

# Compare Asymmetric and Symmetric posteriors
for i in range(len(funcs)):
    # Figure
    fig, ax = plt.subplots(3, len(p1s), figsize=(15, 8))
    sns.set_context("talk")
    for idx, p1 in enumerate(p1s):
        # Generate data where p0=p1
        X, y, X_test, y_test = gen_contaminated_data(funcs[i], n_samples, p1, p1)
        plot_2dsim(X, y, ax=ax[0, idx])
        ax[0, idx].set_title("p0 = p1 = {:.1f}".format(p1))

        # Fit model
        model_kdf = kdf(kwargs={"n_estimators": n_estimators})
        model_kdf.fit(X, y)
        error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)
        res_kdf = np.int32(model_kdf.predict(X_test) == y_test)

        # KDF posterior
        pdf = model_kdf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[1, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[1, idx].set_title("Error = {:.3f}".format(error_kdf))

        # Generate data where p0 =/= p1
        X, y, X_test, y_test = gen_contaminated_data(funcs[i], n_samples, p0, p1)

        # Fit model
        model_kdf = kdf(kwargs={"n_estimators": n_estimators})
        model_kdf.fit(X, y)
        error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)
        res_kdf = np.int32(model_kdf.predict(X_test) == y_test)

        pdf = model_kdf.predict_proba(grid_samples)
        data = pd.DataFrame(
            data={"x": grid_samples[:, 0], "y": grid_samples[:, 1], "z": pdf[:, 0]}
        )
        data = data.pivot(index="x", columns="y", values="z")
        cmap = sns.diverging_palette(40, 240, n=9)
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax[2, idx],
            vmin=0,
            vmax=1,
            yticklabels=False,
            xticklabels=False,
            cbar=False,
        )
        ax[2, idx].set_title("Error = {:.3f}".format(error_kdf))

    plt.tight_layout()
    plt.savefig("plots/" + func_names[i] + "_posteriors_comp_kdf.pdf")
