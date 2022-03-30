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
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf
import pickle


def label_noise_trial(func, n_samples, p0=0.0, p1=0.0, n_estimators=500):
    """Single label noise trial with proportion p0 and p1 of flipped
    labels for class 0 and class 1, respectively.
    Also allows for different data generation functions."""
    if func == generate_gaussian_parity:
        X, y = func(n_samples, cluster_std=0.5)
        X_test, y_test = generate_gaussian_parity(1000, cluster_std=0.5)
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

    model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    model_kdf.fit(X, y)
    error_kdf = 1 - np.mean(model_kdf.predict(X_test) == y_test)

    model_rf = rf(n_estimators=n_estimators)
    model_rf.fit(X, y)
    error_rf = 1 - np.mean(model_rf.predict(X_test) == y_test)
    return error_kdf, error_rf


# Data generation functions
funcs = [
    generate_gaussian_parity,
    generate_ellipse,
    generate_spirals,
    generate_sinewave,
    generate_polynomial,
]
func_names = ["GXOR", "Ellipse", "Spirals", "Sinewave", "Polynomial"]

for i in range(len(funcs)):
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
                err_kdf_i, err_rf_i = label_noise_trial(
                    func=funcs[i],
                    n_samples=n_samples,
                    p0=p0,
                    p1=p1,
                    n_estimators=n_estimators,
                )
                err_kdf.append(err_kdf_i)
                err_rf.append(err_rf_i)
                reps_list.append(ii)
                p0_list.append(p0)
                p1_list.append(p1)
                print(
                    "KDF error = {:.4f}, RF error = {:.4f}".format(err_kdf_i, err_rf_i)
                )

    # Construct DataFrame
    df["reps"] = reps_list
    df["p0"] = p0_list
    df["p1"] = p1_list
    df["error_kdf"] = err_kdf
    df["error_rf"] = err_rf

    # Heatmap params
    mapping = np.zeros((len(p0s), len(p1s)))

    for j, p0 in enumerate(p0s):
        for k, p1 in enumerate(p1s):
            df_p0_p1 = df[(df["p0"] == p0) & (df["p1"] == p1)]
            mapping[j, k] = np.median(df_p0_p1["error_kdf"] - df_p0_p1["error_rf"])

    # Save data
    fname = "data/" + func_names[i] + "_mapping_kdf.pkl"
    with open(fname, "wb") as f:
        pickle.dump(mapping, f)