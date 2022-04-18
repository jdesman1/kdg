import numpy as np
import matplotlib.pyplot as plt
import random
from kdg.utils import generate_gaussian_parity
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf
import pandas as pd
import os

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from art.utils import to_categorical

def get_adversarial_examples(model, x_attack, n_attack=20, nb_classes=2, idx=None):
    """ 
    Get adversarial examples from a trained model.
    """

    # Create a BB classifier: prediction function, num features, num classes
    def _predict(x):
        """ Wrapper function to query black box"""
        return to_categorical(model.predict(x), nb_classes=nb_classes)

    art_classifier = BlackBoxClassifier(_predict, x_attack[0].shape, 2)

    # Create an attack model
    attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,
        max_iter=20,
        max_eval=1000,
        init_eval=10,
    )

    # Attack a random subset
    if idx is None:
        idx = random.sample(list(np.arange(0, len(x_attack))), n_attack)
    
    x_train_adv = attack.generate(x_attack[idx])
    return x_train_adv, idx, model

def plot_results(model, x_train, y_train, x_train_adv, num_classes):
    """
    Utility function for visualizing how the attack was performed.
    """
    fig, axs = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))

    colors = ["orange", "blue", "green"]

    for i_class in range(num_classes):

        # Plot difference vectors
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].plot([x_1_0, x_2_0], [x_1_1, x_2_1], c="black", zorder=1)

        # Plot benign samples
        for i_class_2 in range(num_classes):
            axs[i_class].scatter(
                x_train[y_train == i_class_2][:, 0],
                x_train[y_train == i_class_2][:, 1],
                s=20,
                zorder=2,
                c=colors[i_class_2],
            )
        axs[i_class].set_aspect("equal", adjustable="box")

        # Show predicted probability as contour plot
        h = 0.05
        x_min, x_max = -2, 2
        y_min, y_max = -2, 2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z_proba = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z_proba = Z_proba[:, i_class].reshape(xx.shape)
        im = axs[i_class].contourf(
            xx,
            yy,
            Z_proba,
            levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            vmin=0,
            vmax=1,
        )
        if i_class == num_classes - 1:
            cax = fig.add_axes([0.95, 0.2, 0.025, 0.6])
            plt.colorbar(im, ax=axs[i_class], cax=cax)

        # Plot adversarial samples
        for i in range(y_train[y_train == i_class].shape[0]):
            x_1_0 = x_train[y_train == i_class][i, 0]
            x_1_1 = x_train[y_train == i_class][i, 1]
            x_2_0 = x_train_adv[y_train == i_class][i, 0]
            x_2_1 = x_train_adv[y_train == i_class][i, 1]
            if x_1_0 != x_2_0 or x_1_1 != x_2_1:
                axs[i_class].scatter(x_2_0, x_2_1, zorder=2, c="red", marker="X")
        axs[i_class].set_xlim((x_min, x_max))
        axs[i_class].set_ylim((y_min, y_max))

        axs[i_class].set_title("class " + str(i_class))
        axs[i_class].set_xlabel("feature 1")
        axs[i_class].set_ylabel("feature 2")

def highlight_err(row):
    """ If err_adv_kdf is greater than err_adv_rf, then highlight
    that value. Otherwise, highlight err_adv_rf"""
    ret = ["" for _ in row.index]
    if row['err_adv_kdf'] < row['err_adv_rf']:
        # return 'background-color: yellow'
        ret[row.index.get_loc("err_adv_kdf")] = 'background-color: green'
        return ret
    else:
        # return 'background-color: green'
        ret[row.index.get_loc("err_adv_rf")] = 'background-color: green'
        return ret

def plot_adversarial(res_folder):
    files = os.listdir(res_folder)
    fname = []
    l2_rf = []
    l2_kdf = []
    linf_rf = []
    linf_kdf = []
    err_adv_kdf = []
    err_adv_rf = []
    delta_adv_err_list = []
    delta_adv_l2_list = []
    delta_adv_linf_list = []
    for file in files:
        # print(file, ': ')
        df = pd.read_csv(res_folder+'/'+file, index_col=0)

        l2_mean_rf = df['l2_rf'].mean()
        linf_mean_rf = df['linf_rf'].mean()

        l2_mean_kdf = df['l2_kdf'].mean()
        linf_mean_kdf = df['linf_kdf'].mean()

        err_adv_mean_kdf = df['err_adv_kdf'].mean()
        err_adv_mean_rf = df['err_adv_rf'].mean()

        err_mean_kdf = df['err_kdf'].mean()
        err_mean_rf = df['err_rf'].mean()

        delta_adv_err = np.mean(df['err_adv_kdf'] - df['err_adv_rf'])
        delta_adv_l2 = np.mean(df['l2_kdf'] - df['l2_rf'])
        delta_adv_linf = np.mean(df['linf_kdf'] - df['linf_rf'])

        fname.append(file)
        l2_rf.append(l2_mean_rf)
        l2_kdf.append(l2_mean_kdf)
        linf_rf.append(linf_mean_rf)
        linf_kdf.append(linf_mean_kdf)
        err_adv_kdf.append(err_adv_mean_kdf)
        err_adv_rf.append(err_adv_mean_rf)
        delta_adv_err_list.append(delta_adv_err)
        delta_adv_l2_list.append(delta_adv_l2)
        delta_adv_linf_list.append(delta_adv_linf)

    df = pd.DataFrame() 
    df['fname'] = fname
    df['l2_kdf'] = l2_kdf
    df['l2_rf'] = l2_rf
    df['linf_kdf'] = linf_kdf
    df['linf_rf'] = linf_rf
    df['err_adv_kdf'] = err_adv_kdf
    df['err_adv_rf'] = err_adv_rf
    df['delta_adv_err'] = delta_adv_err_list
    df['delta_adv_l2'] = delta_adv_l2_list
    df['delta_adv_linf'] = delta_adv_linf_list
    return df