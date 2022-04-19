#%%
from kdg import kdn
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
import random

from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import ZooAttack
from art.estimators.classification import BlackBoxClassifier
from art.utils import to_categorical

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def getNN(compile_kwargs, input_size, num_classes):
    network_base = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=0)
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(10, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

def experiment(dataset_id, reps=1, T=1e-3, h=0.33, rescale=False):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

    if np.mean(is_categorical) >0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return

    total_sample = X.shape[0]
    unique_classes, counts = np.unique(y, return_counts=True)

    test_sample = min(counts)//3

    indx = []
    for label in unique_classes:
        indx.append(
            np.where(
                y==label
            )[0]
        )

    print('Number unique classes = {}'.format(len(unique_classes)))

    max_sample = min(counts) - test_sample
    train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )

    """
    NOTE(Jacob):
    KDF is throwing divide by zero errors at small samples sizes. Speak with 
    Jayanta about this. Because of this, unable to generate adversarial samples
    since the prediction is always the same number, resulting in l2 of zero which
    doesn't make sense.
    """
    train_samples = [train_samples[-1]]

    # Only use small data for now
    # if max_sample > 5000:
    #     return
    

    l2_kdf_list = []
    l2_rf_list = []
    linf_kdf_list = []
    linf_rf_list = []
    err_adv_rf_list = []
    err_adv_kdf_list = []
    err_rf = []
    err_kdf = []
    mc_rep = []
    samples_attack = []
    samples = []

    for train_sample in train_samples:       
        for rep in range(reps):
            indx_to_take_train = []
            indx_to_take_test = []

            for ii, _ in enumerate(unique_classes):
                np.random.shuffle(indx[ii])
                indx_to_take_train.extend(
                    list(
                            indx[ii][:train_sample]
                    )
                )
                indx_to_take_test.extend(
                    list(
                            indx[ii][-test_sample:counts[ii]]
                    )
                )

            # Define NN parameters
            # compile_kwargs = {
            #     "loss": "binary_crossentropy",
            #     "optimizer": keras.optimizers.Adam(3e-4),
            # }
            compile_kwargs = {
                "loss": 'categorical_crossentropy',
                "optimizer": keras.optimizers.Adam(3e-4),
            }
            fit_kwargs = {
                "epochs": 200,
                "batch_size": 64,
                "verbose": False,
            }
            # kdn_kwargs = {
            #     "k": 1e-5,
            #     "T": 1e-3,
            #     "h": 1/2,
            #     "verbose": False
            # }
            kdn_kwargs = {
                "k": 1e-5,
                "T": T,    # NOTE(Jacob): FIX by allowing low weighted polytopes
                "h": h,
                "verbose": False
            }

            # Scaling
            if rescale:
                scaler = MinMaxScaler()
                scaler.fit(X[indx_to_take_train])
                X[indx_to_take_train] = scaler.transform(X[indx_to_take_train])
                X[indx_to_take_test] = scaler.transform(X[indx_to_take_test])

            # Train vanilla NN
            nn = getNN(compile_kwargs, X.shape[-1], len(unique_classes))
            nn.fit(X[indx_to_take_train], keras.utils.to_categorical(y[indx_to_take_train]), **fit_kwargs)

            model_kdn = kdn(nn, **kdn_kwargs)
            model_kdn.fit(X[indx_to_take_train], y[indx_to_take_train])
            proba_kdf = model_kdn.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdn.predict_proba_nn(X[indx_to_take_test])
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            # Initial classification error
            err_rf.append(
                1 - np.mean(
                    predicted_label_rf==y[indx_to_take_test]
                )
            )
            err_kdf.append(
                1 - np.mean(
                        predicted_label_kdf==y[indx_to_take_test]
                    )
            )
            print('NN error = {:.4f}'.format(err_rf[-1]))
            print('KDN error = {:.4f}'.format(err_kdf[-1]))
    
    return model_kdn, X[indx_to_take_test], y[indx_to_take_test], X[indx_to_take_train], y[indx_to_take_train]


model_kdn, X_test, y_test, X_train, y_train = experiment(16, T=1e-3)
# model_kdn, X_test, y_test, X_train, y_train = experiment(11)
# model_kdn, X_test, y_test, X_train, y_train = experiment(12, T=1e-50)
# single_sample = X_train[0].reshape(-1, 1).T
# proba_kdf = model_kdn.predict_proba(single_sample)

#%% 
unique_classes, counts = np.unique(y_train, return_counts=True)
for i, label in enumerate(unique_classes):
    print(sum(model_kdn.polytope_samples[label]),"\t" , counts[i])
print("")
for i, label in enumerate(unique_classes):
    print(model_kdn.bias[label],"\t" , counts[i])