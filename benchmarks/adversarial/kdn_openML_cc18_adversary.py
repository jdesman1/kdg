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
# %%
def getNN(compile_kwargs, input_size, num_classes):
    network_base = keras.Sequential()
    initializer = keras.initializers.GlorotNormal(seed=0)
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer, input_shape=(input_size,)))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(50, activation="relu", kernel_initializer=initializer))
    network_base.add(keras.layers.Dense(units=num_classes, activation="softmax", kernel_initializer=initializer))
    network_base.compile(**compile_kwargs)
    return network_base

def experiment(dataset_id, folder, n_estimators=500, reps=5, n_attack=50):
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
    if train_samples[-1] > 1000:
        return
    

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
            compile_kwargs = {
                "loss": "binary_crossentropy",
                "optimizer": keras.optimizers.Adam(3e-4),
            }
            fit_kwargs = {
                "epochs": 200,
                "batch_size": 64,
                "verbose": False,
            }
            kdn_kwargs = {
                "k": 1e-5,
                "T": 1e-3,
                "h": 1/2,
                "verbose": False
            }
            
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

            ## Adversarial attack ###
            def _predict_kdf(x):
                """ Wrapper to query black box"""
                proba_kdf = model_kdn.predict_proba(x)
                predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
                return to_categorical(predicted_label_kdf, nb_classes=len(np.unique(y[indx_to_take_train])))

            def _predict_rf(x):
                """ Wrapper to query blackbox for rf"""
                proba_rf = model_kdn.predict_proba_nn(x)
                predicted_label_rf = np.argmax(proba_rf, axis = 1)
                return to_categorical(predicted_label_rf, nb_classes=len(np.unique(y[indx_to_take_train])))

            art_classifier_kdf = BlackBoxClassifier(_predict_kdf, X[indx_to_take_train][0].shape, len(np.unique(y[indx_to_take_train])))
            art_classifier_rf = BlackBoxClassifier(_predict_rf, X[indx_to_take_train][0].shape, len(np.unique(y[indx_to_take_train])))
            attack_rf = HopSkipJump(
                classifier=art_classifier_rf,
                targeted=False,
                max_iter=50,  
                max_eval=1000,
                init_eval=10,
            )
            attack_kdf = HopSkipJump(
                classifier=art_classifier_kdf,
                targeted=False,
                max_iter=50,  
                max_eval=1000,
                init_eval=10,
            )
            # attack_kdf = ZooAttack(
            #     classifier=art_classifier_kdf,
            #     confidence=0.0,
            #     targeted=False,
            #     learning_rate=1e-1,
            #     max_iter=20,
            #     binary_search_steps=10,
            #     initial_const=1e-3,
            #     abort_early=True,
            #     use_resize=False,
            #     use_importance=False,
            #     nb_parallel=1,
            #     batch_size=1,
            #     variable_h=0.2,
            # )

            # attack_rf = ZooAttack(
            #     classifier=art_classifier_rf,
            #     confidence=0.0, # originally 0.0
            #     targeted=False,
            #     learning_rate=1e-1,
            #     max_iter=20,
            #     binary_search_steps=10,
            #     initial_const=1e-3,
            #     abort_early=True,
            #     use_resize=False,
            #     use_importance=False,
            #     nb_parallel=1,
            #     batch_size=1,
            #     variable_h=0.2,
            # )


            ### For computational reasons, attack a random subset that is identified correctly
            # Get indices of correctly classified samples common to both
            # selection_idx = indx_to_take_test
            selection_idx = indx_to_take_train
            proba_kdf = model_kdn.predict_proba(X[selection_idx])
            proba_rf = model_kdn.predict_proba_nn(X[selection_idx])
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            # print(proba_kdf)
            # print(predicted_label_rf)
            # print(np.unique(predicted_label_rf, return_counts=True))
            # print(predicted_label_kdf)
            # print(np.unique(predicted_label_kdf, return_counts=True))
            # print(np.where(predicted_label_kdf==y[selection_idx])[0])
            idx_kdf = np.where(predicted_label_kdf==y[selection_idx])[0]
            idx_rf = np.where(predicted_label_rf==y[selection_idx])[0]
            idx_common = list(np.intersect1d(idx_kdf, idx_rf))

            # Randomly sample from the common indices
            if n_attack > len(idx_common):
                n_attack = len(idx_common)
            idx = random.sample(idx_common, n_attack)
            if n_attack == 0:
                return
            
            ### Generate samples
            x_adv_kdf = attack_kdf.generate(X[selection_idx][idx])
            x_adv_rf = attack_rf.generate(X[selection_idx][idx])

            # Compute norms
            l2_kdf = np.mean(np.linalg.norm(X[selection_idx][idx] - x_adv_kdf, ord=2, axis=1))
            l2_rf = np.mean(np.linalg.norm(X[selection_idx][idx] - x_adv_rf, ord=2, axis=1))
            linf_rf = np.mean(np.linalg.norm(X[selection_idx][idx] - x_adv_rf, ord=np.inf, axis=1))
            linf_kdf = np.mean(np.linalg.norm(X[selection_idx][idx] - x_adv_kdf, ord=np.inf, axis=1))

            ### Classification
            # Make adversarial prediction
            proba_rf = model_kdn.predict_proba_nn(x_adv_rf)
            predicted_label_rf_adv = np.argmax(proba_rf, axis = 1)
            err_adv_rf = 1 - np.mean(predicted_label_rf_adv == y[selection_idx][idx])

            proba_kdf = model_kdn.predict_proba(x_adv_kdf)
            predicted_label_kdf_adv = np.argmax(proba_kdf, axis = 1)
            err_adv_kdf = 1 - np.mean(predicted_label_kdf_adv == y[selection_idx][idx])


            print("l2_rf = {:.4f}, linf_rf = {:.4f}, err_rf = {:.4f}".format(l2_rf, linf_rf, err_adv_rf))
            print("l2_kdf = {:.4f}, linf_kdf = {:.4f}, err_kdf = {:.4f}".format(l2_kdf, linf_kdf, err_adv_kdf))
            

            l2_kdf_list.append(l2_kdf)
            l2_rf_list.append(l2_rf)
            linf_kdf_list.append(linf_kdf)
            linf_rf_list.append(linf_rf)
            err_adv_kdf_list.append(err_adv_kdf)
            err_adv_rf_list.append(err_adv_rf)

            mc_rep.append(rep)
            samples_attack.append(n_attack)
            samples.append(train_sample)

    # df = pd.DataFrame() 
    # df['l2_kdf'] = l2_kdf_list
    # df['l2_rf'] = l2_rf_list
    # df['linf_kdf'] = linf_kdf_list
    # df['linf_rf'] = linf_rf_list
    # df['err_kdf'] = err_kdf
    # df['err_rf'] = err_rf
    # df['err_adv_kdf'] = err_adv_kdf_list
    # df['err_adv_rf'] = err_adv_rf_list
    # df['rep'] = mc_rep
    # df['samples_attack'] = samples_attack
    # df['samples'] = samples

    # df.to_csv(folder+'/'+'openML_cc18_'+str(dataset_id)+'.csv')

folder = 'openml_res_adv_zoo'
# os.mkdir(folder)
# benchmark_suite = openml.study.get_suite('OpenML-CC18')
# experiment(6, folder, n_estimators=500, reps=2, n_attack=5)
# experiment(11, folder, n_estimators=500, reps=2, n_attack=10)
# experiment(1497, folder, n_estimators=500, reps=2, n_attack=20)
experiment(16, folder, n_estimators=500, reps=2, n_attack=5)
# for dataset_id in openml.study.get_suite("OpenML-CC18").data:
#     experiment(dataset_id, folder, n_estimators=500, reps=10, n_attack = 20)


#%%
folder = 'openml_res_adv_hsj'
os.mkdir(folder)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                folder
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )
