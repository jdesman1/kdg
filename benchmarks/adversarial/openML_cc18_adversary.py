#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
import os
from os import listdir, getcwd 

from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import ZooAttack
from art.estimators.classification import BlackBoxClassifier
from art.utils import to_categorical
import random
# %%
def experiment(dataset_id, folder, n_estimators=500, reps=10, n_attack=20):
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
    

    l2_kdf_list = []
    l2_rf_list = []
    linf_kdf_list = []
    linf_rf_list = []
    err_adv_rf_list = []
    err_adv_kdf_list = []
    err_rf = []
    err_kdf = []
    mc_rep = []
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

            # Fit the estimators
            model_kdf = kdf(kwargs={'n_estimators':n_estimators, 'min_samples_leaf':int(np.ceil(X.shape[1]*10/np.log(train_sample)))})
            model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train])
            proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdf.rf_model.predict_proba(X[indx_to_take_test])
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
                proba_kdf = model_kdf.predict_proba(x)
                predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
                return to_categorical(predicted_label_kdf, nb_classes=len(np.unique(y[indx_to_take_train])))

            def _predict_rf(x):
                """ Wrapper to query blackbox for rf"""
                proba_rf = model_kdf.rf_model.predict_proba(x)
                predicted_label_rf = np.argmax(proba_rf, axis = 1)
                return to_categorical(predicted_label_rf, nb_classes=len(np.unique(y[indx_to_take_train])))

            art_classifier_kdf = BlackBoxClassifier(_predict_kdf, X[indx_to_take_train][0].shape, len(np.unique(y[indx_to_take_train])))
            art_classifier_rf = BlackBoxClassifier(_predict_rf, X[indx_to_take_train][0].shape, len(np.unique(y[indx_to_take_train])))
            # attack_rf = HopSkipJump(
            #     classifier=art_classifier_rf,
            #     targeted=False,
            #     max_iter=0,  # TODO: examples show 0, try changing
            #     max_eval=1000,
            #     init_eval=10,
            # )
            # attack_kdf = HopSkipJump(
            #     classifier=art_classifier_kdf,
            #     targeted=False,
            #     max_iter=0,  # TODO: examples show 0, try changing
            #     max_eval=1000,
            #     init_eval=10,
            # )
            attack_kdf = ZooAttack(
                classifier=art_classifier_kdf,
                confidence=0.0,
                targeted=False,
                learning_rate=1e-1,
                max_iter=20,
                binary_search_steps=10,
                initial_const=1e-3,
                abort_early=True,
                use_resize=False,
                use_importance=False,
                nb_parallel=1,
                batch_size=1,
                variable_h=0.2,
            )

            attack_rf = ZooAttack(
                classifier=art_classifier_rf,
                confidence=0.3, # originally 0.0
                targeted=False,
                learning_rate=1e-1,
                max_iter=20,
                binary_search_steps=10,
                initial_const=1e-3,
                abort_early=True,
                use_resize=False,
                use_importance=False,
                nb_parallel=1,
                batch_size=1,
                variable_h=0.2,
            )

            # For computational reasons, attack a random subset
            # idx = random.sample(indx_to_take_test, n_attack)
            idx = random.sample(indx_to_take_train, n_attack)

            ### Generate samples
            x_adv_kdf = attack_kdf.generate(X[idx])
            x_adv_rf = attack_rf.generate(X[idx])

            # Compute norms
            l2_kdf = np.mean(np.linalg.norm(X[idx] - x_adv_kdf, ord=2, axis=1))
            l2_rf = np.mean(np.linalg.norm(X[idx] - x_adv_rf, ord=2, axis=1))
            linf_rf = np.mean(np.linalg.norm(X[idx] - x_adv_rf, ord=np.inf, axis=1))
            linf_kdf = np.mean(np.linalg.norm(X[idx] - x_adv_kdf, ord=np.inf, axis=1))

            ### Classification
            # Make adversarial prediction
            proba_rf = model_kdf.rf_model.predict_proba(x_adv_rf)
            predicted_label_rf_adv = np.argmax(proba_rf, axis = 1)
            err_adv_rf = 1 - np.mean(predicted_label_rf_adv == y[idx])

            proba_kdf = model_kdf.predict_proba(x_adv_kdf)
            predicted_label_kdf_adv = np.argmax(proba_kdf, axis = 1)
            err_adv_kdf = 1 - np.mean(predicted_label_kdf_adv == y[idx])


            print("l2_rf = {:.2f}, linf_rf = {:.2f}, err_rf = {:.2f}".format(l2_rf, linf_rf, err_adv_rf))
            print("l2_kdf = {:.2f}, linf_kdf = {:.2f}, err_kdf = {:.2f}".format(l2_kdf, linf_kdf, err_adv_kdf))
            

            l2_kdf_list.append(l2_kdf)
            l2_rf_list.append(l2_rf)
            linf_kdf_list.append(linf_kdf)
            linf_rf_list.append(linf_rf)
            err_adv_kdf_list.append(err_adv_kdf)
            err_adv_rf_list.append(err_adv_rf)

            mc_rep.append(rep)
            samples.append(train_sample*len(unique_classes))

    df = pd.DataFrame() 
    df['l2_kdf'] = l2_kdf_list
    df['l2_rf'] = l2_rf_list
    df['linf_kdf'] = linf_kdf_list
    df['linf_rf'] = linf_rf_list
    df['err_kdf'] = err_kdf
    df['err_rf'] = err_rf
    df['err_adv_kdf'] = err_adv_kdf_list
    df['err_adv_rf'] = err_adv_rf_list
    df['rep'] = mc_rep
    df['samples'] = samples

    df.to_csv(folder+'/'+'openML_cc18_'+str(dataset_id)+'.csv')

# folder = 'openml_res_adv_zoo'
# os.mkdir(folder)
# benchmark_suite = openml.study.get_suite('OpenML-CC18')
# experiment(6, folder, n_estimators=500, reps=2, n_attack=5)
# experiment(11, folder, n_estimators=500, reps=2, n_attack=5)
# for dataset_id in openml.study.get_suite("OpenML-CC18").data:
#     experiment(dataset_id, folder, n_estimators=500, reps=10, n_attack = 20)

# %%


#%%
folder = 'openml_res_adv_zoo'
# os.mkdir(folder)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment)(
                dataset_id,
                folder
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )

'''Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment_rf)(
                dataset_id,
                folder_rf
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )'''
'''for task_id in benchmark_suite.tasks:
    filename = 'openML_cc18_' + str(task_id) + '.csv'

    if filename not in files:
        print(filename)
        try:
            experiment(task_id,folder)
        except:
            print("couldn't run!")
        else:
            print("Ran successfully!")'''
# %%