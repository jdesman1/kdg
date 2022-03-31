# %%
import numpy as np
import matplotlib.pyplot as plt
import random
from kdg.utils import generate_gaussian_parity
from kdg import kdf
from sklearn.ensemble import RandomForestClassifier as rf

from art.attacks.evasion import HopSkipJump
from art.attacks.evasion import ZooAttack
from art.estimators.classification import BlackBoxClassifier
from art.utils import to_categorical

# %% Utility functions
def get_adversarial_examples(model, x_attack, n_attack=20, nb_classes=2, idx=None):
    """ 
    Get adversarial examples from a trained model.
    """

    # Create a BB classifier: prediction function, num features, num classes
    def _predict(x):
        """ Surrogate function to query black box"""
        return to_categorical(model.predict(x), nb_classes=nb_classes)

    art_classifier = BlackBoxClassifier(_predict, x_attack[0].shape, 2)

    # Create an attack model
    attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,
        max_iter=0,  # TODO: examples show 0, try changing
        max_eval=1000,
        init_eval=10,
    )
    # attack = ZooAttack(
    #     classifier=art_classifier,
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

    # Attack a random subset
    if idx is None:
        idx = random.sample(list(np.arange(0, len(x_attack))), n_attack)

    x_train_adv = attack.generate(x_attack[idx])
    return x_train_adv, idx, model


def plot_results(model, x_train, y_train, x_train_adv, num_classes):
    """ Utility function for visualizing how the attack was performed."""
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


# %% Create intuitive figures
n_classes = 2
n_samples = 1000
n_attack = 30  # number of samples to attack
x_train, y_train = generate_gaussian_parity(n_samples, cluster_std=0.25)

# KDF
model = kdf()
model.fit(x_train, y_train)
x_train_adv_kdf, idx, model = get_adversarial_examples(
    model, x_train, n_attack=n_attack, nb_classes=n_classes
)
plot_results(model, x_train[idx], y_train[idx], x_train_adv_kdf, n_classes)
plt.savefig("plots/attack_kdf.pdf")

# RF
model = rf()
model.fit(x_train, y_train)
x_train_adv_rf, idx, model = get_adversarial_examples(
    model, x_train, n_attack=n_attack, nb_classes=n_classes, idx=idx
)
plot_results(model, x_train[idx], y_train[idx], x_train_adv_rf, n_classes)
plt.savefig("plots/attack_rf.pdf")

l2_norm_kdf = np.linalg.norm(x_train[idx] - x_train_adv_kdf, ord=2)
l2_norm_rf = np.linalg.norm(x_train[idx] - x_train_adv_rf, ord=2)
print("KDF l2 norm = {:.2f}, RF l2 norm = {:.2f}".format(l2_norm_kdf, l2_norm_rf))

# %% Run a number of trials
reps = 10
n_estimators = 100
n_samples = 1000
n_classes = 2
n_attack = 50  # stub

l2s_kdf = []
l2s_rf = []

for ii in range(reps):
    X, y = generate_gaussian_parity(n_samples, cluster_std=0.25)

    model_kdf = kdf(kwargs={"n_estimators": n_estimators})
    model_kdf.fit(X, y)
    x_train_adv_kdf, idx, model_kdf = get_adversarial_examples(
        model_kdf, X, n_attack=n_attack, nb_classes=n_classes
    )

    model_rf = rf(n_estimators=n_estimators)
    model_rf.fit(X, y)
    x_train_adv_rf, idx, model_rf = get_adversarial_examples(
        model_rf, X, n_attack=n_attack, nb_classes=n_classes, idx=idx
    )

    l2_norm_kdf = np.linalg.norm(x_train[idx] - x_train_adv_kdf, ord=2)
    l2_norm_rf = np.linalg.norm(x_train[idx] - x_train_adv_rf, ord=2)
    l2s_kdf.append(l2_norm_kdf)
    l2s_rf.append(l2_norm_rf)
    print("KDF l2 norm: {:.2f}, RF l2 norm: {:.2f}".format(l2_norm_kdf, l2_norm_rf))

l2s_mean_kdf = np.mean(l2s_kdf)
l2s_std_kdf = np.std(l2s_kdf)
l2s_mean_rf = np.mean(l2s_rf)
l2s_std_rf = np.std(l2s_rf)

print("KDF l2 mean: {:.2f}, std: {:.2f}".format(l2s_mean_kdf, l2s_std_kdf))
print("RF l2 mean: {:.2f}, std: {:.2f}".format(l2s_mean_rf, l2s_std_rf))
