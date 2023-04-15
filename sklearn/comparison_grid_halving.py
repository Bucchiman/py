#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	comparison_grid_halving
# Author: 8ucchiman
# CreatedDate:  2023-02-08 22:16:22 +0900
# LastModified: 2023-02-08 22:26:50 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV

# import utils
# from import utils import get_args, get_logger


def make_heatmap(fig, ax, gs, Cs, gammas, is_sh=False, make_cbar=False):
    """Helper to make a heatmap."""
    results = pd.DataFrame(gs.cv_results_)
    results[["param_C", "param_gamma"]] = results[["param_C", "param_gamma"]].astype(
        np.float64
    )
    if is_sh:
        # SH dataframe: get mean_test_score values for the highest iter
        scores_matrix = results.sort_values("iter").pivot_table(
            index="param_gamma",
            columns="param_C",
            values="mean_test_score",
            aggfunc="last",
        )
    else:
        scores_matrix = results.pivot(
            index="param_gamma", columns="param_C", values="mean_test_score"
        )

    im = ax.imshow(scores_matrix)

    ax.set_xticks(np.arange(len(Cs)))
    ax.set_xticklabels(["{:.0E}".format(x) for x in Cs])
    ax.set_xlabel("C", fontsize=15)

    ax.set_yticks(np.arange(len(gammas)))
    ax.set_yticklabels(["{:.0E}".format(x) for x in gammas])
    ax.set_ylabel("gamma", fontsize=15)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if is_sh:
        iterations = results.pivot_table(
            index="param_gamma", columns="param_C", values="iter", aggfunc="max"
        ).values
        for i in range(len(gammas)):
            for j in range(len(Cs)):
                ax.text(
                    j,
                    i,
                    iterations[i, j],
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=20,
                )

    if make_cbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        cbar_ax.set_ylabel("mean_test_score", rotation=-90, va="bottom", fontsize=15)


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    rng = np.random.RandomState(0)
    X, y = make_classification(n_samples=1000, random_state=rng)

    gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    Cs = [1, 10, 100, 1e3, 1e4, 1e5]
    param_grid = {"gamma": gammas, "C": Cs}

    clf = SVC(random_state=rng)

    tic = time()
    gsh = HalvingGridSearchCV(
        estimator=clf, param_grid=param_grid, factor=2, random_state=rng
    )
    gsh.fit(X, y)
    gsh_time = time() - tic

    tic = time()
    gs = GridSearchCV(estimator=clf, param_grid=param_grid)
    gs.fit(X, y)
    gs_time = time() - tic
    fig, axes = plt.subplots(ncols=2, sharey=True)
    ax1, ax2 = axes

    make_heatmap(fig, ax1, gsh, Cs, gammas, is_sh=True)
    make_heatmap(fig, ax2, gs, Cs, gammas, make_cbar=True)

    ax1.set_title("Successive Halving\ntime = {:.3f}s".format(gsh_time), fontsize=15)
    ax2.set_title("GridSearch\ntime = {:.3f}s".format(gs_time), fontsize=15)
    plt.show()


if __name__ == "__main__":
    main()
