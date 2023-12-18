#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	svm
# Author: 8ucchiman
# CreatedDate:  2023-02-07 16:00:02 +0900
# LastModified: 2023-02-09 17:21:39 +0900
# Reference: https://datawokagaku.com/svm/
#


import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def render_support_vector(iris, model, X_train_pca, y_train):
    fig, ax = plt.subplots(1, 1)
    DecisionBoundaryDisplay.from_estimator(model, X_train_pca, plot_method='contour',
                                           cmap=plt.cm.Paired,
                                           levels=[-1, 0, 1],
                                           alpha=0.5,
                                           linestyles=['--', '-', '--'],
                                           xlabel='first principal component',
                                           ylabel='second principal component',
                                           ax=ax)

    for i, color in zip(model.classes_, 'bry'):
        idx = np.where(y_train == i)
        ax.scatter(X_train_pca[idx, 0],
                   X_train_pca[idx, 1],
                   c=color,
                   label=iris.target_names[i],
                   edgecolor='black',
                   s=20,)

    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=100,
               linewidth=1,
               facecolors='none',
               edgecolors='k')
    fig.savefig("svm.pdf")
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    print(X_train_pca.shape)
    model = SVC()
    model.fit(X_train_pca, y_train)
    X_test_pca = pca.transform(X_test)
    y_pred = model.predict(X_test_pca)
    print(accuracy_score(y_test, y_pred))

    print(model.support_vectors_)
    render_support_vector(iris, model, X_train_pca, y_train)


if __name__ == "__main__":
    main()
