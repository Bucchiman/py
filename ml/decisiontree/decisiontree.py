#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	decisiontree
# Author: 8ucchiman
# CreatedDate:  2023-02-06 16:15:49 +0900
# LastModified: 2023-02-14 15:15:24 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    X, y = load_iris(return_X_y=True)
    print(X)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    plot_tree(clf)
    plt.show()


if __name__ == "__main__":
    main()
