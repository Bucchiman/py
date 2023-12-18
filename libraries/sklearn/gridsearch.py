#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	gridsearch
# Author: 8ucchiman
# CreatedDate:  2023-02-08 21:34:14 +0900
# LastModified: 2023-02-08 21:43:09 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    iris = load_iris()
    parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
    svc = SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(iris.data, iris.target)
    print(sorted(clf.cv_results_.keys()))


if __name__ == "__main__":
    main()
