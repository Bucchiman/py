#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	logisticregression
# Author: 8ucchiman
# CreatedDate:  2023-02-09 17:34:42 +0900
# LastModified: 2023-02-09 17:37:39 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    print(clf.predict(X[:2, :]))
    print(clf.predict_proba(X[:2, :]))
    print(clf.score(X, y))


if __name__ == "__main__":
    main()
