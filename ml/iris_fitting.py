#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	iris_fitting
# Author: 8ucchiman
# CreatedDate:  2023-02-06 15:13:14 +0900
# LastModified: 2023-02-06 15:19:53 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
# import utils
# from import utils import get_args, get_logger


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    X, y = load_iris(as_frame=True, return_X_y=True)
    sepal_cols = ["sepal length (cm)", "sepal width (cm)"]
    petal_cols = ["petal length (cm)", "petal width (cm)"]
    preprocessor = ColumnTransformer(
        [
            ("scaler", StandardScaler(), sepal_cols),
            ("kbin", KBinsDiscretizer(encode="ordinal"), petal_cols),
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    X_out = preprocessor.fit_transform(X)
    print(X_out.sample(n=5, random_state=0))



if __name__ == "__main__":
    main()
