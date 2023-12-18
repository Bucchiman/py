#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	columntransformer
# Author: 8ucchiman
# CreatedDate:  2023-02-06 15:30:52 +0900
# LastModified: 2023-02-06 15:49:36 +0900
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
#


import os
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
# import utils
# from import utils import get_args, get_logger


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    X = np.array([[0., 1., 2., 2.],
                  [1., 1., 0., 1.]])
    norm01 = ("norm01", Normalizer(norm='l1'), [0, 1])
    norm02 = ("norm02", Normalizer(norm='l1'), slice(2, 4))

    ct01 = ColumnTransformer(
        [norm01]
    )
    print(ct01.fit_transform(X))    # -> [[0., 1.], [0.5, 0.5]]

    ct02 = ColumnTransformer(
        [norm02]
    )
    print(ct02.fit_transform(X))    # -> [[0.5, 0.5], [0., 1.]]

    ctboth = ColumnTransformer(
        [norm01, norm02]
    )
    print(ctboth.fit_transform(X))  # -> [[0., 1., 0.5, 0.5], [0.5, 0.5, 0., 1.]]

    X = pd.DataFrame({
        "documents": ["First item", "second one here", "Is this the last?"],
        "width": [3, 4, 5]
    })
    ct = ColumnTransformer(
        [
            ("text_preprocess", FeatureHasher(input_type="string"), "documents"),
            ("num_preprocess", MinMaxScaler(), ["width"])
        ]
    )
    X_trans = ct.fit_transform(X)


if __name__ == "__main__":
    main()
