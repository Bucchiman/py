#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	featurehasher
# Author: 8ucchiman
# CreatedDate:  2023-02-06 15:50:19 +0900
# LastModified: 2023-02-06 15:56:12 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.feature_extraction import FeatureHasher
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    h = FeatureHasher(n_features=10)
    D = [
        {'dog': 1, 'cat': 2, 'elephant': 4},
        {'dog': 2, 'run': 5}
    ]
    f = h.transform(D)
    print(f.toarray())              # -> [*, *, *..., *],
                                    #    [*, *, *..., *]  2x10 matrix

    h = FeatureHasher(n_features=8, input_type="string")
    raw_X = [["dog", "cat", "snake"], ["snake", "dog"], ["cat", "bird"]]
    f = h.transform(raw_X)
    print(f.toarray())
    


if __name__ == "__main__":
    main()
