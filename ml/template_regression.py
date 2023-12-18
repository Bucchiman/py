#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	template_regression
# Author: 8ucchiman
# CreatedDate:  2023-02-06 14:50:50 +0900
# LastModified: 2023-02-06 14:59:42 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


class TemplateModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    pass


if __name__ == "__main__":
    main()
