#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	classifier
# Author: 8ucchiman
# CreatedDate:  2023-02-10 17:17:07 +0900
# LastModified: 2023-02-10 17:25:22 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import seaborn as sns
from sklearn.model_selection import train_test_split
import lightgbm as lgb
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    datas = sns.load_dataset("titanic")
    X = datas.drop(['Embarked', 'PassengerId'], axis=1)
    y = datas.Embarked
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = lgb.LGBMClassifier(learning_rate=0.09, max_depth=-5, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)],
              verbose=20, eval_metric='logloss')



if __name__ == "__main__":
    main()
