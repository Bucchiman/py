#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     utils
# Author:       8ucchiman
# CreatedDate:  2023-10-18 16:15:04
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest


def main():
    iris = load_iris()
    pipe = Pipeline(steps=[
        ('select', SelectKBest(k=2)),
        ('clf', LogisticRegression())
    ])
    print(pipe.fit(iris.data, iris.target))


if __name__ == "__main__":
    main()
