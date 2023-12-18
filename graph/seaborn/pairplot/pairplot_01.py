#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	pairplot_01
# Author: 8ucchiman
# CreatedDate:  2023-02-15 11:18:42 +0900
# LastModified: 2023-02-15 11:21:44 +0900
# Reference: https://seaborn.pydata.org/generated/seaborn.pairplot.html
#


import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def dataset_penguins():
    penguins = sns.load_dataset("penguins")
    print(penguins.head())
    sns.pairplot(penguins)
    # plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    dataset_penguins()


if __name__ == "__main__":
    main()
