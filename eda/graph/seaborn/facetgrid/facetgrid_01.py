#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	facetgrid_01
# Author: 8ucchiman
# CreatedDate:  2023-02-15 10:36:05 +0900
# LastModified: 2023-02-15 11:17:55 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    tips = sns.load_dataset("tips")
    print(type(tips))
    grid = sns.FacetGrid(tips, col="time", row="sex")
    grid.map_dataframe(sns.scatterplot, "total_bill", "tip")
    plt.show()


if __name__ == "__main__":
    main()
