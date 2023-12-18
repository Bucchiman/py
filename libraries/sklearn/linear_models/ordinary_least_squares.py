#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     ordinary_least_squares
# Author:       8ucchiman
# CreatedDate:  2023-10-18 14:32:37
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    reg = LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    print(reg.coef_)
    pass


if __name__ == "__main__":
    main()
