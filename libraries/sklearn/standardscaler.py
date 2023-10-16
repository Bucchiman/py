#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	standardscaler
# Author: 8ucchiman
# CreatedDate:  2023-02-06 22:33:30 +0900
# LastModified: 2023-02-06 23:32:08 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from sklearn.preprocessing import StandardScaler
# import utils
# from import utils import get_args, get_logger
import numpy as np


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    scaler = StandardScaler()
    # X = np.arange(15).reshape(3, 5)
    X = [[0, 0, 0], [0, 0, 0], [1, 1, 2], [1, 1, 1]]
    scaler.fit(X)
    print(scaler.mean_)         # [(0+0+1+1)/4=0.5, (0+0+1+1)/4=0.5, (0+0+2+1)/4=0.75]
    print(scaler.var_)          # [((0-0.5)**2+(0-0.5)**2+(1-0.5)**2+(1-0.5)**2)**(1/2)/4=0.25,
                                #  0.25,
                                # ((0-0.75)**2+(0-0.75)**2+(2-0.75)**2+(1-0.75)**2)**(1/2)/4]
    print(scaler.transform(X))



if __name__ == "__main__":
    main()
