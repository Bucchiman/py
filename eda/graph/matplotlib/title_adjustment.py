#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	title_adjustment
# Author: 8ucchiman
# CreatedDate:  2023-02-08 11:05:23 +0900
# LastModified: 2023-02-08 13:31:26 +0900
# Reference: https://py-memo.com/python/matplotlib-titleposition/
#


import os
import sys
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    label = ["a", "b", "c", "d", "e"]
    x, y = range(0, 5), [9, 6, 7, 8, 4]
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.bar(x, y, tick_label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # グラフの下にスペース
    fig.subplots_adjust(bottom=0.15)

    title = 'figure1:bar'
    '''
        arg01: x軸座標(0, 1)
        arg02: y軸座標(0, 1)
    '''
    fig.text(0.40, 0.02, title, fontsize=18)        # 

    plt.show()


if __name__ == "__main__":
    main()
