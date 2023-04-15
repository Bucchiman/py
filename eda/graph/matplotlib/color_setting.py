#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	color_setting
# Author: 8ucchiman
# CreatedDate:  2023-02-08 14:15:24 +0900
# LastModified: 2023-02-09 16:25:52 +0900
# Reference: https://pythondatascience.plavox.info/matplotlib/%E8%89%B2%E3%81%AE%E5%90%8D%E5%89%8D
#            https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
#            https://shikaku-mafia.com/matplotlib-color/


import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger


def specify_one_str():
    x = np.arange(1, 9)
    fig, ax = plt.subplots(1, 1)
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    height = np.repeat(1, 8)
    ax.bar(x, height, color=colorlist, tick_label=colorlist, align="center")
    plt.show()


def specify_16_bits():
    x = np.arange(1, 9)
    fig, ax = plt.subplots(1, 1)
    colorlist = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    height = np.arange(1, 9)
    fig.patch.set_facecolor("white")
    ax.barh(x, height, color=colorlist, tick_label=colorlist, align="center")
    plt.show()


def specify_gradation():
    fig, ax = plt.subplots(1, 1)
    x = np.arange(1, 6)
    height = np.arange(100, 600, 100)
    cm = plt.get_cmap("Wistia")         # https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    color_maps = [cm(0.1), cm(0.3), cm(0.5), cm(0.7), cm(0.9)]
    ax.bar(x, height, color=color_maps)
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    # specify_one_str()
    # specify_16_bits()
    pass


if __name__ == "__main__":
    main()
