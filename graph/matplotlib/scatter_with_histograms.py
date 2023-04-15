#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	scatter_with_histograms
# Author: 8ucchiman
# CreatedDate:  2023-02-08 20:55:45 +0900
# LastModified: 2023-02-08 21:05:26 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
# import utils
# from import utils import get_args, get_logger
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Draw the scatter plot and marginals.
    scatter_hist(x, y, ax, ax_histx, ax_histy)
    plt.show()


def scatter_hist(x, y, ax, ax_histx, ax_histy):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(x, y)
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth)+1) * binwidth
    bins = np.arange(-lim, lim+binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation="horizontal")





if __name__ == "__main__":
    main()
