#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	sample_plot
# Author: 8ucchiman
# CreatedDate:  2023-02-09 16:31:50 +0900
# LastModified: 2023-02-09 16:59:13 +0900
# Reference: https://matplotlib.org/stable/tutorials/introductory/quick_start.html#sphx-glr-tutorials-introductory-quick-start-py
#


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger


def myplot(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out


def simple_plot():
    x = np.linspace(0, 2, 100)
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    # fig, ax = plt.subplots(figsize=(5, 2.7), layout=None)
    ax.plot(x, x, label='linear')
    ax.plot(x, x**2, label='quadratic')
    ax.plot(x, x**3, label='cubic')
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_title("Simple plot")
    ax.legend()
    plt.show()


def style_plot():
    fig, ax = plt.subplots(figsize=(5, 2.7))
    data1 = np.random.randn(100)
    data2 = np.random.randn(100)
    x = np.arange(len(data1))
    ax.plot(x, np.cumsum(data1), color='blue', linewidth=3, linestyle='--')
    l, = ax.plot(x, np.cumsum(data2), color='orange', linewidth=2)
    l.set_linestyle(':')
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)

    # data1, data2, data3, data4 = np.random.randn(4, 100)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
    # myplot(ax1, data1, data2, {'marker': 'x'})
    # myplot(ax2, data3, data4, {'marker': 'o'})
    # plt.show()
    style_plot()


if __name__ == "__main__":
    main()
