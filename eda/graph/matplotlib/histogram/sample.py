#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	sample
# Author: 8ucchiman
# CreatedDate:  2023-02-09 17:03:57 +0900
# LastModified: 2023-02-09 17:04:49 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd



def simple_histogram():
    mu, sigma = 115, 15
    x = mu + sigma * np.random.randn(10000)
    fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    # the histogram of the data
    n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.75)

    ax.set_xlabel('Length [cm]')
    ax.set_ylabel('Probability')
    ax.set_title('Aardvark lengths\n (not really)')
    ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
    ax.axis([55, 175, 0, 0.03])
    ax.grid(True)
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    simple_histogram()
    


if __name__ == "__main__":
    main()
