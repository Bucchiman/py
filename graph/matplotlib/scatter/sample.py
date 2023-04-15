#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	sample
# Author: 8ucchiman
# CreatedDate:  2023-02-09 16:59:30 +0900
# LastModified: 2023-02-09 17:01:55 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import matplotlib.pyplot as plt
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def simple_scatter():
    fig, ax = plt.subplots(figsize=(5, 2.7))
    ax.scatter([1, 2, 34, 4, 98, 2], [2, 3, 5, 1, 4, 22], facecolor='C0', edgecolor='k')
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    simple_scatter()
    


if __name__ == "__main__":
    main()
