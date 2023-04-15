#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	transition_diagram
# Author: 8ucchiman
# CreatedDate:  2023-02-28 13:20:33 +0900
# LastModified: 2023-02-28 13:31:53 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


def transition_diagram(box_bg='#CCCCCC', arrow1='#88CCFF', arrow2='#88FF88'):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    pc = Circle((4, 5), 0.7, fc=box_bg)
    uc = Circle((6, 5), 0.7, fc=box_bg)
    ax.add_patch (pc)
    ax.add_patch (uc)

    plt.text(4, 5, "Predict\nStep",ha='center', va='center', fontsize=12)
    plt.text(6, 5, "Update\nStep",ha='center', va='center', fontsize=12)

    #btm arrow from update to predict
    ax.annotate('',
                xy=(4.1, 4.5),  xycoords='data',
                xytext=(6, 4.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=pc,
                                patchA=uc,
                                connectionstyle="arc3,rad=-0.5"))
    #top arrow from predict to update
    ax.annotate('',
                xy=(6, 5.5),  xycoords='data',
                xytext=(4.1, 5.5), textcoords='data',
                size=20,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                patchB=uc,
                                patchA=pc,
                                connectionstyle="arc3,rad=-0.5"))

    ax.annotate('Measurement ($\mathbf{z_k}$)',
                xy=(6.3, 5.6),  xycoords='data',
                xytext=(6,6), textcoords='data',
                size=14,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    # arrow from predict to state estimate
    ax.annotate('',
                xy=(4.0, 3.8),  xycoords='data',
                xytext=(4.0,4.3), textcoords='data',
                size=12,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    ax.annotate('Initial\nConditions ($\mathbf{x_0}$)',
                xy=(4.05, 5.7),  xycoords='data',
                xytext=(2.5, 6.5), textcoords='data',
                size=14,
                arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none"))

    plt.text (4, 3.7,'State Estimate ($\mathbf{\hat{x}_k}$)',
              ha='center', va='center', fontsize=14)
    plt.axis('equal')
    plt.show()


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    transition_diagram()
    pass


if __name__ == "__main__":
    main()
