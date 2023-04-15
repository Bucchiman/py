#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	visualize
# CreatedDate:  2023-01-19 05:39:20 +0900
# LastModified: 2023-02-08 10:42:13 +0900
#


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import get_args


def main():
    args = get_args()
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Paired')

    # x = np.array(['power_shi', 'power_dla', 'power_eye',
    #               'power_dpt', 'area_shi', 'area_dla',
    #               'area_eye', 'area_dpt'])
    x = np.array(args.columns)
    x_position = np.arange(len(x))
    y_conf = np.array(args.confuciux)
    y_bayesian = np.array(args.bayesian)
    y_genetic = np.array(args.genetic)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y_conf, width=0.2, label='confuciux')
    ax.bar(x_position+0.2, y_bayesian, width=0.2, label='bayesian')
    ax.bar(x_position+0.4, y_genetic, width=0.2, label='genetic')
    ax.legend()
    ax.set_xticks(x_position+0.2)
    label_name = ""
    if args.fitness == "latency":
        label_name = "latency(cycles)"
    else:
        label_name = "energy(nJ)"
    ax.set_ylabel(label_name)
    ax.set_xlabel("constraint_dataflow")
    ax.set_xticklabels(x)
    plt.xticks(rotation=30)
    plt.ylim(min=0, max=60000000)
    plt.tight_layout()
    # plt.show()
    fig.savefig("./outputs/{}_{}.png".format(args.model, args.fitness))


if __name__ == "__main__":
    main()
