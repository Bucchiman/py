#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	visualize02
# CreatedDate:  2023-01-19 13:43:17 +0900
# LastModified: 2023-02-08 16:01:33 +0900
#


import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_args02


def main():
    args = get_args02()
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Paired')
    cm_bar = plt.get_cmap("Wistia")
    color_maps_bar = [cm_bar(0.1), cm_bar(0.3), cm_bar(0.5), cm_bar(0.9)]
    cm_text = plt.get_cmap("Blues")
    color_maps_text = [cm_text(0.2), cm_bar(0.4)]

    x = np.array(args.columns)
    x_position = np.arange(len(x))
    y_shi = np.array(args.shi)
    y_dla = np.array(args.dla)
    y_eye = np.array(args.eye)
    y_dpt = np.array(args.dpt)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y_shi, width=0.2, label='shi', color=color_maps_bar[0])
    ax.bar(x_position+0.2, y_dla, width=0.2, label='dla', color=color_maps_bar[1])
    ax.bar(x_position+0.4, y_eye, width=0.2, label='eye', color=color_maps_bar[2])
    ax.bar(x_position+0.6, y_dpt, width=0.2, label='dpt', color=color_maps_bar[3])
    ax.legend()
    ax.set_xticks(x_position+0.3)
    label_name = ""
    label_color = ""
    if args.fitness == "latency":
        label_name = "latency(cycles)"
        label_color = "darkviolet"
    else:
        label_name = "energy(nJ)"
        label_color = "firebrick"
    ax.set_ylabel(label_name, color=label_color, fontsize=20)
    #ax.set_xlabel("constraint_dataflow")
    ax.set_xticklabels(x, fontsize=20)
    #plt.xticks(rotation=30)
    plt.xticks()
    ax.set_ylim(0, args.max_lim)
    plt.tight_layout()
    plt.legend(fontsize=15)
    #plt.show()
    fig.savefig("./outputs/{}_{}_dataflow.png".format(args.model, args.fitness))


if __name__ == "__main__":
    main()
