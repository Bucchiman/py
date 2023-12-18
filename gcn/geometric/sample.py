#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	sample
# Author: 8ucchiman
# CreatedDate:  2023-02-14 14:39:52 +0900
# LastModified: 2023-02-14 14:44:33 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from torch_geometric.datasets import KarateClub
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    dataset = KarateClub()
    print(f'Graph: {dataset[0]}')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


if __name__ == "__main__":
    main()
