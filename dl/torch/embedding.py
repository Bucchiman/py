#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	embedding
# Author: 8ucchiman
# CreatedDate:  2023-02-16 12:06:33 +0900
# LastModified: 2023-02-17 11:42:34 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import torch
from torch import nn
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    torch.manual_seed(43)
    embedding = nn.Embedding(10, 5)    # id 0~9で5行の配列を取り出せる
    print("Embedding Weight is here.\n{}".format(embedding.weight))
    print("Index 00\n{}".format(embedding(torch.tensor([0]))))



if __name__ == "__main__":
    main()
