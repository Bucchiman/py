#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	bit_operate
# Author: 8ucchiman
# CreatedDate:  2023-01-29 15:12:15 +0900
# LastModified: 2023-01-29 15:14:47 +0900
# Reference: https://qiita.com/y518gaku/items/07961c61f5efef13cccc
#


import os
import sys
# import utils
# import numpy as np
# import pandas as pd


class BitOperator(object):
    def __init__(self, value):
        self.value = value

    def __and__(self, other):
        return self.value & other.value

    def __or__(self, other):
        return self.value | other.value


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    bo01 = BitOperator(1)
    bo02 = BitOperator(0)
    print(bo01 & bo02)  # 0
    print(bo01 | bo02)  # 1


if __name__ == "__main__":
    main()
