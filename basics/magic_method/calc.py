#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: calc
# Author: 8ucchiman
# CreatedDate:  2023-01-29 15:02:20 +0900
# LastModified: 2023-01-29 15:11:52 +0900
# Reference: https://qiita.com/y518gaku/items/07961c61f5efef13cccc
#


import os
import sys
# import utils
# import numpy as np
# import pandas as pd


class Calc(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value

    def __mul__(self, other):
        return self.value * other.value

    def __truediv__(self, other):
        return self.value / other.value

    def __floordiv__(self, other):
        return self.value // other.value


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    calc_val1 = Calc(10)
    calc_val2 = Calc(30)
    print("magic methods of Calc class: {}".format(dir(Calc)))
    print(calc_val1+calc_val2)
    print(calc_val1-calc_val2)
    print(calc_val1*calc_val2)
    print(calc_val1/calc_val2)
    print(calc_val1//calc_val2)


if __name__ == "__main__":
    main()
