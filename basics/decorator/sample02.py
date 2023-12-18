#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     sample02
# Author:       8ucchiman
# CreatedDate:  2023-05-11 11:12:21
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    https://qiita.com/mtb_beta/items/d257519b018b8cd0cc2e
# Description:  ---
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


def deco02(func):
    import os
    def wrapper(*args, **kwargs):
        res = '__start__' + os.linesep
        res += func(*args, **kwargs) + '!' + os.linesep
        res += '__end__'
        return res
    return wrapper

@deco02
def test02():
    return "HELLO DECOSKE"


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(test02())
    pass


if __name__ == "__main__":
    main()

#<br />
