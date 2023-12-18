#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     sample01
# Author:       8ucchiman
# CreatedDate:  2023-05-11 11:01:10
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


def deco(func):
    def wrapper(*args, **kwargs):
        print('--start--')
        func(*args, **kwargs)
        print('--end--')
    return wrapper


@deco
def test():
    print('8ucchiman was here!')


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    test()
    pass


if __name__ == "__main__":
    main()

