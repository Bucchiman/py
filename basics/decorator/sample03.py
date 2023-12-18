#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     sample03
# Author:       8ucchiman
# CreatedDate:  2023-05-11 17:40:30
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


def deco_html(func):
    def wrapper(*args, **kwargs):
        res = '<html>'
        res += func(*args, **kwargs)
        res += '</html>'
        return res
    return wrapper

def deco_body(func):
    def wrapper(*args, **kwargs):
        res = '<body>'
        res += func(*args, **kwargs)
        res += '</body>'
        return res
    return wrapper

@deco_html
@deco_body
def test():
    return "Hello Bucchiman"


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(test())
    pass


if __name__ == "__main__":
    main()

