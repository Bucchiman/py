#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     sample04
# Author:       8ucchiman
# CreatedDate:  2023-05-11 17:46:47
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



def deco_p(func):
    def wrapper(*args, **kwargs):
        res = '<p>'
        res += func(args[0], **kwargs)
        res += '</p>'
        return res
    return wrapper


@deco_p
def test(str):
    return str


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(test("hello bucchiman"))
    pass


if __name__ == "__main__":
    main()

