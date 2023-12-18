#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	decorator
# Author: 8ucchiman
# CreatedDate:  2023-01-29 15:54:38 +0900
# LastModified: 2023-02-03 22:35:46 +0900
# Reference: https://qiita.com/macinjoke/items/1be6cf0f1f238b5ba01b
#            https://www.datacamp.com/tutorial/decorators-python
#            https://qiita.com/mtb_beta/items/d257519b018b8cd0cc2e
#


import os
import sys
# import utils
# import numpy as np
# import pandas as pd


def args_logger(f):
    def wrapper(*args, **kwargs):
        f(*args, **kwargs)
        print('args: {}, kwargs: {}'.format(args, kwargs))
    return wrapper


@args_logger
def print_message(msg):
    print(msg)


@args_logger
def print_no_augments():
    print()


@args_logger
def print_some_augments(a, b, c):
    print(a, b, c)


@args_logger
def print_some_augments_kwargs(a, b, c, bucchiman="8ucchiman"):
    print(a, b, c, bucchiman)


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print_message("hello")
    print_no_augments()
    print_some_augments("here", "is", "8ucchiman")
    print_some_augments_kwargs("here", "is", "8ucchiman", bucchiman="geho")



if __name__ == "__main__":
    main()
