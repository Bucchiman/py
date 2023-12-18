#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	binary_search
# Author: 8ucchiman
# CreatedDate:  2023-02-23 21:18:16 +0900
# LastModified: 2023-02-25 00:04:43 +0900
# Reference: https://qiita.com/drken/items/97e37dd6143e33a64c8c
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd



class BinarySearch(object):
    @classmethod
    def casual_binary_search(datas: list[int], target: int):
        """
            [left_idx, right_idx]
        """
        left_idx = 0
        right_idx = len(datas) - 1
        while left_idx <= right_idx:
            mid_idx = (left_idx+right_idx) // 2
            guess = datas[mid_idx]
            if guess == target:
                return mid_idx
            elif guess < target:
                left_idx = mid_idx + 1
            else:
                right_idx = mid_idx - 1

        return None

    @classmethod
    def meguru_binary_search(datas: list[int], target: int):
        """
            [left_idx, right_idx)
        """
        left_idx = 0
        right_idx = len(datas)
        while right_idx - right_idx > 1:
            mid_idx = (left_idx + right_idx) / 2
            guess = datas[mid_idx]
            if guess < target:
                left_idx = mid_idx
            else:
                right_idx = mid_idx





def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    my_list = [1, 3, 5, 7, 9]
    print(BinarySearch.casual_binary_search(my_list, 3))
    pass


if __name__ == "__main__":
    main()
