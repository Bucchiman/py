#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	calculate_pmi
# Author: 8ucchiman
# CreatedDate:  2023-02-18 13:57:27 +0900
# LastModified: 2023-02-19 16:18:27 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
# import utils
# from import utils import get_args, get_logger
import numpy as np
# import pandas as pd


def calculate_ppmi(co_occurence_matrix: np.array, eps=1e-8):
    '''
        positive pmi
        ppmi = max(0, pmi(x, y))
        pmi(x, y) = log2(P(x, y)/P(x)P(y))
                  = log2(C(x, y)N/C(x)C(y))
    '''
    ppmi_map = np.zeros_like(co_occurence_matrix, dtype=np.float32)
    N = np.sum(co_occurence_matrix)
    vector_sum =  np.sum(co_occurence_matrix, axis=0)
    for a in range(co_occurence_matrix.shape[0]):
        for b in range(co_occurence_matrix.shape[1]):
            pmi = np.log2(co_occurence_matrix[a][b]*N/(vector_sum[a]*vector_sum[b]+eps))
            ppmi_map[a][b] = max(0, pmi)
    pass




def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
