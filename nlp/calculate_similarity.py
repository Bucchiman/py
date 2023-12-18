#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	calculate_similarity
# Author: 8ucchiman
# CreatedDate:  2023-02-18 13:07:42 +0900
# LastModified: 2023-02-18 13:13:38 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from utils import cos_similarity
from make_corpus import make_corpus
from make_co_occurence_matrix import make_co_occurence_matrix
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = make_corpus(text)
    co_occurence_matrix = make_co_occurence_matrix(corpus, len(word2id))
    vec00 = co_occurence_matrix[word2id["you"]]
    vec01 = co_occurence_matrix[word2id["i"]]
    print(cos_similarity(vec00, vec01))
    pass


if __name__ == "__main__":
    main()
