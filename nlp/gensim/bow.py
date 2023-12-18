#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	bow
# Author: 8ucchiman
# CreatedDate:  2023-03-28 14:52:00 +0900
# LastModified: 2023-03-28 14:59:18 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from gensim.corpora import Dictionary
from gensim import matutils as mtu
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


def bow(morphemes):
    dct = Dictionary()
    for line in morphemes:
        dct.add_documents([line.split()])

    word2id = dct.token2id
    print(word2id)

    bow_set = []

    for line in morphemes:
        bow_format = dct.doc2bow(line.split())
        bow_set.append(bow_format)

        print(line)
        print("BoW format: (word ID, word frequency)")
        print(bow_format)

        bow = mtu.corpus2dense([bow_format], num_terms=len(dct)).T[0]
        print("BoW")
        print(bow)
        print(bow.tolist())
        print(list(map(int, bow.tolist())))


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    morphemes = ['私 は ラーメン が 好き です 。',
             '私 は 餃子 が 好き です 。',
             '私 は ラーメン が 嫌い です 。']
    bow(morphemes)
    pass


if __name__ == "__main__":
    main()
