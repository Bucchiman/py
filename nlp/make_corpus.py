#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	make_corpus
# Author: 8ucchiman
# CreatedDate:  2023-02-18 11:47:42 +0900
# LastModified: 2023-02-18 12:18:27 +0900
# Reference: deeplearning(自然言語処理)
#


import os
import sys
from typing import Tuple
# import utils
# from import utils import get_args, get_logger
import numpy as np
# import pandas as pd


def make_corpus(simple_text: str = "You say goodbye and I say hello.") -> Tuple[np.array, dict[str, int], dict[int, str]]:
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    simple_text = simple_text.lower()
    simple_text = simple_text.replace(".", " .")
    simple_text = simple_text.split()
    words = list(set(simple_text))
    word2id = {}
    id2word = {}
    for i, word in enumerate(words):
        word2id[word] = i
        id2word[i] = word
    corpus = [word2id[word] for word in simple_text]
    corpus = np.array(corpus)
    return corpus, word2id, id2word
    pass


def main():
    print(make_corpus())
    pass


if __name__ == "__main__":
    main()
