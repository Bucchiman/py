#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	make_co_occurence_matrix
# Author: 8ucchiman
# CreatedDate:  2023-02-18 12:06:39 +0900
# LastModified: 2023-02-19 15:40:29 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from make_corpus import make_corpus
# import utils
# from import utils import get_args, get_logger
import numpy as np
# import pandas as pd


def make_co_occurence_matrix(corpus: np.array, vocab_size: int, window_size: int = 1) -> np.array:
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    num_words = corpus.shape[0]
    co_occurence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int8)
    for idx, word_idx in enumerate(corpus):
        for ws in range(1, window_size+1):
            lidx = idx - ws
            ridx = idx + ws
            if lidx >= 0:
                co_occurence_matrix[word_idx][corpus[lidx]] += 1
            if ridx < num_words:
                co_occurence_matrix[word_idx][corpus[ridx]] += 1
    return co_occurence_matrix


def main():
    text = "You say goodbye and I say hello."
    corpus, word2id, id2word = make_corpus(text)
    co_occurence_matrix = make_co_occurence_matrix(corpus, len(word2id))
    print(word2id)
    print(np.sum(co_occurence_matrix))
    print(np.sum(co_occurence_matrix, axis=0))
    pass


if __name__ == "__main__":
    main()
