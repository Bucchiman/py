#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	bow
# Author: 8ucchiman
# CreatedDate:  2023-03-28 15:01:46 +0900
# LastModified: 2023-03-28 15:08:36 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd



def bow(corpus):
    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    bag = vectorizer.fit_transform(corpus)
    print(bag.toarray())
    print(vectorizer.vocabulary_)
    print(vectorizer.get_feature_names_out())


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    corpus = [
        "私はラーメンが好きです。",
        "私は餃子が好きです。",
        "私はラーメンが大嫌いです。"]
    tagger = MeCab.Tagger('-Owakati')
    corpus = [tagger.parse(sentence).strip() for sentence in corpus]
    bow(corpus)
    pass


if __name__ == "__main__":
    main()
