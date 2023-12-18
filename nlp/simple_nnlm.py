#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	simple_nnlm
# Author: 8ucchiman
# CreatedDate:  2023-02-16 11:07:11 +0900
# LastModified: 2023-02-17 11:39:04 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import torch
from torch import nn
from torch import optim
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


def make_batch(sentences, word_dict):
    '''
        make problem
        文章の一番最後の単語を予測
        input_batch: 全てのsentence
                     [[1, 2, ],
                      [3, 1, ],
                      [3, 0, ]]
        target_batch: 全てのsentenceに関するmask単語
    '''
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch


class NNLM(nn.Module):
    def __init__(self, n_class):
        super(NNLM, self).__init__()
        self.n_step = 2
        self.n_hidden = 2
        self.m = 2
        self.n_class = n_class

        self.C = nn.Embedding(self.n_class, self.m)
        self.H = nn.Linear(self.n_step*self.m, self.n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(self.n_hidden))
        self.U = nn.Linear(self.n_hidden, self.n_class, bias=False)
        self.W = nn.Linear(self.n_step*self.m, self.n_class, bias=False)
        self.b = nn.Parameter(torch.ones(self.n_class))

    def forward(self, X: torch.LongTensor):
        print("input size:{}".format(X.size()))     # [3, 2]
        X = self.C(X)
        print("C:{}".format(self.C.weight.size()))  # [7, 2]
        print("embedded size:{}".format(X.size()))
        X = X.view(-1, self.n_step*self.m)
        tanh = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(tanh)
        return output


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)

    model = NNLM(n_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch(sentences, word_dict)

    # 64bit型のtensorに変換
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])


if __name__ == "__main__":
    main()
