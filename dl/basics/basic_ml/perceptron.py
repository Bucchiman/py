#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     perceptron
# Author:       8ucchiman
# CreatedDate:  2023-05-08 14:40:47
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt


def xy_visualize(X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray):
    '''
        2変数における分布をラベルで色分けする関数
    '''
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='class 0', marker='o')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='class 1', marker='s')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.legend()
    plt.show()
    pass


def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)


class Perceptron():
    def __init__(self, num_features, device):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1,
                                   dtype=torch.float32, device=device)
        self.bias = torch.zeros(1, dtype=torch.float32, device=device)

    def forward(self, x):
        linear = torch.add(torch.mm(x, self.weights), self.bias)
        predictions = custom_where((linear > 0.).int(), 1, 0).float()
        return predictions

    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors

    def train(self, x, y, epochs):
        for e in range(epochs):

            for i in range(y.size()[0]):
                # use view because backward expects a matrix (i.e., 2D tensor)
                errors = self.backward(x[i].view(1, self.num_features), y[i]).view(-1)
                self.weights += (errors * x[i]).view(self.num_features, 1)
                self.bias += errors

    def evaluate(self, x, y):
        predictions = self.forward(x).view(-1)
        accuracy = torch.sum(predictions == y).float() / y.size()[0]
        return accuracy


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    data = np.genfromtxt("../datas/perceptron_toydata.txt", delimiter='\t')
    X, y = data[:, :2], data[:, 2]
    y = y.astype(np.int64)
    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X, y = X[shuffle_idx], y[shuffle_idx]

    X_train, X_test = X[shuffle_idx[:70]], X[shuffle_idx[70:]]
    y_train, y_test = y[shuffle_idx[:70]], y[shuffle_idx[70:]]

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)

    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # xy_visualize(X_train, X_test, y_train, y_test)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppn = Perceptron(num_features=2, device=device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    ppn.train(X_train_tensor, y_train_tensor, epochs=5)

    #X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    #y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    #test_acc = ppn.evaluate(X_test_tensor, y_test_tensor)


if __name__ == "__main__":
    main()
