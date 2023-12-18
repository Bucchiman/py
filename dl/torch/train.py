#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	./src/train
# Author: 8ucchiman
# CreatedDate:  2023-02-17 15:01:51 +0900
# LastModified: 2023-02-17 15:57:04 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


class Train(object):
    def __init__(self, dataloader: DataLoader, model: nn.Module, **params):
        self.dataloader = dataloader
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.epochs = 100

    def running(self):
        """
        """
        for epoch in range(self.epochs):
            running_loss = 0.0

            for i, datas in enumerate(self.dataloader):
                inputs = datas["image"]
                labels = datas["label"]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 500 == 499:
                    print(f'{epoch+1}/{i+1:5d} loss: {running_loss / 200:.3f}')


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
