#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	transform
# Author: 8ucchiman
# CreatedDate:  2023-02-17 14:00:45 +0900
# LastModified: 2023-02-17 14:56:54 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from torchvision import transforms
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


class Transform(object):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),

        ])

    def get_transform(self):
        return self.transform


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
