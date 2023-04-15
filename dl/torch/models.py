#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	models
# Author: 8ucchiman
# CreatedDate:  2023-02-17 13:36:35 +0900
# LastModified: 2023-02-17 15:00:57 +0900
# Reference: https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055
#


import os
import sys
import timm
from pprint import pprint
# import utils
# from import utils import get_args, get_logger
# import numpy as np
# import pandas as pd


class TimmModel(object):
    def __init__(self, model_name: str, num_classes: int, pretrained=False):
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        pass

    def __call__(self):
        return self.model

    @staticmethod
    def show_models(regex="*", pretrained=False):
        """
        Args:
            regex : 正規表現でモデルを抽出
                    e.g. "resnet*"
            pretrained: 事前学習済みであるか
                        default=False
        """
        pprint(timm.list_models(regex, pretrained=pretrained))


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    TimmModel.show_models()
    pass


if __name__ == "__main__":
    main()
