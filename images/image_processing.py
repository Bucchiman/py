#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	image_processing
# Author: 8ucchiman
# CreatedDate:  2023-03-02 10:35:59 +0900
# LastModified: 2023-03-02 11:18:49 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import cv2
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


class ImageProcessing(object):
    def __init__(self, img_path: str):
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR)
        pass

    def morphological_transformations(self):
        pass

    def erosion(self):
        pass

    def fourier_transformation(self):
        """
            フーリエ変換
            画像処理では[0, 255]の離散値をとり画像なのでHxWの二次元表示であるので、
            DFT(Discrete Fourier Transformation, 二次元離散フーリエ変換)を用いる。
            DFT algorithm

            G(k, l) = 1/(HW) \sum\sum I(x, y)exp(-2\pi j(kx/W+ly/H))
            ただし、k=[0, W-1], l=[0, H-1], 入力画像をIとする。
            パワースペクトルGとは複素数であらわされるので、Gの絶対値を求めることである。

            逆二次元離散フーリエ変換(IDFT: Inverse DFT)とは周波数成分Gから元の画像を復元する手法である

        """


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
