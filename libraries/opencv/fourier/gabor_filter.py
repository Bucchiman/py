#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     gabor_filter
# Author:       8ucchiman
# CreatedDate:  2023-07-27 13:18:37
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    https://cppx.hatenablog.com/entry/2017/12/02/170204
# Description:  ---
#


import os
import sys
import cv2 as cv
import numpy as np
import pylab
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(sys.argv)
    img = cv.imread(sys.argv[1])    # pathを間違えるとNoneが返ってきます
    gabor = cv.getGaborKernel((30, 30), 4.0, np.radians(0), 10, 0.5, 0)
    dst = cv.filter2D(img, -1, gabor)
    pylab.imshow(dst) and pylab.show()



if __name__ == "__main__":
    main()
