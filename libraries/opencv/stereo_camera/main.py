#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     main
# Author:       8ucchiman
# CreatedDate:  2023-06-02 14:44:43
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    imgL = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
    imgR = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)
    stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()
    pass


if __name__ == "__main__":
    main()
