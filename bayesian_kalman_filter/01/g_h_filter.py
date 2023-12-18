#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     g_h_filter
# Author:       8ucchiman
# CreatedDate:  2023-05-30 19:13:33
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd



def g_h_filter(data, x0, dx, g, h, dt):
    """
        data: contains the data to be filtered.
        x0  : initial value for our state variable.
        dx  : initial change rate for our state variable
        g   : g scale factor
        h   : h scale factor
        dt  : length of the time step
    """
    gain_rate = g
    prediction = x0
    esitimates = [prediction]
    predictions = []
    for z in data:
        estimate = prediction + gain_rate * (z - prediction)
        predictions.append(estimate)



def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    pass


if __name__ == "__main__":
    main()

