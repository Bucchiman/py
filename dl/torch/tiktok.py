#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	tiktok
# Author: 8ucchiman
# CreatedDate:  2023-02-24 15:06:36 +0900
# LastModified: 2023-02-24 15:32:49 +0900
# Reference: https://www.mattari-benkyo-note.com/2021/03/21/pytorch-cuda-time-measurement/
#


import os
import sys
import torch
from time import time
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


class TikTok(object):
    """
        torchのGPU計算時間測定
        reference: https://www.mattari-benkyo-note.com/2021/03/21/pytorch-cuda-time-measurement/
    """

    def __init__(self, level: str = "simple"):
        self.level = level
        if self.level == "simple":
            self.start = 0.0
            self.end = 0.0
        elif self.level == "difficult":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
        else:
            pass

    def start(self):
        if self.level == "simple":
            torch.cuda.synchronize()
            self.start = time()
        else:
            self.start.record()

    def end(self):
        if self.level == "simple":
            torch.cuda.synchronize()
            self.end = time()
        else:
            self.end.record()
            torch.cuda.synchronize()

    def elapsed_time(self) -> float:
        """
            測定結果
        """
        if self.level == "simple":
            elapse_time = self.end - self.start
        else:
            elapse_time = self.start.elapsed_time(self.end)
        return elapse_time


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
