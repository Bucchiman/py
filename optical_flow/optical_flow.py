#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	optical_flow
# Author: 8ucchiman
# CreatedDate:  2023-03-01 14:17:02 +0900
# LastModified: 2023-03-01 14:31:19 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import cv2
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
import numpy as np
# import pandas as pd


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    # Shi-Tomas法のパラメータは以下のように設定
    ST_param = {
        "maxCorners": 50,       # 特徴点の最大数
        "qualityLevel": 0.1,    # 特徴点を選択する閾値
        "minDistance": 7,       # 特徴点間の最小距離
        "blockSize": 7          # 特徴点の計算に使う周辺領域のサイズ
    }
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft1 = cv2.goodFeaturesToTrack(gray1, mask=None, **ST_param)

    mask = np.zeros_like(frame)

    LK_param = {
        "winSize": (15, 15), # オプティカルフローの推定計算に使う周辺領域サイズ  小さくするとノイズに敏感になり、大きな動きを見逃す可能性がある
        "maxLevel": 2,       # ピラミッド数 0の場合、ピラミッドを使用しない。ピラミッドを用いることで、画像の様々な解像度でオプティカルフローを見つけられる
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) # 繰り返しの終了条件
                                                                                # cv2.TERM_CRITERIA_EPS：指定した精度(epsilon)に到達したら計算を終了する
                                                                                # cv2.TERM_CRITERIA_COUNT：指定した繰り返しの最大回数(count)に到達したら計算を終了。
    }
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ft2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, ft1, None, **LK_param)

    if status is not None:
        g1 = ft1[status == 1]
        g2 = ft2[status == 1]
        for i, (pt2, pt1) in enumerate()zip(g2, g1)):
            x1, y1 = pt1.ravel()
            x2, y2 = pt2.ravel()

            mask = cv2.line(mask, (int(x2), int(y2)), (int(x1), int(y1)), [0, 0, 200], 2)

            frame = cv2.circle(frame, (int(x2), int(y2)), 5, [0, 0, 200], -1)

        img = cv2.add(frame, mask)


    pass


if __name__ == "__main__":
    main()
