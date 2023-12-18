#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	transform
# Author: 8ucchiman
# CreatedDate:  2023-02-21 10:59:49 +0900
# LastModified: 2023-02-21 11:01:45 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer


class Transform(object):
    """
        変換
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        self.train_df = train_df
        self.test_df = test_df

    def standard_scaler(self, features: list[str]):
        """
            標準化(平均0, 標準偏差1)
            x' = (x-μ)/σ

            e.g. 線形回帰, ロジスティック回帰ではスケールが大きいほど回帰係数は小さくなる
                 ニューラルネットワークでも同様

            * 0, 1のような二値変数の場合、0, 1の割合が偏っている場合には標準偏差が小さいため、
              二値変数では標準化しなくてよい

        """
        scaler = StandardScaler()
        scaler.fit(self.df[features])
        self.train_df[features] = scaler.transform(self.train_df[features])
        self.test_df[features] = scaler.transform(self.test_df[features])
        pass

    def min_max_scaler(self, features: list[str]):
        """
            x' = (x-x_{min})/(x_{max}-x_{min})
            変換後の平均がちょうど0にならない、外れ値の影響を受けることからStandarScalerのほうがより一般的
            * 画像では0~255ともともと範囲が決まっている変数なのでMin-Maxスケーリングが用いられる
        """
        scaler = MinMaxScaler()
        scaler.fit(self.train_df[features])
        self.train_df[features] = scaler.transform(self.train_df[features])
        self.test_df[features] = scaler.transform(self.test_df[features])
        pass

    def log_transform(self):
        """
            * 非線形変換
            一方向に裾が伸びた分布(金額、カウント)では対数変換が有効
            発散を避けるため log(x+1)変換がよく用いられる(np.log1p)
            負の値についてlog(x)は適応できないのでlog(|x|) -> np.sign(x)*log(|x|)

        """
        pass

    def power_transform(self, features: list[str], method: str = "box-cox"):
        """
            Box-Cox

                   - (x^λ-1)/λ if λ not 0
            x^λ =
                   - log(x)
            Args
                features: 正の値をとるカラムのみ有効
                method: "box-cox"

            Yeo-Johnson
            Args
                features:
                method: "yeo-johnson"

        """
        pt = PowerTransformer(method=method)
        pt.fit(self.train_df[features])

        self.train_df[features] = pt.transform(self.train_df[features])
        self.test_df[features] = pt.transform(self.test_df[features])


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
