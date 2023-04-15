#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	completion
# Author: 8ucchiman
# CreatedDate:  2023-02-21 11:03:50 +0900
# LastModified: 2023-02-21 11:16:13 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from typing import Any
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
import pandas as pd
from sklearn.impute = SimpleImputer


class Completion(object):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, index: str):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.index = index
        pass

    def identity(self, features: list[str]):
        """
            欠損値のまま扱う
            何らかの理由で欠損している場合、特徴量として使えるためその情報を捨てるのはもったいない
            * ランダムフォレストでは欠損値を扱えない
              その場合、欠損値に-9999などの通常とりえる範囲外の値を代入すればよい
              こうした処理ができるのはその値自体に意味があるのではなく、相対的な値の比較によって木が
              学習されるからである
        """
        pass

    def completion_typical_value(self, features: list[str], method: str):
        """
            平均: 正規分布
            中央値: 年収などの歪みがある分布
            others: 対数変換後、歪みの少ない分布にしてから平均値をとる
                    全データの平均にする必要はない 別カテゴリ変数の値でグループ分けしたのち平均とするのも可能
                                                   この場合、データ数が少なければ平均値が信用できない場合もある
                                                   この場合、Bayesian averageという手法がある
            最頻値: カテゴリ変数の場合
        """
        pass

    def predict_missing_col_from_others(self):
        """
            欠損値があるカラムについて欠損値なしのカラムを元に予測する
        """
        pass

    def create_feature_from_missing_values(self):
        """
            欠損値である理由が何かしらある場合、その情報を捨てるのはもったいないので
            欠損があるかどうかを新たにカラムに付与すればよい
            欠損しているカラムを補完した場合でもその情報を残すことができる
        """
        pass

    def imputer(self, columns: list[str], strategy="mean"):
        """
            Imputerで一括補完
        """
        imputer = SimpleImputer(strategy)
        imputer.fit(self.train_df[columns])
        self.train_df[columns] = imputer.transform(self.train_df[columns])
        self.test_df[columns] = imputer.transform(self.test_df[columns])

    def fillna(self, column: str, completion: Any):
        """
            各カラムで補完
            Args
                completion "Z"
                           self.train_df[column].mode()
        """
        self.train_df[column].fillna(completion, inplace=True)


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
