#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	/home/ykiwabuchi/.config/snippets/codes/python/ml/pandas
# Author: 8ucchiman
# CreatedDate:  2023-02-03 16:44:18 +0900
# LastModified: 2023-02-16 13:04:04 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import pandas as pd
from typing import Any
# import utils


class MyPandas(object):
    @staticmethod
    def make_DataFrame(**kwargs):
        '''
            kwargs
                apple = [1, 2, 3]
                banana = [3.0, 2.0, 1.2]
                ->
                    apple      banana
                0     1         3.0
                1     2         2.0
                2     3         1.2
        '''
        return pd.DataFrame(kwargs)

    @staticmethod
    def extract_from_df(df: pd.DataFrame, operation: str):
        '''
            query('customer_id == "CS09"')
            query('customer_id == "CS09" and amount >= 100')
            query('customer_id == "CS09" and (amount >= 100 or quantity >=5)')
            query('customer_id == "CS09" and product_cd != "P081"')
            query("store_cd.str.startswith('S14')", engine='python')
            query("store_cd.str.startswith('S14')") これでもOK
            query("store_cd.str.endswith('1')")
            query("address.str.contains('横浜')")
            query("address.str.contains(r'^[A-F]')")        A~Fで始まるデータ
            query("address.str.contains(r'[1-9]$')")        1~9で終わるデータ
            query("address.str.contains(r'^[A-F].*[1-9]$')")
            query("address.str.contains(r'^[0-9]{3}-[0-9]{3}-[0-9]{4}'$)")      080-802-2224
        '''
        print(df.query(operation))

    @staticmethod
    def sort(df: pd.DataFrame, column: str):
        print(df.sort_values(column))

    @staticmethod
    def startswith(df: pd.DataFrame, target_str):

        operation = '{}.str.startswith("{}")'

    @staticmethod
    def contains(df: pd.DataFrame, target_col: str, target_str: str):
        '''
            r"^[1-9]"
        '''
        operation = '{}.str.contains(r"{}")'.format(target_col, target_str)
        print(df.query(operation))


    @staticmethod
    def rank(df: pd.DataFrame, target_col: str, method: str):
        pass

    @staticmethod
    def groupby_aggregate(df: pd.DataFrame, key_col: str, target_features: dict[str, Any]):
        '''
            key_colごとにdict(key)の値を,dict(value)のメソッドで集計
            key_col: col
            target_features: dict({col01: "sum", col02: ["min", "max", "median"], col03: "mean", col04: ["std", "var"]})
        '''
        print(df.groupby(key_col).agg(target_features))

    @staticmethod
    def groupby_aggregate_lambda(df: pd.DataFrame, key_col: str):
        print(df.groupby(key_col).product_cd.apply(lambda x: x.mode()))



def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(MyPandas.make_DataFrame(hoge= [1, 0], kan= [0.1, 0.56]))
    train_df = pd.read_csv("train.csv")
    print(train_df.head())
    MyPandas.contains(df=train_df, target_col="PassengerId", target_str="20")
    MyPandas.groupby_aggregate(train_df, "HomePlanet", {"Age": "mean"})


if __name__ == "__main__":
    main()
