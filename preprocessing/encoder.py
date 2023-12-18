#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	encoder
# Author: 8ucchiman
# CreatedDate:  2023-02-21 10:42:51 +0900
# LastModified: 2023-02-21 11:17:34 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LabelEncoder


class Encoder(object):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, index: str):
        self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.index = index
        pass


    def label_encoder(self, columns: list[str]):
        '''
            カテゴリカラムについての自動的ラベルづけ
            columns: カテゴリカラム
        '''
        for col in columns:
            self.train_df[col] = self.train_df[col].astype(str)
            self.test_df[col] = self.test_df[col].astype(str)
            self.train_df[col] = LabelEncoder().fit_transform(self.train_df[col])
            self.test_df[col] = LabelEncoder().fit_transform(self.test_df[col])


    def frequency_encoder(self, columns: list[str]):
        for col in columns:
            freq = self.train_df[col].value_counts()
            self.train_df[col] = self.train_df[col].map(freq)
            self.test_df[col] = self.test_df[col].map(freq)
        pass

    def target_encoder(self, columns: list[str]):
        '''
            単純にデータ全体から平均をとってしまうと、自身のレコードの
            目的変数をカテゴリ変数に取り込んでしまう(これをリークと呼ぶ)
            -> 自身のレコードの目的変数を使わないで変換
            method
                学習データをtarget encoding用にfold分割
                   - out-of-hold
                   - k fold cross validation

            単純に全体のデータから平均をとった場合
            e.g. ある水準に所属するレコードが1つしかなかった場合、
                 その水準に対するtarget encodingの結果は目的変数の値そのものになるから
                 各レコードでユニークな値(IDのような)列に対してtarget encodingを適用した場合
                 こうした場合、目的変数の列に完全に一致してしまうため、
        '''
        for col in columns:
            data_tmp = pd.DataFrame({col: self.train_df[col], "target": self.target})
            target_mean = data_tmp.groupby(col)['target'].mean()
            self.test_df[col] = self.test_df[col].map(target_mean)

            tmp = np.repeat(np.nan, self.train_df.shape[0])

            kf = KFold(n_splits=4, shuffle=True, random_state=43)
            for idx_1, idx_2 in kf.split(self.train_df):
                target_mean = data_tmp.iloc[idx_1].groupby(col)['target'].mean()
                tmp[idx_2] = self.train_df[col].iloc[idx_2].map(target_mean)

            self.train_df[col] = tmp


    def embedding(self, columns: list[str]):
        '''
            単語やカテゴリ変数のような離散的な表現を、実数ベクトルに変換する方法(分散表現)
            カテゴリ変数が多い場合、one-hot encodingでは情報を十分にとらえきれない場合がある
            e.g.
                
                A1  -->   0.200  0.469  0.019
                A2  -->   0.115  0.343  0.711
                A3  ...
                B1
                B2
                A1
                A2
                A1
        '''
        pass



def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    
    pass


if __name__ == "__main__":
    main()
