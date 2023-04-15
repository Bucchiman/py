#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	learn
# Author: 8ucchiman
# CreatedDate:  2023-02-04 11:35:39 +0900
# LastModified: 2023-02-19 15:05:23 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
import argparse
from itertools import product
import numpy as np
from typing import Type
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn import ensemble
from sklearn import svm
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, log_loss
import xgboost as xgb
import lightgbm as lgb
import logging
import pandas as pd
# import utils


class Fitting(object):
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 target: str,
                 index: str,
                 logger: logging.RootLogger,
                 problem_type: str,
                 index_df: pd.DataFrame = None,
                 seed=43,
                 params=None):
        self.train_df = train_df
        self.test_df = test_df
        if index_df is None:
            self.test_index = self.test_df[index]
            self.train_df.drop([index], axis=1, inplace=True)
            self.test_df.drop([index], axis=1, inplace=True)
        else:
            self.test_index = index_df
        self.target = target
        # self.test_id = self.test_df["id"]
        self.X = self.train_df.drop([self.target], axis=1).values
        self.y = self.train_df[self.target].values
        self.X_test = self.test_df.values
        self.problem_type = problem_type
        self.seed = seed
        self.params = params
        if logger is None:
            self.logger = self.get_logger(".", "LOG.log")
        else:
            self.logger = logger

        self.ensemble_algorithm = ["RandomForestClassifier",
                                   "BaggingClassifier",
                                   "AdaBoostClassifier",
                                   "GradientBoostingClassifier",
                                   "ExtraTreesClassifier",
                                   "RandomForestRegressor",
                                   "GradientBoostingRegressor"]
        self.gaussian_process_algorithm = ["GaussianProcessClassifier"]
        self.linear_model_algorithm = ["LogisticRegressionCV",
                                       "PassiveAggressiveClassifier",
                                       "RidgeClassifierCV",
                                       "SGDClassifier",
                                       "Perceptron",
                                       "ElasticNet",
                                       "Lasso",
                                       "BayesianRidge",
                                       "LassoLarsIC"]
        self.kernel_ridge = ["KernelRidge"]
        self.naive_bayes = ["BernoulliNB", "GaussianNB"]
        self.neighbors = ["KNeighborsClassifier"]
        self.svm_algorithm = ["SVC", "NuSVC", "LinearSVC"]
        self.tree_algorithm = ["DecisionTreeClassifier", "ExtraTreeClassifier"]
        self.xgboost = ["XGBClassifier", "XGBRegressor"]
        self.lightgbm_algorithm = ["LGBMClassifier", "LGBMRegressor"]
        self.pred_test = None
        self.results_dict = {}

    def cross_validation(self, X_train, X_valid, y_train, y_valid):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def run(self):
        pass

    def run_lazypredict(self):
        from lazypredict.Supervised import LazyRegressor
        clf = LazyRegressor(verbose=0,
                            ignore_warnings=True,
                            custom_metric=None,
                            random_state=self.seed,)

        models, predictions = clf.fit(self.X_train, self.X_test, self.y_train, self.y_test)

    def kfold_cross_validation(self, clf):
        self.kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        pred_train = np.zeros((self.X.shape[0],))
        pred_test = np.zeros((self.X_test.shape[0],))
        pred_kfold = np.empty((self.kf.get_n_splits(), self.X_test.shape[0]))
        for i, (train_index, valid_index) in enumerate(self.kf.split(self.X)):
            x_train = self.X[train_index]
            y_train = self.y[train_index]
            x_valid = self.X[valid_index]

            clf.train(x_train, y_train)

            pred_train[valid_index] = clf.predict(x_valid)
            pred_kfold[i, :] = clf.predict(self.X_test)
        pred_test[:] = pred_kfold.mean(axis=0)
        return pred_train.reshape(-1, 1), pred_test.reshape(-1, 1)

    def base_models(self, method: str):
        if method in self.ensemble_algorithm:
            module = ensemble
        elif method in self.svm_algorithm:
            module = svm
        elif method in self.lightgbm_algorithm:
            module = lgb
        clf = SklearnHelper(clf=getattr(module, method), seed=self.seed, params=self.params[method])
        _, pred_test =  self.kfold_cross_validation(clf)
        self.submission(pred_test)

    def xgboost(self):
        self.model_xgb.fit(self.X, self.y)
        xgb_train_pred = self.model_xgb.predict(self.X)
        self.xgb_pred = np.expm1(self.model_xgb.predict(self.X_test))

    def lightgbm(self, X_train, y_train, X_valid, y_valid):
        if self.problem_type == "Regression":
            lgbm_params = {
                "objective": "regression",
                "learning_rate": 0.2,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "max_depth": 5,
                "n_estimators": 1000000,
                "colsample_bytree": 0.9,
            }
            self.model_lgb = lgb.LGBMRegressor(**lgbm_params)
        else:
            lgbm_params = {
                "max_depth": 5
            }
            self.model_lgb = lgb.LGBMClassifier(**lgbm_params)

        self.model_lgb.fit(self.X_train, self.y_train, eval_set=[(self.X_valid, self.y_valid)], early_stopping_rounds=20, eval_metric="rmse", verbose=200)
        # self.lgb_pred = self.model_lgb.predict(self.X_test)
        return self.model_lgb.predict(self.X_test)

    def ensembling(self):
        self.ensemble = self.stacked_pred*0.70 + self.xgb_pred*0.15 + self.lgb_pred*0.15

    def submission(self, pred_test):
        sub = pd.DataFrame()

        sub[self.target] = pred_test.reshape(-1) 
        sub = pd.concat([self.test_index, sub[self.target]], axis=1)
        sub.to_csv("submission.csv", index=False)

    def grid_search(self):
        pass

    def make_result_dataframe(self):
        pass

    def get_logger(self, log_dir, file_name):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, file_name)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Log file is %s." % log_path)
        return logger


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

class Metrics(object):
    def __init__(self):
        pass

    def rmse(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            RMSE(Root Mean Squared Error): (1/N\sum(y_true_i-y_pred_i)**2)**1/2
            RMSEを最小化した場合にも止まる解が、誤差が正規分布に従う前提のもとで求まる最尤解と同じ
            外れ値の影響を受けやすい-> 外れ値を除く必要がある

            Args
                y_true: 真の値
                y_pred: 予測値
            return
                rmse: 誤差
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

    def rmsle(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            RMSLE(Root Mean Squared Logarithmic Error): (1/N\sum(log(1+y_true_i)-log(1+y_pred_i))**2)**1/2
            y -> log(1+y)としてRMSEを求める
            目的変数が裾の重い分布をもち、変換しないままだと大きな値の影響が強い場合、
            真の値と予測値の比率に着目したい場合、RMSLEと用いる
            log(1+y_true_i)-log(1+y_pred_i) = log((1+y_true_i)/(1+y_pred_i)) -> 比率に着目
            log(1+y)としているのは真の値が0の時の発散を避けるため(log1p関数が使える)
            Args
                y_true: 真の値
                y_pred: 予測値
            return
                rmsle: 誤差
        """
        pass

    def mae(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            MAE(Mean Absolute Error): 1/N\sum|y_true_i-y_pred_i|
            RMSEと比較して外れ値の影響を低減できる
            y_pred_iによる微分についてy_pred_i-y_true_iで不連続だったり、二次微分が0になったり、扱いづらい。
        """
        pass

    def r2(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            R^2 (決定係数)
                R^2 = 1 - (\sum(y_true_i-y_pred_i)**2)/(\sum(y_true_i-y_mean)**2)
                y_mean = 1/N\sum(y_true_i)

            回帰分析の当てはまりの良さを表す
            分母: 予測値によらない(定数)
            分子: 分母 - 二乗誤差(RMSE)
            指標を最大化 -> 二乗誤差(RMSE)を最小化
            1に近づくほど精度が高い
        """
        pass

    def cm(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
            cm(混合行列)
                                       真の値
                                 Positive     Negative
                             +------------+------------+
                             |            |            |
                   Positive  |     TP     |     FP     |
                             |            |            |
                             |            |            |
            予測値           +------------+------------+
                             |            |            |
                   Negative  |     FN     |     TN     |
                             |            |            |
                             |            |            |
                             +------------+------------+
            Args
                y_true: 1(positive)/0(negative) e.g. [1, 0, 0, 1, 0, 1, 1, 1]
                y_pred: y_trueと同じ

        """
        # tp = np.sum(y_true == 1 & y_pred == 1)
        # tn = np.sum(y_true == 0 & y_pred == 0)
        # fp = np.sum(y_true == 0 & y_pred == 1)
        # fn = np.sum(y_true == 1 & y_pred == 0)

        # cm = np.array([[tp, fp],
        #                [fn, tn]])

        cm = confusion_matrix(y_true, y_pred)
        return cm

    def accuracy_errorrate(self, y_true: np.array, y_pred: np.array):
        """
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            error rate = 1 - accuracy

            不均衡データに対して性能は評価しづらい
        """
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy, 1-accuracy

    def precision_recall(self, y_true: np.array, y_pred: np.array):
        """
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
                                       真の値
                                 Positive     Negative
                             +------------+------------+
                             |  +-+----+------------+  |
                   Positive  |  | |TP  |  |     FP  |  |
                             |  +-|----|------------+  |
                             |    |    |  |  precision |
            予測値           +----|----|--+------------+
                             |    |    |  |            |
                   Negative  |    |FN  |  |     TN     |
                             |    +----+  |            |
                             |    recall  |            |
                             +------------+------------+
            ご検知を少なくを少なくしたい場合、過度にPositiveと予測しないようにprecision重視するべき
            Positiveの見逃しを避けたい場合、過度にNegativeと予測しないようにrecall重視すべき
        """
        pass

    def f1_fbeta(self, y_true: np.array, y_pred: np.array):
        """
            f1: precisionとrecallの調和平均
              f1 = 2 / (1/recall + 1/precision) = 2TP / (2TP + FP + FN)

            fbeta: f1からrecall, precisionのバランスを、recallをどれだけ重視するかを表す係数betaによって調整した指標
              fbeta = (1+beta**2) / (beta**2/recall + 1/precision)
        """
        pass

    def mcc(self, y_true: np.array, y_pred: np.array):
        """
            mcc(Matthews Correlation Coefficient)
                mcc = (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**1/2
            不均衡データに対してモデルの性能を適切に評価しやすい指標
            -1<=mcc<=1, +1: 完全な予測, 0: ランダム予測, -1: 完全に反対予測
            f1と違い、Positive, Negativeを対称に扱っている
        """
        pass

    def logloss(self, y_true: np.array, y_pred: np.array):
        """
            cross entropyと呼ばれることもある
            (2値)分類タスクでの代表的な指標
            Positiveである確率を予測値とする

            logloss = -1/N\sum(y_i\log(p_i) + (1-y_i)\log(1-p_i))
            y_i: ラベル(1: Positive, 0: Negative)
            真の値を予測している確率の対数をとり、符号反転させた値

            L_i = -(y_i\log(p_i)+(1-y_i)\log(1-p_i))
            მL_i/მp_i = (p_i-y_i)/p_i(1-p_i)
            p_i=y_iの時、L_iは最小となる
        """
        logloss = log_loss(y_true, y_pred)
        return logloss

    def auc(self, y_true: np.array, y_pred: np.array):
        """
            AUC(Area Under the ROC Curve)
            ROC Curve(Receiver Operating Characteristic Curve)が描く曲線を元に計算
        """
        pass

    def multiclass_logloss(self, y_true: np.array, y_pred: np.array):
        """
            マルチクラス分類に対するlogloss
            muticlass logloss = - 1/N \sum\sum y_{i,m} \log(p_{i, m})
        """
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default="../results", help="results dir specify")
    parser.add_argument('--data_dir', type=str, default="./datas", help="data directory specify")
    parser.add_argument('--train_csv', type=str, default="train.csv", help="train.csv specify")
    parser.add_argument('--test_csv', type=str, default="test.csv", help="test.csv specify")
    parser.add_argument('--target_col', type=str, required=True, help="target to predict")
    parser.add_argument('--index_col', type=str, required=True, help="sample id")
    parser.add_argument('--problem_type', type=str, required=True, choices=['Regression', 'Classification'], help="problem type[Regression, Classification]")
    # parser.add_argument('--method_name', type="str", default="make_date_log_directory", help="method name here in utils.py")

    args = parser.parse_args()
    return args


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    args = get_args()
    train_path = os.path.join(args.data_dir, args.train_csv)
    test_path = os.path.join(args.data_dir, args.test_csv)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    fitting = Fitting(train_df, test_df, args.target_col, args.index_col, logger=None, problem_type=args.problem_type)
    fitting.kfold_cross_validation()


if __name__ == "__main__":
    main()
