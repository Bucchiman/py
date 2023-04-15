#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	eda
# Author: 8ucchiman
# CreatedDate:  2023-02-02 22:18:03 +0900
# LastModified: 2023-02-19 21:47:26 +0900
# Reference: 8ucchiman.jp
#


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import argparse
# from logging import getLogger, config


class EDA(object):
    def __init__(self,
                 train_csv: str,
                 test_csv: str,
                 target: str,
                 index_col: str,
                 logger: logging.RootLogger,
                 imshow=False,
                 results_dir="results",):
        self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv)
        self.target = target
        self.index_col = index_col
        # self.train_df.drop(["PassengerId"], axis=1, inplace=True)
        self.features = [col for col in self.train_df.columns if col != self.target]
        self.results_dir = results_dir
        self.logger = logger
        self.imshow = imshow
        # self.total_df = pd.concat([self.train_df[self.features], self.test_df[self.features]], axis=0)
        self.basic_infomation()

    def distinguish_features(self, continuous_features: list[str], category_features: list[str], text_features: list[str]):
        '''
            連続値(実数等), カテゴリー, テキストとしてカラムを区別する
        '''
        self.continuous_features = continuous_features
        self.category_features = category_features
        self.text_features = text_features

    def basic_infomation(self):
        self.logger.info(self.features)
        self.logger.info("*"*20)
        self.logger.info("features info:\n{}".format(self.train_df.info()))
        self.logger.info("describe/numerical\n{}".format(self.train_df.describe()))
        # self.logger.info("describe/categorical\n{}".format(self.train_df.describe(include='O')))
        for col in self.features:
            self.logger.info("value_counts\n{}".format(self.train_df[[col]].value_counts(normalize=True)))
        self.logger.info("*"*20)
        self.logger.info("features info:\n{}".format(self.test_df.info()))
        self.logger.info("describe/numerical\n{}".format(self.test_df.describe()))
        # self.logger.info("describe/categorical\n{}".format(self.test_df.describe(include='O')))
        for col in self.features:
            self.logger.info("value_counts\n{}".format(self.test_df[[col]].value_counts(normalize=True)))
        self.logger.info("*"*20)


    def column_wise_missing(self):
        self.logger.info("-"*5+"train missing value"+"-"*5)
        self.logger.info("\n{}".format(self.train_df.isna().sum().sort_values(ascending=False)))
        self.logger.info("-"*5+"test missing value"+"-"*5)
        self.logger.info("\n{}".format(self.test_df.isna().sum().sort_values(ascending=False)))
        missing_train_column = pd.DataFrame(self.train_df.isna().sum()).sort_values(by=0, ascending=False)[:-1]
        missing_test_column = pd.DataFrame(self.test_df.isna().sum()).sort_values(by=0, ascending=False)

        fig = make_subplots(rows=1,
                            cols=2,
                            column_titles=["Train Data", "Test Data"],
                            x_title="Missing Values")
        fig.add_trace(go.Bar(x=missing_train_column[0],
                             y=missing_train_column.index,
                             orientation="h",
                             marker=dict(color=[n for n in range(12)],
                                         line_color='rgb(0,0,0)',
                                         line_width=2,
                                         coloraxis="coloraxis")),
                      1, 1)
        fig.add_trace(go.Bar(x=missing_test_column[0],
                             y=missing_test_column.index,
                             orientation="h",
                             marker=dict(color=[n for n in range(12)],
                                         line_color='rgb(0,0,0)',
                                         line_width=2,
                                         coloraxis="coloraxis")),
                      1, 2)
        fig.update_layout(showlegend=False, title_text="Column wise Null Value Distribution", title_x=0.5)
        if self.imshow:
            fig.show()
        fig.write_image(os.path.join(self.results_dir, "column_wise_distribution.png"))

    def row_wise_missing(self):
        missing_train_row = self.train_df.isna().sum(axis=1)
        missing_train_row = pd.DataFrame(missing_train_row.value_counts()/missing_train_row.shape[0]).reset_index()
        missing_test_row = self.test_df.isna().sum(axis=1)
        missing_test_row = pd.DataFrame(missing_test_row.value_counts()/missing_test_row.shape[0]).reset_index()
        missing_train_row.columns = ['no', 'count']
        missing_test_row.columns = ['no', 'count']
        missing_train_row["count"] = missing_train_row["count"]*100
        missing_test_row["count"] = missing_test_row["count"]*100
        fig = make_subplots(rows=1, cols=2, column_titles=["Train Data", "Test Data"], x_title="Missing Values",)
        fig.add_trace(go.Bar(x=missing_train_row["no"],
                             y=missing_train_row["count"],
                             marker=dict(color=[n for n in range(4)],
                             line_color='rgb(0,0,0)',
                             line_width=3,
                             coloraxis="coloraxis")),
                      1, 1)
        fig.add_trace(go.Bar(x=missing_test_row["no"],
                             y=missing_test_row["count"],
                             marker=dict(color=[n for n in range(4)],
                             line_color='rgb(0,0,0)',
                             line_width=3,
                             coloraxis="coloraxis")),
                      1, 2)
        fig.update_layout(showlegend=False, title_text="Row wise Null Value Distribution", title_x=0.5)
        if self.imshow:
            fig.show()
        fig.write_image(os.path.join(self.results_dir, "row_wise_distribution.png"))

    def single_histogram(self, feature: str, **kwargs):
        '''
            分布(正規分布, ポアソン分布)
            実数カラムの分布を見るのに役立つ
            カテゴリカラムではユニークなカテゴリーが多ければ使える
            kwargs
                hue: カテゴリー別で色分け
                kde: 分布曲線
            Reference: https://seaborn.pydata.org/generated/seaborn.displot.html
        '''
        fig = sns.displot(data=self.train_df, x=feature, **kwargs)
        if self.imshow:
            plt.show()
        fig.savefig(os.path.join(self.results_dir, "single_{}_histoplot.png".format(feature)))

    def multi_histograms(self, features: list[str], **kwargs):
        self.train_df[features].hist(bins=100)
        plt.savefig(os.path.join(self.results_dir, "multi_histoplot.png"))

    def single_scatterplot(self, feature_x: str, feature_y: str, **kwargs):
        '''
            2変数の分布を表示
            kwargs
                hue: カテゴリー別で色分け
            Reference: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        '''
        fig = sns.scatterplot(data=self.train_df, x=feature_x, y=feature_y)
        fig.savefig(os.path.join(self.results_dir, "scatterplot_{}_{}.png".format(feature_x, feature_y)))

    # def scatter_target_feature(self, feature: str):
    #     data = pd.concat([self.train_df[self.target], self.train_df[feature]], axis=1)
    #     fig = data.plot.scatter(x=feature, y=self.target)
    #     fig.figure.savefig("scatter_{}_{}.png".format(feature, self.target))

    def boxplot_target_category_feature(self, category_feature):
        data = pd.concat([self.train_df[self.target], self.train_df[category_feature]], axis=1)
        f, ax = plt.subplots(figsize=[8, 6])
        fig = sns.boxplot(x=category_feature, y=self.target, data=data)
        fig.savefig("boxplot_{}_{}.png".format(self.target, category_feature))

    def distribution_of_continuous(self):
        labels = ['Categorical', 'Continuous', 'Text']
        values = [len(self.cat_features), len(self.cont_features), len(self.text_features)]
        colors = ['#DE3163', '#58D68D']

        fig = go.Figure(data=[go.Pie(labels=labels,
                                     values=values, pull=[0.1, 0, 0],
                                     marker=dict(colors=colors,
                                                 line=dict(color='#000000', width=2)))])
        fig.write_image(os.path.join(self.results_dir, "features_cat_cont_text_Pie.png"))
        train_age = self.train_df.copy()
        test_age = self.test_df.copy()
        train_age["type"] = "Train"
        test_age["type"] = "Test"
        ageDf = pd.concat([train_age, test_age])
        fig = px.histogram(data_frame=ageDf,
                           x="Age",
                           color="type",
                           color_discrete_sequence=['#58D68D', '#DE3163'],
                           marginal="box",
                           nbins=100,
                           template="plotly_white")
        fig.update_layout(title="Distribution of Age", title_x=0.5)
        fig.write_image(os.path.join(self.results_dir, "age_histogram.png"))

    def distribution_of_category(self):
        if len(self.cat_features) == 0:
            self.logger.info("No categories")
        else:
            ncols = 2
            nrows = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10))
            for r in range(nrows):
                for c in range(ncols):
                    col = self.cat_features[r*ncols+c]
                    sns.countplot(x=col,
                                  data=self.train_df,
                                  ax=axes[r, c],
                                  palette="viridis",
                                  label='Train data')
                    sns.countplot(x=col,
                                  data=self.test_df,
                                  ax=axes[r, c],
                                  palette="magma",
                                  label='Test data')
                    axes[r, c].legend()
                    axes[r, c].set_ylabel('')
                    axes[r, c].set_xlabel(col, fontsize=20)
                    axes[r, c].tick_params(labelsize=10, width=0.5)
                    axes[r, c].xaxis.offsetText.set_fontsize(4)
                    axes[r, c].yaxis.offsetText.set_fontsize(4)
            if self.imshow:
                fig.show()
            fig.savefig(os.path.join(self.results_dir, "category_distribution.png"))

    def correlation_matrix(self):
        fig = px.imshow(self.train_df.corr(),
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="viridis")
        if self.imshow:
            fig.show()

        fig = sns.heatmap(self.train_df.corr())
        fig.figure.savefig(os.path.join(self.results_dir, "correlation_matrix.png"))

    def multi_scatterplot(self, continuous_features: list[str], **kwargs):
        '''
            実数値のみの特徴量を全て出力
            continuous_features: カラムのリスト
            kwargs
                hue: カテゴリごとに色分け
        '''
        sns.set()
        fig = sns.pairplot(self.train_df[continuous_features], size=2.5, **kwargs)
        fig.figure.savefig(os.path.join(self.results_dir, "multi_scatterplot.png"))

    def groupby_pivoting(self, features: list[str], target: str = None):
        if not target:
            target = self.target
        for feature in features:
            self.logger.info("{}/{} groupby\n{}".format(feature, target, self.train_df[[feature, target]].groupby([feature], as_index=False).mean().sort_values(target)))

    def facetgrid(self, method: str, feature: str, target: str = None, **kwargs):
        '''
            条件を絞ってその条件のグラフを表示することができる
            row, col: カテゴリーカラムにすればカテゴリー分のgridが用意される
            e.g.
                grid.map(getattr(sns, scatterplot), x, y)    分布
                grid.map(getattr(sns, histplot), x(カテゴリor実数))    分布
                grid.map(getattr(sns, barplot), x(カテゴリ), y(実数))

            Reference: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
        '''
        if not target:
            target = self.target
        grid = sns.FacetGrid(self.train_df, row=feature, col=target)
        grid.map_dataframe(getattr(sns, method), **kwargs)
        if self.imshow:
            plt.show()
        grid.savefig(os.path.join(self.results_dir, "facet_key_{}_{}_map_{}.png".format(feature, target, method)))

    @classmethod
    def get_logger(log_dir, file_name):
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='.')
    parser.add_argument('--log_file', type=str, default='LOG.log')
    parser.add_argument('--results_dir', type=str, default=".", help="results dir specify")
    parser.add_argument('--data_dir', type=str, default="./datas", help="data directory specify")
    parser.add_argument('--train_csv', type=str, default="train.csv", help="train.csv specify")
    parser.add_argument('--test_csv', type=str, default="test.csv", help="test.csv specify")
    parser.add_argument('--target_col', type=str, required=True, help="target to predict")
    parser.add_argument('--index_col', type=str, required=True, help="sample id")
    parser.add_argument('--problem_type', type=str, required=True, choices=['Regression', 'Classification'], help="problem type[Regression, Classification]")
    parser.add_argument('--imshow', action='store_true')
    # parser.add_argument('--method_name', type="str", default="make_date_log_directory", help="method name here in utils.py")

    # parser.add_argument('arg1')     # 必須の引数
    # parser.add_argument('-a', 'arg')    # 省略形
    # parser.add_argument('--flag', action='store_true')  # flag
    # parser.add_argument('--strlist', required=True, nargs="*", type=str, help='a list of strings') # --strlist hoge fuga geho
    # parser.add_argument('--method', type=str)
    # parser.add_argument('--fruit', type=str, default='apple', choices=['apple', 'banana'], required=True)
    # parser.add_argument('--address', type=lambda x: list(map(int, x.split('.'))), help="IP address") # --address 192.168.31.150 --> [192, 168, 31, 150]
    # parser.add_argument('--colors', nargs='*', required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_path = os.path.join(args.data_dir, args.train_csv)
    test_path = os.path.join(args.data_dir, args.test_csv)
    logger = EDA.get_logger(log_dir=args.log_dir, file_name=args.log_file)
    eda = EDA(train_path, test_path, args.target_col, args.index_col, logger, imshow=args.imshow, results_dir=args.results_dir)
    # eda.get_logger(".", "sample.log")
    eda.single_histogram("Age", kde=True, hue=args.target_col)
    # eda.scatter_target_feature('GrLivArea')
    # eda.scatterplot(['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'])
    # eda.column_wise_missing()
    # eda.row_wise_missing()
    # eda.distribution_of_continuous()
    # eda.distribution_of_category()
    # eda.correlation_matrix()


if __name__ == "__main__":
    main()
