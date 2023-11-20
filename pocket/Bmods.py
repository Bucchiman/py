#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     Bmods
# Author:       8ucchiman
# CreatedDate:  2023-07-27 13:18:37
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#

############################################################################################

import os
import cv2 as cv
import numpy as np
import scipy as cp
import pandas as pd
import inspect
from time import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


# -------------------------------------------
# basic
import logging
from logging import getLogger, config

def get_logger(file_name: str, log_dir="../logs"):
    """TL;DR
    Description
    ----------

    ----------
    Parameters
    ----------
    gamma : float, default: 1
        Desc
    s : float, default: 0.5 (purple)
        Desc
    ----------
    Return
    ----------

    ----------
    Example
    ----------
    get_logger()
    ----------
    Reference
    ----------
    """
    os.makedirs(os.path.join(log_dir, make_date_log_directory()), exist_ok=True)
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

def make_date_log_directory():
    from datetime import datetime
    return datetime.now().strftime(r"%Y_%m_%d_%H_%M")

import argparse
def get_base_parser():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--log_dir', type=str, default="../logs", help="log directory specify")
    base_parser.add_argument('--log_file', type=str, default=make_date_log_directory(), help="log file specify")
    base_parser.add_argument('--config_dir', type=str, default="../params")
    base_parser.add_argument('--config_file', type=str, default="config.yaml")
    base_parser.add_argument('--results_dir', type=str, default="../results", help="results dir specify")
    base_parser.add_argument('--data_dir', type=str, default="../data", help="data directory specify")
    # parser.add_argument('--method_name', type="str", default="make_date_log_directory", help="method name here in utils.py")

    # parser.add_argument('arg1')     # 必須の引数
    # parser.add_argument('-a', 'arg')    # 省略形
    # parser.add_argument('--flag', action='store_true')  # flag
    # parser.add_argument('--strlist', required=True, nargs="*", type=str, help='a list of strings') # --strlist hoge fuga geho
    # parser.add_argument('--method', type=str)
    # parser.add_argument('--fruit', type=str, default='apple', choices=['apple', 'banana'], required=True)
    # parser.add_argument('--address', type=lambda x: list(map(int, x.split('.'))), help="IP address") # --address 192.168.31.150 --> [192, 168, 31, 150]
    # parser.add_argument('--colors', nargs='*', required=True)

    return base_parser


def get_ml_args():
    ml_parser = argparse.ArgumentParser(parents=[get_base_parser()])
    ml_parser.add_argument('--train_csv', type=str, default="train.csv", help="train.csv specify")
    ml_parser.add_argument('--test_csv', type=str, default="test.csv", help="test.csv specify")
    ml_parser.add_argument('--target_col', type=str, required=True, help="target to predict")
    ml_parser.add_argument('--index_col', type=str, required=True, help="sample id")
    ml_parser.add_argument('-e', '--eda', action='store_true', help="eda flag")
    ml_parser.add_argument('-p', '--preprocessing', action='store_true', help="preprocessing flag")
    ml_parser.add_argument('-f', '--fitting', action='store_true', help="fitting flag")
    ml_parser.add_argument('--problem_type', type=str, required=True, choices=['Regression', 'Classification'], help="problem type[Regression, Classification]")
    ml_parser.add_argument('--save_csv_dir', type=str, default="../preprocessing_dir", help="save dir specify")
    ml_parser.add_argument('--imshow', action='store_true')
    args = ml_parser.parse_args()
    return args


def get_dl_args():
    dl_parser = argparse.ArgumentParser(parents=[get_base_parser()])
    dl_parser.add_argument('--train_img_dir', type=str, required=False)
    dl_parser.add_argument('--test_img_dir', type=str, required=False)
    dl_parser.add_argument('--train_label_file', type=str, required=False)
    dl_parser.add_argument('--batch_size', type=int, default=5)
    dl_parser.add_argument('--model_name', type=str, default='resnet18')
    dl_parser.add_argument('--gpus', type=str, default="all", choices=['all', 'cuda:0'])
    args = dl_parser.parse_args()
    return args


# -------------------------------------------


def reflection_methods (target_obj):
    """
    Description

    Parameters

    Return

    Reference
        https://blog.goo.ne.jp/dak-ikd/e/fa1766c5ded886ac513ef3e761551560
    """
    import inspect
    function_lst = list(map(lambda x: x[0], inspect.getmembers(target_obj)))

    return function_lst


def Bucchiman_was_here ():
    """
        Description: for debug to check pythonpath
    """
    print("8ucchiman was here!!!!!")


# Do you like gulliver codes??????????????????
if 0:
    from gutils_misc import OPEN, RUN, CPR, LNS, MKDIR, RMRF, RMRF_FULL
    from gutils_misc import CAT, HEAD, TAIL, HEAD_TAIL, TOUCH, LINE
    from gutils_misc import WHOAMI, MYTMP
    from gutils_misc import HOME, lena, gm_im_dir, gm_im_lst, gm_im_calib_dir
    from gutils_misc import get_latest_file
    
    from gutils_misc import nx_info, nx_shape, nx_npz, nx_npz_load, nx_dir, nx_dict, nx_cfg, notqdm

    from gutils_time import gTicToc, time_func, FPS, gFPS_fixed

    from gutils_im import im_grid, asIm, asArray
    from gutils_tands import im_movie, cv2_noise

    from gutils_model_params_info_sparsity import plt_decorate, from_numpy, to_numpy



if 0:
    from gutils_misc import nx_npz, nx_npz_load
    from gutils_numpy import np_savetxt, np_loadtxt
    from gutils_grid import loadnpz, savenpz, asArray

    from gutils_im import asIm
    from gutils_ocv import calibrate_camera

    from gutils_model_params_info_sparsity import plt_decorate, axtext, axvline, axhline, axhline_mean_std
    from gutils_model_params_info_sparsity import from_numpy, to_numpy



def _simple_httpserver ():
    """
    Description sample code for multithread
                

    Parameters

    Return

    Reference   質実剛健 Rust Interface
    """

    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    hostName = "127.0.0.1"      # サーバのlistenアドレス
    serverPort = 8080           # サーバのlistenポート

    class Handler (BaseHTTPRequestHandler):
        def do_GET (self):
            if self.path == "/":
                self.send_response(200)
                self.send_header ("Content-type", "text/plain")

                self.end_headers()
                self.wfile.write(bytes("Hello, world.", "utf-8"))

            else:
                self.send_response(404)
                self.send_header("Content-type", "text/plain")

                self.end_headers()
                self.wfile.write(bytes("Not Found", "utf-8"))

        def log_message (self, format, *args):
            return

    class ThreadedHTTPServer  (ThreadingMixIn, HTTPServer):
        """ Handle requests in a separate thread. """


    # 標準ライブラリ上で定義したハンドラの機能を持つマルチスレッドサーバを初期化
    server = ThreadedHTTPServer((hostName, serverPort), Handler)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
    print("Server stopped.")


# def



def Bshow_image (path: str):
    """
        Arguments:
            - path: image path
        Reference: https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
    """
    image = cv.imread(path)
    cv.imshow("image", image)
    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing) 
    cv.waitKey(0)

    # closing all open windows
    cv.destroyAllWindows()

def Bshow_triplet (a, b, c):
    concat_images = cv.hconcat([a, b, c])
    return concat_images

def Bshow_pairs (a, b):
    """
        show pair images
        Arguments:
            - a: image A
            - b: image B
    """
    concat_images = cv.hconcat([a, b])
    cv.imshow("image", concat_images)


def Bfzfprompt(func):
    """fzf prompt
    Description
    ----------
    
    ----------
    Parameters
    ----------
    gamma : float, default: 1
        Desc
    s : float, default: 0.5 (purple)
        Desc
    ----------
    Return
    ----------

    ----------
    Example
    ----------
    @Bfzfprompt
    def _run_bfzfprompt(lst):
        print(lst)


    _run_bfzfprompt(['hoge', 'kie', 'becori'])
    ----------
    Reference
    ----------
    """
    def wrapper(*args, **kwargs):
        from pyfzf.pyfzf import FzfPrompt
        fzf = FzfPrompt()
        results = fzf.prompt(args[0])        # ['hoge', 'kie', 'becori']
        func(results, **kwargs)
    return wrapper

@Bfzfprompt
def _run_bfzfprompt(lst):
    print(lst)


# _run_bfzfprompt(['hoge', 'kie', 'becori'])

def Bfzfprompt4path (path: str):
    from pyfzf.pyfzf import FzfPrompt
    #fzf

def Bshow_video(path: str):
    """
        Arguments:
            - path: video path
        Reference: https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        Description:
            This is VideoShow
    """
    cap = cv.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
 
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
 
            # Press Q on keyboard to  exit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break
 
    # When everything done, release the video capture object
    cap.release()
 
    # Closes all the frames
    cv.destroyAllWindows()

#
# Reference: gulliver, gutils.zip, gutils_time.py
# decorator
def Gtimer(func):
    def wrapper(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print(f"[{func.__name__}] elapsed {after-before:.6f}")
        return rv
    return wrapper


@Gtimer
def add(x, y):
    return x + y


def Bimage_filter (func):
    def wrapper(*args, **kwargs):
        processing_image = func(*args, **kwargs)
        concat_images = cv.hconcat([args[0], processing_image])
        cv.imshow("image", concat_images)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing) 
        cv.waitKey(0)
        # closing all open windows
        cv.destroyAllWindows()
    return wrapper

@Gtimer
@Bimage_filter
def gaussianfilter (image):
    return cv.GaussianBlur(image, (5, 5), 0)

@Gtimer
@Bimage_filter
def medianfilter (image):
    return cv.medianBlur(image, 7)

@Gtimer
@Bimage_filter
def bilateralfilter (image):
    return cv.bilateralFilter(image, 9, 75, 75)



################################################################
# gpio

class Sample4GPIO (object):
    def _gpio_mcp3008_3002 (self):
        """
        # Reference: https://www.denshi.club/pc/raspi/5raspberry-pi-zeroiot9a-d6mcp3002.html
                     https://qiita.com/M_Study/items/fc0df5069c76418ef7e2
                     https://101010.fun/raspberry-pi-adc-mcp3008.html
        CH0	1   16  Vdd     3.3V電源
        CH1	2   15  Vref    基準電圧入力
        CH2	3   14  AGND    アナログ・グラウンド
        CH3	4   13  CLK
        CH4	5   12  Dout
        CH5	6   11  Din
        CH6	7   10  /CS・SHDN
        CH7	8    9  DGND    ディジタル・グラウン
        """


################################################################
# fourier




################################################################

################################################################


def Bimagegrid (func):
    """
        image grid sample
        https://qiita.com/mtb_beta/items/d257519b018b8cd0cc2e
    """
    def wrapper(*args, **kwargs):
        # toDo
        # print("wrapper area")
        results = func(*args, **kwargs)
        return results
    return wrapper

@Bimagegrid
def _run ():
    # toDo
    pass


def scraping ():
    from urllib.request import urlopen
    url = "https://matplotlib.org/stable/gallery/index"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    title_index = html.find("<title>")


############################################################################################
# Ray tune

from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler



############################################################################################
# toDo pytorch


from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image


class BImageTransform (object):
    """TL;DR
    Description
    ----------

    ----------
    Parameters
    ----------
    gamma : float, default: 1
        Desc
    s : float, default: 0.5 (purple)
        Desc
    ----------
    Return
    ----------

    ----------
    Example
    ----------

    ----------
    Reference
    ----------
    Date
    2023-11-19 16時44分35秒
    ----------
    """
    def __init__(self):
        from torchvision.transforms import v2
        from pyfzf.pyfzf import FzfPrompt
        # print(v2.__dict__)
        fzf = FzfPrompt()
        results = fzf.prompt(v2.__dict__.keys())

        pass

    #def 


class BImageModel():
    pass


class BImageCustomDataset (Dataset):
    """Custom Dataset
    Description
    ----------

    ----------
    Parameters
    ----------
    labels:             dict(index: ["image_name" as str, classname as int])
                        {
                            0: ["train_001.png", 4],
                        }
    img_dir:            string of path
                        img_dir = "../data/train"
    transform           transforms = torchvision.transforms.v2.Compose([
                            v2.RandomResizedCrop(size=(224, 224), antialias=True),
                            v2.RandomHorizontalFlip(p=0.5),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    target_transform    hogehoge
    ----------
    Return
    ----------

    ----------
    Example
    ----------
        image_dir = "../data/train"

    ----------
    Reference
    transforms          https://pytorch.org/vision/0.15/transforms.html
    cross validation    https://qiita.com/ground0state/items/ad879a84bf946ef94da8
    ----------
    Date
    2023-11-19 16時44分56秒
    ----------
    """
    def __init__(self, labels, img_dir, transform=None):
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def cross_validation(labels, image_dir, transform=None):
        """TL;DR
        Description
        ----------

        ----------
        Parameters
        ----------
        gamma : float, default: 1
            Desc
        s : float, default: 0.5 (purple)
            Desc
        ----------
        Return
            train_dataset
            valid_dataset
        ----------

        ----------
        Example
        labels = df[["image_name", "class"]]
        split_lst = BImageCustomDataset.cross_validation(labels)

        split_n = 0
        train_idx, valid_idx = split_lst[split_n]
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        train_dataset = BImageCustomDataset(train_df, image_dir)
        valid_dataset = BImageCustomDataset(valid_df, image_dir)

        train_batch_size = 4
        valid_batch_size = 4
        train_dataloader = DataLoader(train_dataset, train_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, valid_batch_size, shuffle=True)

        ----------

        ----------
        Reference
        cross validation        https://qiita.com/ground0state/items/ad879a84bf946ef94da8
        ----------
        """
        n_split = 3
        kf = KFold(n_split)
        split_lst = list(kf.split(range(len(labels))))
        packet_dataset = []
        for fold, (train_idx, valid_idx) in enumerate(split_lst):
            train_df = labels.iloc[train_idx]
            valid_df = labels.iloc[valid_idx]
            train_dataset = BImageCustomDataset(train_df, image_dir, transform)
            valid_dataset = BImageCustomDataset(valid_df, image_dir, transform)
            packet_dataset.append([train_dataset, valid_dataset])
        return packet_dataset


import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np


class BImageTrain(object):
    """TL;DR
    Description
    ----------

    ----------
    Parameters
    ----------
    gamma : float, default: 1
        Desc
    s : float, default: 0.5 (purple)
        Desc
    ----------
    Return
    ----------

    ----------
    Example
    ----------

    ----------
    Reference
    ----------
    """
    def __init__(self, packet_dataset, split_n_lst, model, epochs=10, device="cpu", loss_fn=nn.CrossEntropyLoss):
        self.packet_dataset = packet_dataset
        self.split_n_lst = split_n_lst
        self.model = model
        self.device = device
        self.loss_fn = loss_fn()
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)

    def loop_cross_validation(self):
        for fold_n in self.split_n_lst:
            train_dataset, valid_dataset = self.packet_dataset[fold_n]
            train_dataloader = DataLoader(train_dataset, 5, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, 5, shuffle=True)
            total_train_loss = []
            total_valid_loss = []
            valid_interval = 5

            for epoch in range(1, self.epochs+1):
                train_loss, train_correct = self.train_one_epoch(train_dataloader)
                total_train_loss.append(train_loss)
                if epoch % valid_interval == 0:
                    valid_loss, valid_correct = self.valid_one_epoch(valid_dataloader)
                    total_valid_loss.append(valid_loss)
            BMatplotlib.loss_curve(total_train_loss, total_valid_loss, valid_interval, f"{str(fold_n).zfill(2)}.png")


    # @classmethod
    def train_one_epoch(self, dataloader):
        """TL;DR
        Description
        ----------

        ----------
        Parameters
        ----------
        gamma : float, default: 1
            Desc
        s : float, default: 0.5 (purple)
            Desc
        ----------
        Return
        ----------

        ----------
        Example
        ----------

        ----------
        Reference
        Default_base            https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        Cross validation        https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
        ----------
        """
        train_loss = 0.
        train_correct = 0
        self.model.train()

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for images, labels in dataloader:
            # print(images.size())
            labels = torch.tensor(labels, dtype=torch.long)
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()           # Zero your gradients for every batch!
            outputs = self.model(images)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            train_loss += loss.item() * images.size(0)      # total of size of minibatch

            scores, predictions = torch.max(outputs.data, 1)
            train_correct += (predictions == labels).sum().item()
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(train_dataloader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.

        return train_loss, train_correct

    def valid_one_epoch(self, dataloader):
        valid_loss, val_correct = 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long)
                output = self.model(images)
                loss = self.loss_fn(output, labels)
                valid_loss += loss.item()*images.size(0)
                scores, predictions = torch.max(output.data, 1)
                val_correct += (predictions == labels).sum().item()
        return valid_loss, val_correct


############################################################################################
# Scraping web


############################################################################################
# toDo matplotlib
def Bcalc_histgram():
    pass


class BMatplotlib(object):
    """TL;DR
    Description
    ----------

    ----------
    Parameters
    ----------
    gamma : float, default: 1
        Desc
    s : float, default: 0.5 (purple)
        Desc
    ----------
    Return
    ----------

    ----------
    Example
    ----------

    ----------
    Reference
    https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9
    https://towardsdatascience.com/clearing-the-confusion-once-and-for-all-fig-ax-plt-subplots-b122bb7783ca
    ----------
    """

    def __init__(self):
        pass

    @staticmethod
    def loss_curve(train_loss, valid_loss, valid_interval, fig_name):
        train_x = list(range(1, len(train_loss)+1))
        valid_x = list(range(1, len(train_loss)+1, valid_interval))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(train_x, train_loss)
        ax.plot(valid_x, valid_loss)
        # plt.show()
        fig.savefig(fig_name)


############################################################################################
# toDo pandas

############################################################################################
# toDo hagging face

############################################################################################
# toDo mlflow

############################################################################################
# toDo scikit-learn

class MyScikitLearn (object):
    def get_dataset(self):
        """TL;DR
        Description
        ----------

        ----------
        Parameters
        ----------
        gamma : float, default: 1
            Desc
        s : float, default: 0.5 (purple)
            Desc
        ----------
        Return
        ----------
        return a tuple (X, y) consisting of a n_samples * n_features numpy array X and an array of length n_samples containing the targets y.
        ----------
        Example
        ----------

        ----------
        Reference
        ----------
        """
        from sklearn import datasets
        from pyfzf.pyfzf import FzfPrompt
        dataset_lst = {
                "iris":         datasets.load_iris,
                "diabetes":     datasets.load_diabetes,
                "digits":       datasets.load_digits,
                "linnerud":     datasets.load_linnerud,
                "wine":         datasets.load_wine,
                "breast_cancer":datasets.load_breast_cancer
                }
        fzf = FzfPrompt()
        result = dataset_lst[fzf.prompt(dataset_lst.keys())[0]]
        return result(return_X_y=True)

    def _sample_linear_regression (self):
        """Linear Regression sample code
        Description
        ----------

        ----------
        Parameters
        ----------
        gamma : float, default: 1
            Desc
        s : float, default: 0.5 (purple)
            Desc
        ----------
        Return
        ----------
        Show figure
        ----------
        Example
        ----------

        ----------
        Reference
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        ----------
        """
        import matplotlib.pyplot as plt
        import numpy as np

        from sklearn import datasets, linear_model
        from sklearn.metrics import mean_squared_error, r2_score

        X, y = self.get_dataset()

        # Use only one feature
        X = X[:, np.newaxis, 2]

        X_train = X[:-20]
        X_test = X[-20:]

        y_train = y[:-20]
        y_test = y[-20:]

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)

        y_pred = regr.predict(X_test)

        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

        plt.scatter(X_test, y_test, c="black")
        plt.plot(X_test, y_pred, c="blue", linewidth=3)
        plt.xticks(())
        plt.yticks(())

        plt.show()




############################################################################################
# toDo onnx

############################################################################################
# toDo others

############################################################################################
# Fractal
def Bcreate_fractal_image ():
    """
        Model:      IFS
        Keywords:   FractalDB-1k
        Reference:  https://link.springer.com/content/pdf/10.1007/s11263-021-01555-8.pdf
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from gutils_model_params_info_sparsity import plt_decorate, axtext, axvline, axhline, axhline_mean_std
    def _image_show(Xt, Xt1):
        plt.plot(Xt, c="r", label="Xt")
        plt.plot(Xt1, c="r", label="Xt")
        figsize = (16, 9)
        title = "IFS h"
        xlim, ylim = None, None
        xlabel, ylabel = "x", "y"
        show = False
        save_fn = "/tmp/a.png"
        legend = ("upper right", "func type")
        opts = dict(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                    show=show, save_fn=save_fn, legend=legend, figsize=figsize)
        plt_decorate(**opts)
        OPEN(save_fn)
    x = np.random.random([2, 1])
    for i in range(10):
        A = np.random.random([2, 2])
        B = np.random.random([2, 1])
        W = A@x + B
        print(W)
        _image_show(X, W)
        X = W

    

#Bcreate_fractal_image()
############################################################################################

# epipolar

def create_epipolar ():
    """
        Reference:  https://docs.opencv.org/4.8.0/da/de9/tutorial_py_epipolar_geometry.html
                    https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
        Code:       https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/epipolar.py
        Image:      https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/left.jpg
                    https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/right.jpg
    """
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1]])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    ##########

    # Instead SIFT+FLANN, we use SIFT+BFMatch

    img1 = cv2.imread('/home/yk.iwabuchi/.config/sample/pics/epipolar_sample_left.jpg',0)  # left
    img2 = cv2.imread('/home/yk.iwabuchi/.config/sample/pics/epipolar_sample_right.jpg',0) # right

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # Brute-Force Matcher
    bf = cv2.BFMatcher()

    # 
    matches = bf.knnMatch(des1, des2, k=2)

    # 
    ratio = 0.65
    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    #------------------------------

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


############################################################################################

# gutils
def _sample_main_mpl_plot ():
    from gutils_howto import main_mpl_plot
    main_mpl_plot()


def _sample_main_fps ():
    from gutils_howto import main_fps
    main_fps()


# misc utils, cp, lns, rm, head, tail, ...
def _sample_main_misc ():
    '''
    misc utils, cp, lns, rm, head, tail, ...

    touch src; cp src dst; rm src; ln -s src dst; 
    '''
    from gutils_misc import OPEN, RUN, CPR, LNS, MKDIR, RMRF, RMRF_FULL
    from gutils_misc import CAT, HEAD, TAIL, HEAD_TAIL, TOUCH, LINE
    from gutils_misc import WHOAMI, MYTMP

    tmp = MYTMP
    print(tmp)
    src = f"{tmp}/xxx.txt"
    dst = f"{tmp}/yyy.txt"
    lns = f"{tmp}/zzz.txt"
    TOUCH(src)
    CPR(src, dst)
    RMRF(src)
    LNS(dst, lns)



def main_mpl_plot ():

    pass


def main_fps_fixed ():
    pass


def main_fps ():
    from gutils_time import FPS # XXX: 
    from time import sleep
    ''' 
    report FPS for a process

    class: FPS
        Arguments:
            fps: 5
            msg: "fps label"
        FPS.tick()
            

    output:
        0 1693398288.7831407
        1 1693398289.2840245
        2 1693398289.7843204
        3 1693398290.2849793
        [   2 fps] fps label4 1693398290.7868018
        5 1693398291.2874384
        6 1693398291.788087
        7 1693398292.2887347
        8 1693398292.7893777
        [   2 fps] fps label9 1693398293.2908878
    '''
    fps = FPS(5, msg="fps label")
    for i in range(10):
        fps.tick()
        print(i, time())
        sleep(0.5)
    pass

############################################################################################


if __name__ == "__main__":
    _HOME = os.environ["HOME"]

    image_lst = os.listdir(os.path.join(_HOME, ".config/sample/pics"))
    # image_lst = ["baboon.jpg", "board.jpg", "building.jpg", "fruits.jpg",
    #              "lena.jpg", "nujabes.jpg", "Nujabes.webp", "nujabes_illust.jpeg", "stuff.jpg", "rain.jpg"]
    # video_lst = ["Mountain.mp4", "slow_traffic_small.mp4", "vtest.avi"]
    video_lst = os.listdir(os.path.join(_HOME, ".config/sample/videos"))
    # image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[4])
    image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[9])
    # print(IMAGE_NAME)
    # show_image(IMAGE_NAME)
    # show_video(os.path.join(_HOME, ".config/sample/videos", video_lst[0]))
    # add(2, 4)
    # image = cv.imread(image_path)
    # gaussianfilter(image)

    # Bgabor_filter(image)
    #_sample_main_mpl_plot()

    # mymatplotlib
    # mymatplotlib = MyMatplotlib()
    # methods = reflection_methods(mymatplotlib)
    # _run_bfzfprompt(methods, mymatplotlib)

    _simple_httpserver()
