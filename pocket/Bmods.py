#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     Bmods
# Author:       8ucchiman
# CreatedDate:  2023-07-27 13:18:37
# LastModified: 2023-12-16 20:38:32
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




# -------------------------------------------
# basic
import logging
from logging import getLogger, config


class BDeploy(object):
    def __init__(self):
        pass

    def torch2onnx(self):
        pass

    def torch2trt(self):
        pass

    def torch2tensor(self):
        pass


class BLogger(object):
    """logger
    logger for machine learning, image processing
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
    log_dir = os.path.join("../logs")
    Blogger.get_logger("hoge.log", "../logs")
    ----------
    Reference
    ----------
    """
    @staticmethod
    def get_logger(file_name: str, log_dir: str):
        """TL;DR
        logger
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

    @staticmethod
    def make_date_log_directory():
        from datetime import datetime
        return datetime.now().strftime(r"%Y_%m_%d_%H_%M")

# -------------------------------------------
import hydra
from omegaconf import DictConfig, OmegaConf


class BConfig(object):
    """TL;DR
    Config yaml
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
    @staticmethod
    def get_cnf(params_dir: str, config_file: str):
        '''
        @return
            cnf: OmegaDict
        '''
        with initialize_config_dir(version_base=None, config_dir=Path(params_dir).resolve()._str):
            cnf = compose(config_name=config_file)
            return cnf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()

# -------------------------------------------
# Tune
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler


class BTune(object):
    def __init__(self):
        pass

    @staticmethod
    def _sample():
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
        https://docs.ray.io/en/latest/tune/getting-started.html
        ----------
        """
        from ray import train, tune
        from ray.tune.schedulers import ASHAScheduler
        import torch.nn.functional as F
        import numpy as np
        import torch
        import torch.optim as optim
        import torch.nn as nn
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        import matplotlib.pyplot as plt

        from ray.train import RunConfig

        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                # In this example, we don't change the model architecture
                # due to simplicity.
                self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
                self.fc = nn.Linear(192, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 3))
                x = x.view(-1, 192)
                x = self.fc(x)
                return F.log_softmax(x, dim=1)

            # Change these values if you want the training to run quicker or slower.
        EPOCH_SIZE = 512
        TEST_SIZE = 256

        def train_func(model, optimizer, train_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # We set this just for the example to run quickly.
                if batch_idx * len(data) > EPOCH_SIZE:
                    return
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()


        def test_func(model, data_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    # We set this just for the example to run quickly.
                    if batch_idx * len(data) > TEST_SIZE:
                        break
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            return correct / total

        def train_mnist(config, data=None):
            print(data)
            # Data Setup
            mnist_transforms = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))])

            train_loader = DataLoader(
                datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
                batch_size=64,
                shuffle=True)
            test_loader = DataLoader(
                datasets.MNIST("~/data", train=False, transform=mnist_transforms),
                batch_size=64,
                shuffle=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ConvNet()
            model.to(device)

            optimizer = optim.SGD(
                model.parameters(), lr=config["lr"], momentum=config["momentum"])
            for i in range(10):
                train_func(model, optimizer, train_loader)
                acc = test_func(model, test_loader)

                # Send the current training result back to Tune
                train.report({"mean_accuracy": acc})

                if i % 5 == 0:
                    # This saves the model to the trial directory
                    torch.save(model.state_dict(), "./model.pth")

        search_space = {
                "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
                "momentum": tune.uniform(0.1, 0.9),
        }

        # Uncomment this to enable distributed execution
        # `ray.init(address="auto")`

        # Download the dataset first
        datasets.MNIST("~/data", train=True, download=True)

        tuner = tune.Tuner(
            tune.with_parameters(train_mnist, data="hello"),
            tune_config=tune.TuneConfig(
                num_samples=20,
                scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
            ),
            run_config=RunConfig(storage_path="/Users/yk.iwabuchi/results", name="experiment_mnist"),
            param_space=search_space,
        )
        results = tuner.fit()
        # print(results.get_best_result("mean_accuracy", mode="max").path)

        # Obtain a trial dataframe from all run trials of this `tune.run` call.
        dfs = {result.path: result.metrics_dataframe for result in results}
        # Plot by epoch
        ax = None  # This plots everything on the same plot
        for d in dfs.values():
            ax = d.mean_accuracy.plot(ax=ax, legend=False)
            plt.show()

    @staticmethod
    def tuning(trainable, train_dataloader, valid_dataloader, valid_interval):
        from ray import train, tune
        from ray.tune.schedulers import ASHAScheduler
        import torch.nn.functional as F
        import numpy as np
        import torch
        import torch.optim as optim
        import torch.nn as nn
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        import matplotlib.pyplot as plt

        from ray.train import RunConfig
        search_space = {
                "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
                "momentum": tune.uniform(0.1, 0.9),
        }

        tuner = tune.Tuner(
            tune.with_parameters(trainable, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, valid_interval=valid_interval),
            tune_config=tune.TuneConfig(
                num_samples=20,
                scheduler=ASHAScheduler(metric="train_loss", mode="max"),
            ),
            run_config=RunConfig(storage_path="/Users/yk.iwabuchi/results", name="experiment_mnist"),
            param_space=search_space,
        )
        results = tuner.fit()
        # print(results.get_best_result("mean_accuracy", mode="max").path)

        # Obtain a trial dataframe from all run trials of this `tune.run` call.
        dfs = {result.path: result.metrics_dataframe for result in results}
        # Plot by epoch
        ax = None  # This plots everything on the same plot
        for d in dfs.values():
            ax = d.train_loss.plot(ax=ax, legend=False)
            plt.show()



import argparse


class BArgparse(object):

    @staticmethod
    def get_base_parser():
        base_parser = argparse.ArgumentParser(add_help=False)
        base_parser.add_argument('--log_dir', type=str, default=f"../logs/{BLogger.make_date_log_directory()}", help="log directory specify")
        base_parser.add_argument('--log_file', type=str, default=BLogger.make_date_log_directory(), help="log file specify")
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

    @staticmethod
    def get_ml_args():
        ml_parser = argparse.ArgumentParser(parents=[BArgparse.get_base_parser()])
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

    @staticmethod
    def get_dl_args():
        dl_parser = argparse.ArgumentParser(parents=[BArgparse.get_base_parser()])
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

# def Bfzfprompt4path (path: str):
#     from pyfzf.pyfzf import FzfPrompt
#     #fzf


def BVideoWrapper(func):
    """fzf prompt
        frame processing
    ----------

    ----------
    Parameters
    ----------
    func : processing of frame
    ----------
    Return
    ----------

    ----------
    Example
    ----------
    @VideoWrapper
    def gaussianprocessing(frame):
        kernel = np.ones((5, 5), np.float32)/25
        dst = cv.filter2D(frame, -1, kernel)
    ----------
    Reference
    ----------
    """

    def wrapper(*args, **kwargs):
        import cv2 as cv
        cap = cv.VideoCapture(args[0])
        if (cap.isOpened() is False):
            print("Error openening video stream or file")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                # Display the resulting frame
                processing_imgs = func(frame)
                displayed_imgs = np.hstack((frame, *processing_imgs))
                cv.namedWindow("processing image", cv.WND_PROP_FULLSCREEN)
                cv.setWindowProperty("processing image", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

                cv.imshow('processing image', displayed_imgs)

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

        # func(results, **kwargs)
    return wrapper


@BVideoWrapper
def sampleblur(frame):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv.filter2D(frame, -1, kernel)
    return [dst]
import cv2 as cv
sampleblur("Mountain.mp4")


def Bshow_video(path: str):
    """
        Arguments:
            - path: video path
        Reference: https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        Description:
            This is VideoShow
    """
    import cv2 as cv
    cap = cv.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
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
# Reference> https://www.youtube.com/watch?v=mcT_bK1px_g

import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QGridLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PySide6 import QtGui

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        #self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel{
        border: 4px dashed # aaa
        }
                           ''')

    def setPixmap(self, image):
        super().setPixmap(image)


class BQt(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        self.mainLayout = QVBoxLayout()
        #self.mainLayout = QGridLayout()

        self.photoViewer = ImageLabel()
        self.mainLayout.addWidget(self.photoViewer)

        self.setLayout(self.mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()

        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()

        else:
            event.ignore()


    def dropEvent(self, event):
        if event.mimeData().hasImage:
            #event.setDropAction(Qt.CopyAction)
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                self.add_image(file_path)

            event.accept()

        else:
            event.ignore()


    def add_image(self, file_path):
        label = ImageLabel()
        label.setPixmap(QPixmap(file_path))

        self.mainLayout.addWidget(label)

        #self.photoViewer.setPixmap(QPixmap(file_path))



app = QApplication(sys.argv)
bqt = BQt()
bqt.show()
sys.exit(app.exec())


#def BImage():


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
# from torchvision.transforms import v2
from torchvision import transforms
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
        print(img_path)
        image = read_image(img_path)
        print(img_path)
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
            # print(f"fold: {fold}_______________________")
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

    def loop_cross_validation(self):
        for fold_n in self.split_n_lst:
            train_dataset, valid_dataset = self.packet_dataset[fold_n]
            train_dataloader = DataLoader(train_dataset, 5, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, 5, shuffle=True)
            valid_interval = 5
            BTune.tuning(self._subset, train_dataloader, valid_dataloader, valid_interval)
            # self._subset(train_dataloader, valid_dataloader, valid_interval)

    def _subset(self, config, train_dataloader, valid_dataloader, valid_interval):
        total_train_loss = []
        total_valid_loss = []
        model = self.model
        # optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

        for epoch in range(1, self.epochs+1):
            train_loss, train_accuracy = self.train_one_epoch(model, optimizer, train_dataloader)
            total_train_loss.append(train_loss)
            # train.report({"train_mean_accuracy": train_accuracy})
            train.report({"train_loss": train_loss})
            if epoch % valid_interval == 0:
                valid_loss, valid_accuracy = self.valid_one_epoch(model, optimizer, valid_dataloader)
                total_valid_loss.append(valid_loss)
        # BMatplotlib.loss_curve(total_train_loss, total_valid_loss, valid_interval, f"{str(fold_n).zfill(2)}.png")


    # @classmethod
    def train_one_epoch(self, model, optimizer, dataloader):
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
        total = 0
        model.train()

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for images, labels in dataloader:
            # print(images.size())
            labels = torch.tensor(labels, dtype=torch.long)
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()           # Zero your gradients for every batch!
            outputs = model(images)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_loss += loss.item() * images.size(0)      # total of size of minibatch

            scores, predictions = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predictions == labels).sum().item()
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(train_dataloader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.

        return train_loss, train_correct / total

    def valid_one_epoch(self, model, optimizer, dataloader):
        valid_loss, val_correct, total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                labels = torch.tensor(labels, dtype=torch.long)
                output = self.model(images)
                loss = self.loss_fn(output, labels)
                valid_loss += loss.item()*images.size(0)
                total += labels.size(0)
                scores, predictions = torch.max(output.data, 1)
                val_correct += (predictions == labels).sum().item()
        return valid_loss, val_correct / total


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
    def instant():
        """TL;DR
        gulliver code!!
        main_mpl_plot() code!!
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
        figsize = (16, 9)
        title = "sin(x) vs cos(x) vs sin(x)*cos(x)"
        xlim, ylim = None, None
        xlabel, ylabel = "x-axis", "y-axis"
        show = False
        save_fn = "/tmp/a.png"
        legend = ("upper right", "func type") # (loc, label)
        opts = dict(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, show=show, save_fn=save_fn, legend=legend, figsize=figsize)
        plt_decorate(**opts) # XXX: 
        OPEN(save_fn)


    def plt_decorate(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, show=True, save_fn=None, grid="xy", ax=None, fig=None, legend=None, yscale=None, save_pdf=False, close=True, title_fs=None, hist_centered=False, figsize=None) :
        """TL;DR
        gulliver code!!
        gutils_model_params_info_sparsity.py
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
        #    legend = ("upper right", "weights")
        ax = ax or plt.gca()
        if title:
            ax.set_title(title, fontsize=title_fs)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if yscale is not None:
            plt.yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if 1:
            # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
            # ax.xaxis.get_ticklabels()
            axis = None
            if grid not in "xy x y both".split():
                print(__file__, "grid should be one of 'xy x y both'", f"<{grid}> given")
            if grid == "xy":
                axis = "both"
            elif grid in ("x", "y"):
                axis = grid
# XXX: 19/Sep/2023 @ Tue 22:20:01
   #     axis = "y" if axis and len(ax.xaxis.get_ticklabels()) > 20 else axis
            plt.grid(axis=axis, **GRID_OPTS)
        if legend is None:
            legend = ("upper right", "")
        if legend:
# Set legend position when plotting a pandas dataframe with a second y-axis via pandas plotting interface [duplicate]
# 	https://stackoverflow.com/questions/54090983/set-legend-position-when-plotting-a-pandas-dataframe-with-a-second-y-axis-via-pa
            try:
                h1, l1 = ax.get_legend_handles_labels()
                h2 = l2 = []
                try:
                    h2, l2 = ax.right_ax.get_legend_handles_labels()
                except AttributeError: pass
                ax.legend(h1+h2, l1+l2, loc=legend[0], title=legend[1])
            except:
                ax.legend(loc=legend[0], title=legend[1])
        fig = fig or plt.gcf();
        fig.canvas.draw()
        # XXX: 30/Aug/2023 @ Wed 18:35:14 
        if figsize:
            fig = plt.gcf()
            fw, fh = figsize
            fig.set_figwidth(fw)
            fig.set_figheight(fh)
        plt.tight_layout()
        if hist_centered:
            histogram_centered(ax=ax)       # XXX: 24/Jul/2022 @ Sun 11:40:46 
        if save_fn:
            fig = fig or plt.gcf()
            try : # AttributeError: 'Figure' object has no attribute 'set_linewidth'
                edgewidth, edgecolor = 1, "#04253a"
                fig.set_linewidth(edgewidth)
                fig.set_edgecolor(edgecolor)
            except AttributeError:
                pass
            opts = dict(edgecolor=fig.get_edgecolor())
            plt.savefig(save_fn, **opts)
            save_fn_pdf = Path(save_fn).with_suffix(".pdf")
            if save_pdf : plt.savefig(save_fn_pdf, **opts)
        if show: plt.show()
        else :
            if close : plt.close(plt.gcf())
        return




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




# -------------------------------------------

class BMetrics(object):
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

# -------------------------------------------


# model zoo

# class Bzoo(object):
class ConvNet(nn.Module):
    # Reference: https://docs.ray.io/en/latest/tune/getting-started.html
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
















# -------------------------------------------







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
