#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	utils
# CreatedDate:  2023-01-06 11:00:12 +0900
# LastModified: 2023-02-24 15:35:26 +0900
#


import os
import sys
from pathlib import Path
from pprint import pprint
import argparse
import logging
from logging import getLogger, config
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from hydra import compose, initialize_config_dir


def get_logger(file_name, log_dir="logs"):
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


def make_date_log_directory():
    return datetime.now().strftime(r"%Y_%m_%d_%H_%M")


def get_base_parser():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--log_dir', type=str, default="../logs", help="log directory specify")
    base_parser.add_argument('--log_file', type=str, default=make_date_log_directory(), help="log file specify")
    base_parser.add_argument('--config_dir', type=str, default="../params")
    base_parser.add_argument('--config_file', type=str, default="config.yaml")
    base_parser.add_argument('--results_dir', type=str, default="../results", help="results dir specify")
    base_parser.add_argument('--data_dir', type=str, default="../datas", help="data directory specify")
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
    dl_parser.add_argument('--train_img_dir', type=str, required=True)
    dl_parser.add_argument('--test_img_dir', type=str, required=True)
    dl_parser.add_argument('--train_label_file', type=str, required=True)
    dl_parser.add_argument('--batch_size', type=int, default=5)
    dl_parser.add_argument('--model_name', type=str, default='resnet18')
    dl_parser.add_argument('--gpus', type=str, default="all", choices=['all', 'cuda:0'])
    args = dl_parser.parse_args()
    return args


def make_barplot():
    x = np.arange()
    x_position = np.arange(len(x))
    y_one = np.array()
    y_two = np.array()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y_one, width=0.2, label="one")
    ax.bar(x_position+0.2, y_two, width=0.2, label="two")
    ax.legend()
    ax.set_xticks(x_position+0.2)
    ax.set_xticklabels(x)
    plt.show()
    fig.savefig("output.png")

def cos_similarity(x: np.array, y: np.array, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2)+eps)
    ny = y / np.sqrt(np.sum(y**2)+eps)
    return np.dot(nx, ny)


class Config(object):
    '''
        Config yaml
    '''
    @staticmethod
    def get_cnf(params_dir: str, config_file: str):
        '''
        @return
            cnf: OmegaDict
        '''
        with initialize_config_dir(version_base=None, config_dir=Path(params_dir).resolve()._str):
            cnf = compose(config_name=config_file)
            return cnf


class DDPSetUp(object):
    def __init__(self):
        pass

    @classmethod
    def setup(rank, world_size=torch.cuda.device_count()):
        '''
            Sets up the process group and configuration for PyTorch Distributed Data Parallelism
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    @classmethod
    def cleanup():
        '''
            Cleans up the distributed environment
        '''
        dist.destroy_process_group()

    @classmethod
    def run_mp(function, world_size):
        mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    '''
    args = get_args()
    method = getattr(utils, "make_date_log_directory")
    print(method())
    logger = get_logger(args.log_file)
    logger.info("hello")
    '''
    params = Config.get_cnf("../params")
    pprint(params, width=4)
