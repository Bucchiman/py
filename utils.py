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
