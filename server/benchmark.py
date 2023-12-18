#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     benchmark
# Author:       8ucchiman
# CreatedDate:  2023-07-04 23:36:34
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import time
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


loop = 500
concurrency = 5
endpoints = {
        "python": "http://127.0.0.1:8171",
        "rust": "http://127.0.0.1:1718",
}

def request(i, url):
    try:
        response = urllib.request.urlopen(url)
        body = response.read().decode("utf-8")
        print(i, body)

    except Exception as e:
        print(e)


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    for k, v in endpoints.items():
        print("request to {} server".format(k))
        start = time.time()
        with ThreadPoolExecutor(max_workers = concurrency) as executor:
            for i in range(loop):
                executor.submit(request, i=i, url=v)
        print("{} elapsed: {}".format(k, time.time() - start))
    pass


if __name__ == "__main__":
    main()

