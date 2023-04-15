#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	datasets
# Author: 8ucchiman
# CreatedDate:  2023-02-17 13:10:07 +0900
# LastModified: 2023-02-17 14:53:56 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
from torch.utils.data import Dataset
from skimage import io, transform
# import utils
# from import utils import get_args, get_logger
# import numpy as np
import pandas as pd


class ImageDataset(Dataset):
    def __init__(self, imgs_path: str, label_path: str, transform=None):
        """
        Args:
            img_path (string): 画像のパス
            label_path (string): ラベルファイル
        """
        self.imgs_path = imgs_path
        self.label_csv = pd.read_csv(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.label_csv)

    def __getitem__(self, idx):
        img_name, label = self.label_csv.iloc[idx, :]
        img_path = os.path.join(self.imgs_path, img_name)

        image = io.imread(img_path)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample


def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    pass


if __name__ == "__main__":
    main()
