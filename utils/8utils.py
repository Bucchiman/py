#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     8utils
# Author:       8ucchiman
# CreatedDate:  2023-10-18 16:29:48
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#



import abc


class IData(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def format_data(self):
        """
            format data
                sklearn: np.array([[],[],[]])
        """


    @abc.abstractmethod
    def input_data(self)
