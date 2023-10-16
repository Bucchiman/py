#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     sample
# Author:       8ucchiman
# CreatedDate:  2023-10-06 17:06:09
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()

fib(1000)
