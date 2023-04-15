#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	random
# Author: 8ucchiman
# CreatedDate:  2023-01-29 15:33:09 +0900
# LastModified: 2023-01-29 15:41:04 +0900
# Reference: https://note.nkmk.me/python-random-randrange-randint/
#


from random import random, uniform, randrange, randint

print(random())                 # 0.0以上1.0以下の乱数生成 -> float
print(uniform(100.0, 200.0))    # 指定した範囲内で乱数生成 -> float


# ベータ分布: random.betavariate()
# 指数分布: random.expovariate()
# ガンマ分布: random.gammavariate()
# ガウス分布: random.gauss()
# 対数正規分布: random.lognormvariate()
# 正規分布: random.normalvariate()
# フォン・ミーゼス分布: random.vonmisesvariate()
# パレート分布: random.paretovariate()
# ワイブル分布: random.weibullvariate()


print(randrange(5, 20, 4))      # [5, 9, 13, 17]内で整数生成

print(randint(5, 20))           # 指定の範囲内で乱数生成 -> int
