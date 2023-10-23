#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     1D_discrete_fourier_transforms
# Author:       8ucchiman
# CreatedDate:  2023-07-27 13:18:37
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    https://docs.scipy.org/doc/scipy/tutorial/fft.html
# Description:  ---
#


from scipy.fft import fft, ifft
import numpy as np


def one_dim_fft_ifft(x):
    """
    """
    y = fft(x)
    print(y)
    yinv = ifft(y)
    print(yinv)


def main():
    x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
    y = fft(x)
    print()
    pass


if __name__ == "__main__":
    main()
