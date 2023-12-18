#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     file
# Author:       8ucchiman
# CreatedDate:  2023-10-14 14:09:06
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
#               https://longervision.github.io/2017/03/16/ComputerVision/OpenCV/opencv-internal-calibration-chessboard/
# Description:  ---
#


import cv2
import numpy as np


def get_args():
    """
        arguments
            - img_path
    """
    import argparse
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--img_path', type=str, required=True, help="入力画像")
    # parser.add_argument('--method_name', type="str", default="make_date_log_directory", help="method name here in utils.py")

    # parser.add_argument('arg1')     # 必須の引数
    # parser.add_argument('-a', 'arg')    # 省略形
    # parser.add_argument('--flag', action='store_true')  # flag
    # parser.add_argument('--strlist', required=True, nargs="*", type=str, help='a list of strings') # --strlist hoge fuga geho
    # parser.add_argument('--method', type=str)
    # parser.add_argument('--fruit', type=str, default='apple', choices=['apple', 'banana'], required=True)
    # parser.add_argument('--address', type=lambda x: list(map(int, x.split('.'))), help="IP address") # --address 192.168.31.150 --> [192, 168, 31, 150]
    # parser.add_argument('--colors', nargs='*', required=True)
    args = vars(base_parser.parse_args())
    return args


def show_image(img):
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def harris_corner_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()] = [0, 0, 255]
    show_image(img)


def main():
    args = get_args()
    img = cv2.imread(args["img_path"])
    harris_corner_detection(img)



if __name__ == "__main__":
    main()
