#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     Bmods
# Author:       8ucchiman
# CreatedDate:  2023-07-27 13:18:37
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#

############################################################################################

import os
import cv2 as cv
import numpy as np
import scipy as cp
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from gutils_misc import OPEN, RUN, CPR, LNS, MKDIR, RMRF, RMRF_FULL
from gutils_misc import CAT, HEAD, TAIL, HEAD_TAIL, TOUCH, LINE
from gutils_misc import HOME, lena, gm_im_dir, gm_im_lst, gm_im_calib_dir
from gutils_misc import get_latest_file

from gutils_misc import nx_info, nx_shape, nx_npz, nx_npz_load, nx_dir, nx_dict, nx_cfg, notqdm

from gutils_time import gTicToc, time_func, FPS, gFPS_fixed

from gutils_im import im_grid, asIm, asArray
from gutils_tands import im_movie, cv2_noise

from gutils_model_params_info_sparsity import plt_decorate, from_numpy, to_numpy
import matplotlib.pyplot as plt



def Bucchiman_was_here ():
    """
        Description: for debug to check pythonpath
    """
    print("8ucchiman was here!!!!!")


def Bshow_image(path: str):
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

def Bshow_triad():
    pass

def Bshow_pairs(a, b):
    """
        show pair images
        Arguments:
            - a: image A
            - b: image B
    """
    concat_images = cv.hconcat([a, b])
    cv.imshow("image", concat_images)


def Bshow_video(path: str):
    """
        Arguments:
            - path: video path
        Reference: https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        Description:
            This is VideoShow
    """
    cap = cv.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
 
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
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


def Bimage_filter(func):
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
def gaussianfilter(image):
    return cv.GaussianBlur(image, (5, 5), 0)

@Gtimer
@Bimage_filter
def medianfilter(image):
    return cv.medianBlur(image, 7)

@Gtimer
@Bimage_filter
def bilateralfilter(image):
    return cv.bilateralFilter(image, 9, 75, 75)



################################################################



################################################################



################################################################
# gabor filter
################################################################
def Bgabor_filter (img):
    """
        Reference: https://github.com/intsynko/gabor_dashboard/tree/main
    """

    # read img and set gray color
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # dashboard settings
    fig, (ax, ax_2) = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.25, bottom=0.45)
    
    # create slider spaces
    axcolor = 'lightgoldenrodyellow'
    ax_sliders = [plt.axes([0.25, 0.1 + 0.05 * i, 0.65, 0.03], facecolor=axcolor) for i in range(6)]
    
    # define parameter sliders
    ksize = Slider(ax_sliders[0], 'ksize', 1, 40, valinit=21, valstep=1)
    sigma = Slider(ax_sliders[1], 'sigma', 0.1, 20.0, valinit=8, valstep=0.1)
    lambd = Slider(ax_sliders[2], 'lambd', 0.1, 20.0, valinit=10, valstep=0.1)
    gamma = Slider(ax_sliders[3], 'gamma', 0.1, 1, valinit=0.5, valstep=0.05)
    psi = Slider(ax_sliders[4], 'psi', -10, 10, valinit=0, valstep=1)
    theta = Slider(ax_sliders[5], 'theta', -5, 5, valinit=0, valstep=0.1)
    
    sliders = [ksize, sigma, lambd, gamma, psi, theta]
    
    
    def update(val):
        # on slider update recalculate gabor kernel
        g_kernel = cv.getGaborKernel(ksize=(ksize.val, ksize.val),
                                      sigma=sigma.val,
                                      theta=np.pi / 4 * theta.val,
                                      lambd=lambd.val,
                                      gamma=gamma.val,
                                      psi=psi.val,
                                      ktype=cv.CV_32F)
        # recalculate img result
        res = cv.filter2D(img, cv.CV_8UC3, g_kernel)
    
        # show new img and gabor kernel
        ax.imshow(res, interpolation="nearest", cmap='gray')
        ax.set_title('gabor result on img', fontsize=10)
        ax_2.imshow(g_kernel, interpolation="nearest", cmap='gray')
        ax_2.set_title('g_kernel', fontsize=10)
    
    
    for i in sliders:
        i.on_changed(update)
    
    update(None)
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    
    def reset(event):
        for slider in sliders:
            slider.reset()


    button.on_clicked(reset)
    plt.show()

################################################################


def Bimagegrid(func):
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
def _run():
    # toDo
    pass


############################################################################################
# toDo pytorch

############################################################################################
# toDo matplotlib
def Bcalc_histgram():
    pass


############################################################################################
# toDo pandas

############################################################################################
# toDo hagging face

############################################################################################
# toDo mlflow

############################################################################################
# toDo scikit-learn


def _sample_kmeans():
    """
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    from sklearn.cluster import KMeans
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    kmeans.labels_
    kmeans.predict([[0, 0], [12, 3]])
    kmeans.cluster_centers_
    pass



############################################################################################
# toDo onnx

############################################################################################
# toDo others

############################################################################################
# Fractal
def Bcreate_fractal_image():
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

# gutils
def _main_mpl_plot():
    from gutils_howto import main_mpl_plot
    main_mpl_plot()




if __name__ == "__main__":
    _HOME = os.environ["HOME"]

    image_lst = ["baboon.jpg", "board.jpg", "building.jpg", "fruits.jpg",
                 "lena.jpg", "nujabes.jpg", "Nujabes.webp", "nujabes_illust.jpeg", "stuff.jpg", "rain.jpg"]
    video_lst = ["Mountain.mp4", "slow_traffic_small.mp4", "vtest.avi"]
    # image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[4])
    image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[9])
    # print(IMAGE_NAME)
    # show_image(IMAGE_NAME)
    # show_video(os.path.join(_HOME, ".config/sample/videos", video_lst[0]))
    # add(2, 4)
    image = cv.imread(image_path)
    # gaussianfilter(image)

    # gabor_filter(image)
    _main_mpl_plot()
