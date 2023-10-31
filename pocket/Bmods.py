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
import inspect
from time import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV


def reflection_methods (target_obj):
    """
    Description

    Parameters

    Return

    Reference
        https://blog.goo.ne.jp/dak-ikd/e/fa1766c5ded886ac513ef3e761551560
    """
    import inspect
    function_lst = list(map(lambda x: x[0], inspect.getmembers(target_obj)))

    return function_lst


def Bucchiman_was_here ():
    """
        Description: for debug to check pythonpath
    """
    print("8ucchiman was here!!!!!")


# Do you like gulliver codes??????????????????
if 0:
    from gutils_misc import OPEN, RUN, CPR, LNS, MKDIR, RMRF, RMRF_FULL
    from gutils_misc import CAT, HEAD, TAIL, HEAD_TAIL, TOUCH, LINE
    from gutils_misc import WHOAMI, MYTMP
    from gutils_misc import HOME, lena, gm_im_dir, gm_im_lst, gm_im_calib_dir
    from gutils_misc import get_latest_file
    
    from gutils_misc import nx_info, nx_shape, nx_npz, nx_npz_load, nx_dir, nx_dict, nx_cfg, notqdm

    from gutils_time import gTicToc, time_func, FPS, gFPS_fixed

    from gutils_im import im_grid, asIm, asArray
    from gutils_tands import im_movie, cv2_noise

    from gutils_model_params_info_sparsity import plt_decorate, from_numpy, to_numpy



if 1:
    from gutils_misc import nx_npz, nx_npz_load
    from gutils_numpy import np_savetxt, np_loadtxt
    from gutils_grid import loadnpz, savenpz, asArray

    from gutils_im import asIm
    from gutils_ocv import calibrate_camera

    from gutils_model_params_info_sparsity import plt_decorate, axtext, axvline, axhline, axhline_mean_std
    from gutils_model_params_info_sparsity import from_numpy, to_numpy


def Bshow_image (path: str):
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

def Bshow_triad ():
    pass

def Bshow_pairs (a, b):
    """
        show pair images
        Arguments:
            - a: image A
            - b: image B
    """
    concat_images = cv.hconcat([a, b])
    cv.imshow("image", concat_images)


def Bfzfprompt(func):
    def wrapper(*args, **kwargs):
        from pyfzf.pyfzf import FzfPrompt
        fzf = FzfPrompt()

        results = fzf.prompt(args[0])        # ['hoge', 'kie', 'becori']
        func(results, args[1], **kwargs)

    return wrapper

@Bfzfprompt
def _run_bfzfprompt(lst, obj):
    method = getattr(obj, lst[0])
    method()


#_run_bfzfprompt(['hoge', 'kie', 'becori'])

def Bfzfprompt4path (path: str):
    from pyfzf.pyfzf import FzfPrompt
    #fzf

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


def Bimage_filter (func):
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
def gaussianfilter (image):
    return cv.GaussianBlur(image, (5, 5), 0)

@Gtimer
@Bimage_filter
def medianfilter (image):
    return cv.medianBlur(image, 7)

@Gtimer
@Bimage_filter
def bilateralfilter (image):
    return cv.bilateralFilter(image, 9, 75, 75)



################################################################
# gpio

class Sample4GPIO (object):
    def _gpio_mcp3008_3002 (self):
        """
        # Reference: https://www.denshi.club/pc/raspi/5raspberry-pi-zeroiot9a-d6mcp3002.html
                     https://qiita.com/M_Study/items/fc0df5069c76418ef7e2
                     https://101010.fun/raspberry-pi-adc-mcp3008.html
        CH0	1   16  Vdd     3.3V電源
        CH1	2   15  Vref    基準電圧入力
        CH2	3   14  AGND    アナログ・グラウンド
        CH3	4   13  CLK
        CH4	5   12  Dout
        CH5	6   11  Din
        CH6	7   10  /CS・SHDN
        CH7	8    9  DGND    ディジタル・グラウン
        """


################################################################
# fourier




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
    
    
    def update (val):
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
    
    
    def reset (event):
        for slider in sliders:
            slider.reset()


    button.on_clicked(reset)
    plt.show()

################################################################


def Bimagegrid (func):
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
def _run ():
    # toDo
    pass


def scraping ():
    from urllib.request import urlopen
    url = "https://matplotlib.org/stable/gallery/index"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    title_index = html.find("<title>")

############################################################################################
# toDo pytorch

############################################################################################
# toDo matplotlib
def Bcalc_histgram():
    pass

class Sample4Matplotlib (object):
    def _mpl_axes_plot (self):
        """
        Description
    
    
        Parameters
        ----------
        ----------
        Return
            image (show)
    
        Reference
            https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_props.html#sphx-glr-gallery-subplots-axes-and-figures-axes-props-py
        """
    
        import matplotlib.pyplot as plt
        import numpy as np
    
        t = np.arange(0.0, 2.0, 0.01)
        s = np.sin(2 * np.pi * t)
    
        fig, ax = plt.subplots()
        ax.plot(t, s)
    
        ax.grid(True, linestyle='-.')
        ax.tick_params(labelcolor='r', labelsize='medium', width=3)
    
        plt.show()
    
    
    def _mpl_zoomin_zoomout (self):
        """
        Description
            zoomin zoomout
    
        Parameters
        ----------
        ----------
        Return
            image (show)
    
        Reference
            https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_margins.html#sphx-glr-gallery-subplots-axes-and-figures-axes-margins-py
        """
    
        import matplotlib.pyplot as plt
        import numpy as np
    
        from matplotlib.patches import Polygon
    
    
        def f(t):
            return np.exp(-t) * np.cos(2*np.pi*t)
    
    
        t1 = np.arange(0.0, 3.0, 0.01)
    
        ax1 = plt.subplot(212)
        ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
        ax1.plot(t1, f(t1))
    
        ax2 = plt.subplot(221)
        ax2.margins(2, 2)           # Values >0.0 zoom out
        ax2.plot(t1, f(t1))
        ax2.set_title('Zoomed out')
    
        ax3 = plt.subplot(222)
        ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
        ax3.plot(t1, f(t1))
        ax3.set_title('Zoomed in')
    
        plt.show()
    
    def _mpl_axes_demo (self):
        """
        Description
            
    
        Parameters
        Return
            image (show)
        Reference
            https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        np.random.seed(19680801)  # Fixing random state for reproducibility.
    
        # create some data to use for the plot
        dt = 0.001
        t = np.arange(0.0, 10.0, dt)
        r = np.exp(-t[:1000] / 0.05)  # impulse response
        x = np.random.randn(len(t))
        s = np.convolve(x, r)[:len(x)] * dt  # colored noise
    
        fig, main_ax = plt.subplots()
        main_ax.plot(t, s)
        main_ax.set_xlim(0, 1)
        main_ax.set_ylim(1.1 * np.min(s), 2 * np.max(s))
        main_ax.set_xlabel('time (s)')
        main_ax.set_ylabel('current (nA)')
        main_ax.set_title('Gaussian colored noise')
    
        # this is an inset axes over the main axes
        right_inset_ax = fig.add_axes([.65, .6, .2, .2], facecolor='k')
        right_inset_ax.hist(s, 400, density=True)
        right_inset_ax.set(title='Probability', xticks=[], yticks=[])
    
        # this is another inset axes over the main axes
        left_inset_ax = fig.add_axes([.2, .6, .2, .2], facecolor='k')
        left_inset_ax.plot(t[:len(r)], r)
        left_inset_ax.set(title='Impulse response', xlim=(0, .2), xticks=[], yticks=[])
    
        plt.show()
    
    
    def _mpl_shared_square_axes (self):
        """
        Description
            share y axies
    
        Parameters
    
        Return
    
        Reference
            https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_box_aspect.html#sphx-glr-gallery-subplots-axes-and-figures-axes-box-aspect-py
        """
    
        fig2, (ax, ax2) = plt.subplots(ncols=2, sharey=True)
    
        ax.plot([1, 5], [0, 10])
        ax2.plot([100, 500], [10, 15])
    
        ax.set_box_aspect(1)
        ax2.set_box_aspect(1)
    
        plt.show()
    
    def _mpl_box_aspect (self):
        """
        Description
    
        Parameters
    
        Return      empty box
    
        Reference
        """
    
        import matplotlib.pyplot as plt
        import numpy as np
    
        fig1, ax = plt.subplots()
    
        ax.set_xlim(300, 400)
        ax.set_box_aspect(1)
    
        plt.show()
    
    
    def _mpl_sample_align_labels (self):
        """
        Description
    
        Parameters
        Reference
            https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html#sphx-glr-gallery-subplots-axes-and-figures-align-labels-demo-py
        """
    
        import matplotlib.pyplot as plt
        import numpy as np
    
        import matplotlib.gridspec as gridspec
    
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(2, 2)
    
        ax = fig.add_subplot(gs[0, :])
        ax.plot(np.arange(0, 1e6, 1000))
        ax.set_ylabel('YLabel0')
        ax.set_xlabel('XLabel0')
    
        for i in range(2):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(np.arange(1., 0., -0.1) * 2000., np.arange(1., 0., -0.1))
            ax.set_ylabel('YLabel1 %d' % i)
            ax.set_xlabel('XLabel1 %d' % i)
            if i == 0:
                ax.tick_params(axis='x', rotation=55)
        fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
        plt.show()
    
    
    def _mpl_sample_user_defined (self):
        """
        Description
            user defined on labels (ADVANCED)
    
        Parameters
    
        Return
    
        Reference
        """
        
        import matplotlib.pyplot as plt
    
        import matplotlib.transforms as mtransforms
    
        fig, ax = plt.subplots()
        ax.plot(range(10))
        ax.set_yticks([2, 5, 7], labels=['really, really, really', 'long', 'labels'])
    
    
        def on_draw(event):
            bboxes = []
            for label in ax.get_yticklabels():
                # Bounding box in pixels
                bbox_px = label.get_window_extent()
                # Transform to relative figure coordinates. This is the inverse of
                # transFigure.
                bbox_fig = bbox_px.transformed(fig.transFigure.inverted())
                bboxes.append(bbox_fig)
            # the bbox that bounds all the bboxes, again in relative figure coords
            bbox = mtransforms.Bbox.union(bboxes)
            if fig.subplotpars.left < bbox.width:
                # Move the subplot left edge more to the right
                fig.subplots_adjust(left=1.1*bbox.width)  # pad a little
                fig.canvas.draw()
    
    
        fig.canvas.mpl_connect('draw_event', on_draw)
    
        plt.show()
    
    
    
    def _mpl_sample_plot (self):
        """
            Reference: https://matplotlib.org/stable/plot_types/basic/plot.html#sphx-glr-plot-types-basic-plot-py
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        plt.style.use('_mpl-gallery')
    
        # make data
        x = np.linspace(0, 10, 100)
        y = 4 + 2 * np.sin(2 * x)
    
        # plot
        fig, ax = plt.subplots()
    
        ax.plot(x, y, linewidth=2.0)
    
        ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
               ylim=(0, 8), yticks=np.arange(1, 8))
    
        plt.show()
    
    def _mpl_sample_scatter (self):
        """
            Reference: https://matplotlib.org/stable/plot_types/basic/scatter_plot.html#sphx-glr-plot-types-basic-scatter-plot-py
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        plt.style.use('_mpl-gallery')
    
        # make the data
        np.random.seed(3)
        x = 4 + np.random.normal(0, 2, 24)
        y = 4 + np.random.normal(0, 2, len(x))
        # size and color:
        sizes = np.random.uniform(15, 80, len(x))
        colors = np.random.uniform(15, 80, len(x))
    
        # plot
        fig, ax = plt.subplots()
    
        ax.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
    
        ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
               ylim=(0, 8), yticks=np.arange(1, 8))
    
        plt.show()
    
    def _mpl_sample_bar (self):
        """
            Reference: https://matplotlib.org/stable/plot_types/basic/bar.html#sphx-glr-plot-types-basic-bar-py
        """
        import matplotlib.pyplot as plt
        import numpy as np
    
        plt.style.use('_mpl-gallery')
    
        # make data:
        x = 0.5 + np.arange(8)
        y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]
    
        # plot
        fig, ax = plt.subplots()
    
        ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
    
        ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
               ylim=(0, 8), yticks=np.arange(1, 8))
    
        plt.show()


#def __Sample_matplotlib ():

############################################################################################
# toDo pandas

############################################################################################
# toDo hagging face

############################################################################################
# toDo mlflow

############################################################################################
# toDo scikit-learn

def dataloader ():
    sample_dataloader = [""]
    fetch_openm

def _sample_PCA ():
    """
        n_components: number of components
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    import numpy as np
    from sklearn.decomposition import PCA
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    print(f"pca.explained_variance_ratio_: {pca.explained_variance_ratio_}")
    print(f"pca.singular_values_: {pca.singular_values_}")


def _sample_kmeans ():
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
def Bcreate_fractal_image ():
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

# epipolar

def create_epipolar ():
    """
        Reference:  https://docs.opencv.org/4.8.0/da/de9/tutorial_py_epipolar_geometry.html
                    https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
        Code:       https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/epipolar.py
        Image:      https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/left.jpg
                    https://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_calib3d/py_epipolar_geometry/right.jpg
    """
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1]])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    ##########

    # Instead SIFT+FLANN, we use SIFT+BFMatch

    img1 = cv2.imread('/home/yk.iwabuchi/.config/sample/pics/epipolar_sample_left.jpg',0)  # left
    img2 = cv2.imread('/home/yk.iwabuchi/.config/sample/pics/epipolar_sample_right.jpg',0) # right

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # Brute-Force Matcher
    bf = cv2.BFMatcher()

    # 
    matches = bf.knnMatch(des1, des2, k=2)

    # 
    ratio = 0.65
    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    #------------------------------

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


############################################################################################

# gutils
def _sample_main_mpl_plot ():
    from gutils_howto import main_mpl_plot
    main_mpl_plot()


def _sample_main_fps ():
    from gutils_howto import main_fps
    main_fps()


# misc utils, cp, lns, rm, head, tail, ...
def _sample_main_misc ():
    '''
    misc utils, cp, lns, rm, head, tail, ...

    touch src; cp src dst; rm src; ln -s src dst; 
    '''
    from gutils_misc import OPEN, RUN, CPR, LNS, MKDIR, RMRF, RMRF_FULL
    from gutils_misc import CAT, HEAD, TAIL, HEAD_TAIL, TOUCH, LINE
    from gutils_misc import WHOAMI, MYTMP

    tmp = MYTMP
    print(tmp)
    src = f"{tmp}/xxx.txt"
    dst = f"{tmp}/yyy.txt"
    lns = f"{tmp}/zzz.txt"
    TOUCH(src)
    CPR(src, dst)
    RMRF(src)
    LNS(dst, lns)



def main_mpl_plot ():

    pass


def main_fps_fixed ():
    pass


def main_fps ():
    from gutils_time import FPS # XXX: 
    from time import sleep
    ''' 
    report FPS for a process

    class: FPS
        Arguments:
            fps: 5
            msg: "fps label"
        FPS.tick()
            

    output:
        0 1693398288.7831407
        1 1693398289.2840245
        2 1693398289.7843204
        3 1693398290.2849793
        [   2 fps] fps label4 1693398290.7868018
        5 1693398291.2874384
        6 1693398291.788087
        7 1693398292.2887347
        8 1693398292.7893777
        [   2 fps] fps label9 1693398293.2908878
    '''
    fps = FPS(5, msg="fps label")
    for i in range(10):
        fps.tick()
        print(i, time())
        sleep(0.5)
    pass

############################################################################################




if __name__ == "__main__":
    _HOME = os.environ["HOME"]

    image_lst = os.listdir(os.path.join(_HOME, ".config/sample/pics"))
    # image_lst = ["baboon.jpg", "board.jpg", "building.jpg", "fruits.jpg",
    #              "lena.jpg", "nujabes.jpg", "Nujabes.webp", "nujabes_illust.jpeg", "stuff.jpg", "rain.jpg"]
    # video_lst = ["Mountain.mp4", "slow_traffic_small.mp4", "vtest.avi"]
    video_lst = os.listdir(os.path.join(_HOME, ".config/sample/videos"))
    # image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[4])
    image_path = os.path.join(_HOME, ".config/sample/pics", image_lst[9])
    # print(IMAGE_NAME)
    # show_image(IMAGE_NAME)
    # show_video(os.path.join(_HOME, ".config/sample/videos", video_lst[0]))
    # add(2, 4)
    # image = cv.imread(image_path)
    # gaussianfilter(image)

    # Bgabor_filter(image)
    #_sample_main_mpl_plot()
    sample4matplotlib = Sample4Matplotlib()
    methods = reflection_methods(sample4matplotlib)
    _run_bfzfprompt(methods, sample4matplotlib)
