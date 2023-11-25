#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName:     tips
# Author:       8ucchiman
# CreatedDate:  2023-11-14 17:54:07
# LastModified: 2023-02-18 14:28:37 +0900
# Reference:    8ucchiman.jp
# Description:  ---
#


# -------------------------------------------

def make_monotone_color_image():
    # 2023-11-14 17時59分19秒
    # [reference](https://tat-pytone.hatenablog.com/entry/2019/03/03/102503)

    import numpy as np
    import cv2 as cv
    img = np.zeros((400, 600, 3), np.uint8)
    img[:, :, 0:3] = [153, 51, 204]
    cv.imwrite('purple.png', img)

# ------------------------------------------- 
# separate  R G B

# 2023-11-14 17時59分25秒
# [reference](https://potesara-tips.com/separating-colors/)

# ------------------------------------------- 
# numpy where

# 2023-11-14 17時59分28秒
# [numpy where docs](https://deepage.net/features/numpy-where.html)
# [reference](https://pystyle.info/opencv-mask-image/)



# ------------------------------------------- 
# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# 
# vid = cv2.VideoCapture('video.mp4')
# global data
# data = pd.read_csv('data.csv')
# 
# assert len(data) == cv2.CAP_PROP_FRAMES_COUNT()
# 
# 
# # write video
# dfi = df.iterrows()
# video = VideoFileClip(vid)
# out_video = video.fl_image(pipeline)
# out_video.write_videofile("vidout.mp4", temp_audiofile='temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")
# 
# def pipeline(frame):
# 
#     frame_index = next(dfi[0])
#     current_plot1 = plt.plot(df.time[:frame_index], df.data1[:frame_index]
#     current_plot2 = plt.plot(df.time[:frame_index], df.data2[:frame_index]
#     
#     # add plots to frame somehow here
#     frame = frame+current_plot1+current_plot2
#     return frame
# 
# 
# Reference: https://stackoverflow.com/questions/67077813/how-to-add-matplotlib-plot-below-and-image-in-opencv

# ------------------------------------------- 

def animated_images ():
    # Reference     https://matplotlib.org/stable/gallery/animation/dynamic_image.html
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.animation as animation

    fig, ax = plt.subplots()


    def f(x, y):
        return np.sin(x) + np.cos(y)

    x = np.linspace(0, 2 * np.pi, 120)
    y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(60):
        x += np.pi / 15
        y += np.pi / 30
        im = ax.imshow(f(x, y), animated=True)
        if i == 0:
            ax.imshow(f(x, y))  # show an initial one first
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    # To save the animation, use e.g.
    #
    # ani.save("movie.mp4")
    #
    # or
    #
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)

    plt.show()

# ------------------------------------------- 

def matplotlib_fig2image():
    """
    Description here

    Parameters
    ----------
    ----------
    Return
    Date                2023-11-16 10時42分20秒

    Reference           https://jun-networks.hatenablog.com/entry/2019/11/01/020536
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import cv2

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.linspace(-np.pi, np.pi)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot(x, np.sin(x), label="sin")

    ax.legend()
    ax.set_title("sin(x)")

    buf = io.BytesIO()  # インメモリのバイナリストリームを作成
    fig.savefig(buf, format="png", dpi=180)  # matplotlibから出力される画像のバイナリデータをメモリに格納する.
    buf.seek(0)  # ストリーム位置を先頭に戻る
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
    buf.close()  # ストリームを閉じる(flushする)
    img = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
    print(img.shape)
    cv2.imshow("sample", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -------------------------------------------

# animation image - histogram

# reference: https://nrsyed.com/2018/02/08/real-time-video-histograms-with-opencv-and-python/

# -------------------------------------------
# image processing

# Reference:        https://datacarpentry.org/image-processing/aio.html
# -------------------------------------------
# mouse event

# Reference:        https://docs.opencv.org/3.1.0/db/d5b/tutorial_py_mouse_handling.html

# -------------------------------------------
def mouse_event_matplotlib():
    # Reference:        https://matplotlib.org/stable/gallery/event_handling/coords_demo.html
    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib.backend_bases import MouseButton

    t = np.arange(0.0, 1.0, 0.01)
    s = np.sin(2 * np.pi * t)
    fig, ax = plt.subplots()
    ax.plot(t, s)


    def on_move(event):
        if event.inaxes:
            print(f'data coords {event.xdata} {event.ydata},',
                  f'pixel coords {event.x} {event.y}')


    def on_click(event):
        if event.button is MouseButton.LEFT:
            print('disconnecting callback')
            plt.disconnect(binding_id)


    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)

    plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# -------------------------------------------
# 


# Reference     https://www.tutorialspoint.com/how-to-convert-an-rgb-image-to-hsv-image-using-opencv-python


# -------------------------------------------
# find frame rate

#!/usr/bin/env python

def find_frame_rate():
    import cv2
    import time

    # Start default camera
    video = cv2.VideoCapture("Mountain.mp4");

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Number of frames to capture
    num_frames = 120;

    print("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()

    # Grab a few frames
    for i in range(0, num_frames) :
        ret, frame = video.read()

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))

    # Release video
    video.release()

# -------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pywt
# Generate the signal
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.cos(2.0 * np.pi * 7 * t) + np.sin(2.0 * np.pi * 13 * t)

# Apply DWT
coeffs = pywt.dwt(signal, 'db1')
cA, cD = coeffs

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(t, signal)
plt.title("Original Signal")
plt.subplot(1, 3, 2)
plt.plot(cA)
plt.title("Approximation Coefficients")
plt.subplot(1, 3, 3)
plt.plot(cD)
plt.title("Detail Coefficients")
plt.tight_layout()
plt.show()

# -------------------------------------------



# Importing libraries
import cv2
import numpy as np
# Capturing the video file 0 for videocam else you can provide the url
capture = cv2.VideoCapture("./sample/videos/vtest.avi")
 
# Reading the first frame
_, frame1 = capture.read()
# Convert to gray scale
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Create mask
hsv_mask = np.zeros_like(frame1)
# Make image saturation to a maximum value
hsv_mask[..., 1] = 255
 
# Till you scan the video
while(1):
     
    # Capture another frame and convert to gray scale
    _, frame2 = capture.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
 
    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnite and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to rgb
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
 
    cv2.imshow('frame2', rgb_representation)
    kk = cv2.waitKey(20) & 0xff
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    # Press 's' to save the video
    elif kk == ord('s'):
        cv2.imwrite('Optical_image.png', frame2)
        cv2.imwrite('HSV_converted_image.png', rgb_representation)
    prvs = next
 
capture.release()
cv2.destroyAllWindows()

# Reference         https://www.appsloveworld.com/opencv/100/18/quiver-plot-with-optical-flow?expand_article=1
#                   https://qiita.com/yusuke_s_yusuke/items/03243490b1fd765fe61f


# -------------------------------------------

def gabor_filter (img):
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



# -------------------------------------------

class MyMatplotlib (object):
    @classmethod
    def _sample_axes_plot (self):
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
    
    @classmethod
    def _sample_zoomin_zoomout (self):
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
    
    @classmethod
    def _sample_axes_demo (self):
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
    
    @classmethod
    def _sample_shared_square_axes (self):
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
    
    @classmethod
    def _sample_box_aspect (self):
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
    
    @classmethod
    def _sample_align_labels (self):
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
    
    @classmethod
    def _sample_user_defined (self):
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
    
    
    @classmethod
    def _sample_plot (self):
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
    
    @classmethod
    def _sample_scatter (self):
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

    @classmethod
    def _sample_bar (self):
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

# -------------------------------------------
# scikit-learn sample

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



class MyScikitLearn (object):
    def __init__(self):
        pass

    @classmethod
    def _linear_regression_example (self):
        """linear regression example
        Description
        ----------
            The example below uses only the first feature of the diabetes dataset,
            in order to illustrate the data points within the two-dimensional plot.
            The straight line can be seen in the plot,
            showing how linear regression attempts to draw a straight line that will best minimize
            the residual sum of squares between the observed responses in the dataset,
            and the responses predicted by the linear approximation.
        ----------
        Parameters
        ----------
        gamma : float, default: 1
            Desc
        s : float, default: 0.5 (purple)
            Desc
        ----------
        Return
        
        ----------

        ----------
        Example
        ----------

        ----------
        Reference
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
        ----------
        """
        # Code source: Jaques Grobler
        # License: BSD 3 clause

        import matplotlib.pyplot as plt
        import numpy as np

        from sklearn import datasets, linear_model
        from sklearn.metrics import mean_squared_error, r2_score

        # Load the diabetes dataset
        diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

        # Use only one feature
        diabetes_X = diabetes_X[:, np.newaxis, 2]

        # Split the data into training/testing sets
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]

        # Split the targets into training/testing sets
        diabetes_y_train = diabetes_y[:-20]
        diabetes_y_test = diabetes_y[-20:]

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(diabetes_X_train, diabetes_y_train)

        # Make predictions using the testing set
        diabetes_y_pred = regr.predict(diabetes_X_test)

        # The coefficients
        print("Coefficients: \n", regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
        # The coefficient of determination: 1 is perfect prediction
        print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

        # Plot outputs
        plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
        plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

        plt.xticks(())
        plt.yticks(())

        plt.show()

    def ridge_regression (self):
        from sklearn import linear_model
        reg = linear_model.Ridge(alpha=.5)
        reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
        print(reg.coef_)
        print(reg.intercept_)




# -------------------------------------------
# how to use newaxis
# Reference: https://note.nkmk.me/en/python-numpy-newaxis/
a = np.arange(6).reshape(2, 3)
print(a)
# [[0 1 2]
#  [3 4 5]]

print(a.shape)
# (2, 3)

print(a[:, :, np.newaxis])
# [[[0]
#   [1]
#   [2]]
# 
#  [[3]
#   [4]
#   [5]]]

print(a[:, :, np.newaxis].shape)
# (2, 3, 1)

# -------------------------------------------


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


# -------------------------------------------

# Reference     https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=custom%20dataset
# FasionMNIST


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# -------------------------------------------
# Reference     https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import torch
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))


# -------------------------------------------
# reference: https://pytorch.org/hub/pytorch_vision_resnet/
# sample for resnet18 model zoo

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()


# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)



# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# Download ImageNet labels
!wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# -------------------------------------------
# loss function
# https://blog.paperspace.com/pytorch-loss-functions/

# -------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
x1 = list(range(10))
y1 = [10, 2, 3, 1, 20, 10, 22, 4, 4, 1]
x2 = list(range(0, 10, 2))
y2 = [1, 20, 22, 4, 4]
ax.plot(x1, y1)
ax.plot(x2, y2)
fig.show()


# -------------------------------------------
# Ray tune
# Reference     https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html

from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

def load_data(data_dir="./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    return trainset, testset


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------------------------

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

def train_func(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_func(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_mnist(config):
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST(".", train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(".", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)

        # Send the current training result back to Tune
        train.report({"mean_accuracy": acc})

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")

search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
}

# Uncomment this to enable distributed execution
# `ray.init(address="auto")`

# Download the dataset first
datasets.MNIST("~/data", train=True, download=True)

tuner = tune.Tuner(
    train_mnist,
    param_space=search_space,
)
results = tuner.fit()

dfs = {result.path: result.metrics_dataframe for result in results}
[d.mean_accuracy.plot() for d in dfs.values()]


# -------------------------------------------
# camera on on mac
# https://qiita.com/seigot/items/9ae0dc3fed6d43bec828
import numpy as np
import cv2

def live_camera():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------
# Reference     https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
#               https://towardsdatascience.com/wavelet-fourier-analysis-on-the-enso-and-monsoon-data-in-python-31504eeadc2c
import numpy as np
import matplotlib.pyplot as plt
t_n = 1
N = 100000
T = t_n / N
f_s = 1/T

xa = np.linspace(0, t_n, num=N)
xb = np.linspace(0, t_n/4, num=N/4)

frequencies = [4, 30, 60, 90]
y1a, y1b = np.sin(2*np.pi*frequencies[0]*xa), np.sin(2*np.pi*frequencies[0]*xb)
y2a, y2b = np.sin(2*np.pi*frequencies[1]*xa), np.sin(2*np.pi*frequencies[1]*xb)
y3a, y3b = np.sin(2*np.pi*frequencies[2]*xa), np.sin(2*np.pi*frequencies[2]*xb)
y4a, y4b = np.sin(2*np.pi*frequencies[3]*xa), np.sin(2*np.pi*frequencies[3]*xb)

composite_signal1 = y1a + y2a + y3a + y4a
composite_signal2 = np.concatenate([y1b, y2b, y3b, y4b])

f_values1, fft_values1 = get_fft_values(composite_signal1, T, N, f_s)
f_values2, fft_values2 = get_fft_values(composite_signal2, T, N, f_s)

fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
axarr[0,0].plot(xa, composite_signal1)
axarr[1,0].plot(xa, composite_signal2)
axarr[0,1].plot(f_values1, fft_values1)
axarr[1,1].plot(f_values2, fft_values2)
(...)
plt.tight_layout()
plt.show()


# -------------------------------------------
# Reference     https://towardsdatascience.com/wavelet-fourier-analysis-on-the-enso-and-monsoon-data-in-python-31504eeadc2c

import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn')

dataset = "monsoon.txt"
df_nino = pd.read_table(dataset, skiprows=19, header=None)

N = df_nino.shape[0]
t0 = 1871
dt = 0.25
time = np.arange(0, N) * dt + t0

signal = df_nino.values.squeeze() #to get the scalar values
signal = signal - np.mean(signal)

scales = np.arange(1, 128) #set the wavelet scales

def plot_signal(time, signal, average_over=5, figname=None):
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(time, signal, label='signal')
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Signal Amplitude', fontsize=18)
    # ax.set_title('Signal + Time Average', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    ax.legend()
    if not figname:
        plt.savefig('signal_plot.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close('all')

plot_signal(time, signal) #plot and label the axis

# -------------------------------------------
# Reference         https://python.atelierkobato.com/axes3d/
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("Helix", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14)
ax.set_ylabel("y", size = 14)
ax.set_zlabel("z", size = 14)

# 軸目盛を設定
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

# 円周率の定義
pi = np.pi

# パラメータ分割数
n = 256

# パラメータtを作成
t = np.linspace(-6*pi, 6*pi, n)

# らせんの方程式
x = np.cos(t)
y = np.sin(t)
z = t

# 曲線を描画
ax.plot(x, y, z, color = "red")

plt.show()

# -------------------------------------------
# Reference         https://python.atelierkobato.com/axes3d/
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

# -5～5の乱数配列(100要素)
x = 10 * np.random.rand(100, 1) - 5
y = 10 * np.random.rand(100, 1) - 5
z = 10 * np.random.rand(100, 1) - 5

# 曲線を描画
ax.scatter(x, y, z, s = 40, c = "blue")

plt.show()


# -------------------------------------------
# https://python.atelierkobato.com/axes3d/
# PYTHON_MATPLOTLIB_3D_PLOT_03

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Figureと3DAxeS
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection="3d")

# 軸ラベルを設定
ax.set_xlabel("x", size = 16)
ax.set_ylabel("y", size = 16)
ax.set_zlabel("z", size = 16)

# 円周率の定義
pi = np.pi

# (x,y)データを作成
x = np.linspace(-3*pi, 3*pi, 256)
y = np.linspace(-3*pi, 3*pi, 256)

# 格子点を作成
X, Y = np.meshgrid(x, y)

# 高度の計算式
Z = np.cos(X/pi) * np.sin(Y/pi)

# 曲面を描画
ax.plot_surface(X, Y, Z, cmap = "summer")

# 底面に等高線を描画
ax.contour(X, Y, Z, colors = "black", offset = -1)

plt.show()

# -------------------------------------------
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
