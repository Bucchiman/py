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

# Reference         https://www.appsloveworld.com/opencv/100/18/quiver-plot-with-optical-flow?expand_article=1
#                   https://qiita.com/yusuke_s_yusuke/items/03243490b1fd765fe61f
