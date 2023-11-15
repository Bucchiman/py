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

# https://datacarpentry.org/image-processing/aio.html
# ------------------------------------------- 
