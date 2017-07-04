# -*- coding: utf-8 -*-
import cv2
import datetime
from matplotlib import pyplot as plt

# 显示图片
def imgshow(image, title):
    (r, g, b) = cv2.split(image)
    image = cv2.merge([b, g, r])

    plt.title(title)
    plt.imshow(image)
    # plt.show()
    plt.pause(1)
    plt.close()

    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def capturevideo():
    i = 0
    time = 1000
    videoCapture = cv2.VideoCapture('north-gate_20160807070000_20160807215959.mp4')                         # 获得视频的格式
    success, frame = videoCapture.read()

    while success:
        success, image = videoCapture.read(i)
        if (i%time == 0):                     # 获取下一帧
            imgshow(image, "adsasd")
            print i

        i += 1

    # return frame

if __name__ == "__main__":
    capturevideo()