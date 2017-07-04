# -*- coding: utf-8 -*-
import cv2
import datetime

def capturevideo():
    #获得视频的格式
    videoCapture = cv2.VideoCapture('02.mpg')

    #读帧
    success, frame = videoCapture.read()
    #cv2.imwrite("image.png", frame)

    i = 1
    time = 10
    image_names = []
    #print datetime.datetime.now()
    while success:
        success, frame = videoCapture.read(i)  # 获取下一帧
        cv2.imshow('video', frame) #显示
        if (i%time == 0):
            image_name = "videoImage/image_" + str(i) +".png"
            cv2.imwrite(image_name, frame)
            #print datetime.datetime.now()
            image_names.append(image_name)
        i += 1

    #return image_names

if __name__ == "__main__":
    capturevideo()