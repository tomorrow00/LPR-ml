#coding=utf8
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import datetime
import numpy as np
from matplotlib import pyplot as plt

# 显示图片
def imgshow(image, title):
    # (r, g, b) = cv2.split(image)
    # image = cv2.merge([b, g, r])

    plt.title(title)
    plt.imshow(image, "gray")
    plt.show()
    # plt.pause(1)
    # plt.close()

    # cv2.imshow(title, image)
    # cv2.waitKey()

# 提取特征
def getfeature(image):
    ret, image_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		                #二值化
    rows, cols = image_binary.shape
    sum_rows = []
    sum_cols = []

    char_point = np.reshape(image_binary, (1, -1))
    char_point = char_point[0]

    for i in range(rows):                                                   # 每行非零元素个数
        sum_rows.append(sum(image_binary[i]) / 255)

    for i in range(cols):                                                   # 每列非零元素个数
        sum_cols.append(sum(image_binary[x][i] for x in range(rows)) / 255)

    vhist, vedge = np.histogram(sum_rows, bins=len(sum_rows), normed=True)              # 水平直方图
    hhist, hedge = np.histogram(sum_cols, bins=len(sum_cols), normed=True)              # 垂直直方图

    char_feature = []
    for i in range(len(vhist)):
        char_feature.append(vhist[i])
    for i in range(len(hhist)):
        char_feature.append(hhist[i])
    for i in range(len(char_point)):
        char_feature.append(char_point[i])

    return char_feature

# 测试
def test(char_feature, name):
    clf = joblib.load("model_chars_" + name + ".m")
    result = clf.predict(char_feature)

    return result[0]

# 识别字符
def recognizechars(tmp_chars):
    results = []
    flag = 0

    for tmp_char in tmp_chars:
        char_feature = getfeature(tmp_char)
        char_feature = np.reshape(char_feature, (1, -1))                                # 解除warning

        if flag == 0:
            result = test(char_feature, "Chinese")
            flag = 1
        else:
            result = test(char_feature, "NumLetter")

        results.append(result)

        # print  "车牌号:",
        plate_con = ""
        for result in results:
            plate_con += result
        # print plate_con, "\n"

    return plate_con