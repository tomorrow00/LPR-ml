#coding=utf8
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import datetime
import numpy as np

# 显示图片
def imgshow(image, title):
    '''(r, g, b) = cv2.split(image)
    image = cv2.merge([b, g, r])

    plt.title(title)
    plt.imshow(image)
    plt.show()'''

    cv2.imshow(title, image)
    cv2.waitKey(0)

# 提取特征
def getfeature(image):
    ret, image_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		                #二值化
    rows, cols = image_binary.shape
    sum_rows = []
    sum_cols = []

    for i in range(rows):                                                   # 每行非零元素个数
        sum_rows.append(sum(image_binary[i]) / 255)

    for i in range(cols):                                                   # 每列非零元素个数
        sum_cols.append(sum(image_binary[x][i] for x in range(rows)) / 255)
    # print sum_rows, "sdhfsfjdjfd", sum_cols, len(sum_rows), len(sum_cols)

    # max_bins = int(image_binary.max() + 1)
    # hist, edges = np.histogram(image_binary, bins=max_bins, normed=True)
    # print hist

    # vmax_bins = int(max(sum_rows) + 1)
    vhist, vedge = np.histogram(sum_rows, bins=len(sum_rows), normed=True)              # 水平直方图
    # hmax_bins = int(max(sum_cols) + 1)
    hhist, hedge = np.histogram(sum_cols, bins=len(sum_cols), normed=True)              # 垂直直方图
    # print vhist, "asdasdasdasd", hhist, len(vhist), len(hhist)

    char_feature = []
    for i in range(len(vhist)):
        char_feature.append(vhist[i])
    for i in range(len(hhist)):
        char_feature.append(hhist[i])

    return char_feature

# 测试
def test(char_feature):
    clf = joblib.load("model_chars.m")
    result = clf.predict(char_feature)

    return result[0]

def main():
    file_all = "ann/"
    files_sub = []
    # filenames = []
    labels = []
    images = []
    filename = "5.jpg"

    image = cv2.imread(filename, 0)
    char_feature = getfeature(image)
    char_feature = np.reshape(char_feature, (1, -1))                                # 解除warning

    # files_sub = os.listdir(file_all)
    # for file_sub in files_sub:
    #     name_temp = os.path.join(file_all, file_sub)
    #     files = os.listdir(name_temp)
    #
    #     for file in files:
    #
    #         filename = os.path.join(name_temp, file)
    #         # filenames.append(filename)
    #
    #         image = imgprocess(filename)
    #         images.append(image)

    # print "Start Training...", "Time:" + str(datetime.datetime.now())
    result = test(char_feature)
    print result
    # print "Training Completed.", "Time:" + str(datetime.datetime.now())

if __name__ == "__main__":
    main()