# coding=utf8
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.externals import joblib


# 显示图片
def imgshow(image, title):
    (r, g, b) = cv2.split(image)
    image = cv2.merge([b, g, r])

    plt.title(title)
    plt.imshow(image)
    #plt.show()
    plt.pause(2)
    plt.close()

    #cv2.imshow(title, image)
    #cv2.waitKey(0)


# 图片处理
def imgprocess(image):
    # image_blur = cv2.GaussianBlur(image, (5, 5), 0)		#高斯模糊
    # imgshow(image_blur, "Blur")

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
    # imgshow(image_gray, "Gray")

    # ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		#二值化

    return image_gray


# 提取LBP特征
def LBPfeature(image):
    radius = 1
    n_points = 9 * radius

    '''image_lbp = local_binary_pattern(image, n_points, radius)
    #imgshow(image_lbp, "LBP")
    hist, edges = np.histogram(image_lbp, bins=32)'''

    image_lbp = local_binary_pattern(image, n_points, radius)
    # imgshow(image_lbp, "LBP")
    # print len(image_lbp)
    max_bins = int(image_lbp.max() + 1)
    hist, edges = np.histogram(image_lbp, bins=max_bins, normed=True)

    return hist


# 训练
def test(hist):
    clf = joblib.load("model_plates.m")
    result = clf.predict(hist)

    return result[0]

#判断车牌
def judgeplate(tmp_boxs, tmp_plates):
    # files = os.listdir("temp_plates")
    sum = len(tmp_plates)
    boxs = []
    plates = []
    # filenames = []

    if sum:
        # for file in files:
        #     filenames.append(os.path.join("temp_plates", file))

        i = 0
        for tmp_plate in tmp_plates:
            # image_ori = cv2.imread(filename)
            image = imgprocess(tmp_plate)
            hist = LBPfeature(image)
            hist = np.array(hist, dtype="float32")
            hist = hist.reshape(1, -1)

            result = test(hist)
            if result == 1:
                # count = int(filename[19:-4])
                # results.append(count)
                plates.append(tmp_plate)
                boxs.append(tmp_boxs[i])

            i += 1

    return boxs, plates