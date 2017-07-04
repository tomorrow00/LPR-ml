#coding=utf8
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import cross_validation
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
def getfeature(filename):
    image = cv2.imread(filename, 0)
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

# 训练
def train(chars_feature, labels, name):
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(chars_feature, labels, test_size=0.1, random_state=0)

    # clf = MLPClassifier(hidden_layer_sizes=(1, 3, cv2.CV_32SC1), activation="logistic", momentum=0.1)
    # clf = MLPClassifier()                                             # 建立MLP多层感知器
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, "model_chars_" + name + ".m")

    # clf = svm.SVC(gamma=0.1, kernel='rbf')
    # clf = svm.SVC()		                                                #建立SVM模型
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, "model_chars_" + name + ".m")

    clf = RandomForestClassifier()                                    # 建立随机森林模型
    # param_grid = []
    clf.fit(x_train, y_train)
    joblib.dump(clf, "model_chars_" + name + ".m")

    score = clf.score(x_test, y_test)
    print score

def main():
    file_all_Chinese = "ann/hanzi/"
    file_all_NumLetter = "ann/shuzimu/"
    # filenames = []

    labels_Chinese = []
    labels_NumLetter = []

    chars_feature_Chinese = []
    chars_feature_NumLetter = []

    files_sub_Chinese = os.listdir(file_all_Chinese)
    files_sub_NumLetter = os.listdir(file_all_NumLetter)

    for file_sub_Chinese in files_sub_Chinese:
        name_temp = os.path.join(file_all_Chinese, file_sub_Chinese)
        files = os.listdir(name_temp)

        for file in files:
            if file_sub_Chinese == "zh_cuan":
                labels_Chinese.append("川")
            elif file_sub_Chinese == "zh_e":
                labels_Chinese.append("鄂")
            elif file_sub_Chinese == "zh_gan":
                labels_Chinese.append("赣")
            elif file_sub_Chinese == "zh_gan1":
                labels_Chinese.append("甘")
            elif file_sub_Chinese == "zh_gui":
                labels_Chinese.append("贵")
            elif file_sub_Chinese == "zh_gui1":
                labels_Chinese.append("桂")
            elif file_sub_Chinese == "zh_hei":
                labels_Chinese.append("黑")
            elif file_sub_Chinese == "zh_hu":
                labels_Chinese.append("沪")
            elif file_sub_Chinese == "zh_ji":
                labels_Chinese.append("冀")
            elif file_sub_Chinese == "zh_jin":
                labels_Chinese.append("津")
            elif file_sub_Chinese == "zh_jing":
                labels_Chinese.append("京")
            elif file_sub_Chinese == "zh_jl":
                labels_Chinese.append("吉")
            elif file_sub_Chinese == "zh_liao":
                labels_Chinese.append("辽")
            elif file_sub_Chinese == "zh_lu":
                labels_Chinese.append("鲁")
            elif file_sub_Chinese == "zh_meng":
                labels_Chinese.append("蒙")
            elif file_sub_Chinese == "zh_min":
                labels_Chinese.append("闽")
            elif file_sub_Chinese == "zh_ning":
                labels_Chinese.append("宁")
            elif file_sub_Chinese == "zh_qing":
                labels_Chinese.append("青")
            elif file_sub_Chinese == "zh_qiong":
                labels_Chinese.append("琼")
            elif file_sub_Chinese == "zh_shan":
                labels_Chinese.append("陕")
            elif file_sub_Chinese == "zh_su":
                labels_Chinese.append("苏")
            elif file_sub_Chinese == "zh_sx":
                labels_Chinese.append("晋")
            elif file_sub_Chinese == "zh_wan":
                labels_Chinese.append("皖")
            elif file_sub_Chinese == "zh_xiang":
                labels_Chinese.append("湘")
            elif file_sub_Chinese == "zh_xin":
                labels_Chinese.append("新")
            elif file_sub_Chinese == "zh_yu":
                labels_Chinese.append("豫")
            elif file_sub_Chinese == "zh_yu1":
                labels_Chinese.append("渝")
            elif file_sub_Chinese == "zh_yue":
                labels_Chinese.append("粤")
            elif file_sub_Chinese == "zh_yun":
                labels_Chinese.append("云")
            elif file_sub_Chinese == "zh_zang":
                labels_Chinese.append("藏")
            elif file_sub_Chinese == "zh_zhe":
                labels_Chinese.append("浙")

            filename = os.path.join(name_temp, file)
            # filenames.append(filename)

            image = cv2.imread(filename)

            char_feature_Chinese = getfeature(filename)
            chars_feature_Chinese.append(char_feature_Chinese)

    # for file_sub_NumLetter in files_sub_NumLetter:
    #     name_temp = os.path.join(file_all_NumLetter, file_sub_NumLetter)
    #     files = os.listdir(name_temp)
    #
    #     for file in files:
    #         labels_NumLetter.append(file_sub_NumLetter)
    #
    #         filename = os.path.join(name_temp, file)
    #         # filenames.append(filename)
    #
    #         image = cv2.imread(filename)
    #
    #         char_feature_NumLetter = getfeature(filename)
    #         chars_feature_NumLetter.append(char_feature_NumLetter)

    print len(chars_feature_Chinese), len(labels_Chinese)
    print len(chars_feature_NumLetter), len(labels_NumLetter)
    print "\n"

    print "Start Chinese Training...", "Time:" + str(datetime.datetime.now())
    train(chars_feature_Chinese, labels_Chinese, "Chinese")
    print "Chinese Training Completed.", "Time:" + str(datetime.datetime.now())
    print "\n"

    print "Start Numbers&Letters Training...", "Time:" + str(datetime.datetime.now())
    # train(chars_feature_NumLetter, labels_NumLetter, "NumLetter")
    print "Numbers&Letters Training Completed.", "Time:" + str(datetime.datetime.now())

if __name__ == "__main__":
    main()