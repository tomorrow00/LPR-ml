#coding=utf8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import RecognizeChars

# 显示图片
def imgshow(image, title):
    # (r, g, b) = cv2.split(image)
    # image = cv2.merge([b, g, r])

    # plt.title(title)
    # plt.imshow(image, "gray")
    # plt.show()
    # plt.pause(1)
    # plt.close()

    cv2.imshow(title, image)
    cv2.waitKey()

# 字符切割
def cutchars(image):
    positions = []
    rows, cols = image.shape[:2]

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # imgshow(image_hsv, "HSV")
    color = verifycolor(image_hsv)                                  #颜色判断
    # print color

    image_processed = imageprocess(color, image)                    #灰度化二值化去除铆钉
    # imgshow(image_processed, "ProcessedImage")

    image_contours, contours, hierarchy = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)           #找出边框
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # imgshow(image, "Contours")

    image_processed = imageprocess(color, image)

    tmp_chars = []
    boxs = []
    rects = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)                             #获取最小外接矩形
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(image, [box], 0, (0, 0, 255), 1)

        Xs = [i[0] for i in box]
        Ys = [j[1] for j in box]
        x1 = 0 if min(Xs) <= 0 else min(Xs)
        y1 = 0 if min(Ys) <= 0 else min(Ys)
        x2 = cols if max(Xs) >= cols else max(Xs)
        y2 = rows if max(Ys) >= rows else max(Ys)
        # angle = rect[2]

        tmp_char = image_processed[y1:y2, x1:x2]                    #切割图片

        if verifysize(tmp_char):
            tmp_char = changeshape(tmp_char)
            tmp_chars.append(tmp_char)
            positions.append((x1, x2))
            boxs.append(box)
            rects.append(rect)

            # cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
            # imgshow(image, "SubRectangle")

        else:
            continue

    if tmp_chars:
        tmp_chars, boxs, rects = sequencing(tmp_chars, positions, boxs, rects)
        tmp_chars, boxs, rects = rebuildrect(tmp_chars,  boxs, rects, cols)
        tmp_chars, boxs = getChinese(tmp_chars, boxs, rects, image_processed)
    else:
        return False

    for i in range(len(tmp_chars)):
        cv2.drawContours(image, [boxs[i]], 0, (0, 0, 255), 1)
         # imgshow(image, "SubRectangle")

    # imgshow(image, "VerifiedRectangle")
    return tmp_chars, color, image

# 灰度化二值化去除铆钉
def imageprocess(color, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            #灰度化
    # imgshow(image_gray, "Gray")

    if color == "BLUE" or color == "BLACK":                         #二值化
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # imgshow(image_binary, "Binary")
    elif color == "YELLOW" or color == "WHITE":
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # imgshow(image_binary, "Binary")
    else:
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # image_binary = removeborder(image_binary)                     #去除边框
    # imgshow(image_binary, "RemoveBorder")

    image_binary = removerivet(image_binary)                        #去除铆钉
    # imgshow(image_binary, "RemoveRivet")

    return image_binary

# 排序
def sequencing(tmp_chars, positions, boxs, rects):
    for i in range(len(positions)):
        for j in range(i, len(positions)):
            if positions[i][0] >= positions[j][0]:
                positions[i], positions[j] = positions[j], positions[i]
                tmp_chars[i], tmp_chars[j] = tmp_chars[j], tmp_chars[i]
                boxs[i], boxs[j] = boxs[j], boxs[i]
                rects[i], rects[j] = rects[j], rects[i]

    return tmp_chars, boxs, rects

# 去除错误中文字符和空隙
def rebuildrect(tmp_chars, boxs, rects, cols):
    new_tmp_chars = []
    new_boxs = []
    new_rects = []
    flag = 0
    mark = 0

    for i in range(len(tmp_chars)):
        x = rects[i][0][0]
        left = cols / 7
        right = cols * 2 / 7

        if x >= left and x <= right:                            # 范围之内即为特殊字符
            flag = 1

        if flag == 1 and mark < 6:                              # 从特殊字符开始存储，特殊字符左边全排除，右边取6个字符，多余即为空隙
            new_tmp_chars.append(tmp_chars[i])
            new_boxs.append(boxs[i])
            new_rects.append(rects[i])
            mark += 1

    return new_tmp_chars, new_boxs, new_rects

# 获取中文
def getChinese(tmp_chars, boxs, rects, image):
    schar_rect = rects[0]

    offsets = []                                                # 偏移量
    for i in range(len(rects)):
        if i == len(rects) - 1:
            break
        temp_offset = rects[i + 1][0][0] - rects[i][0][0]
        offsets.append(temp_offset)

    offset = min(offsets)

    change = schar_rect[0][0] - offset
    Chineschar_rect = ((change, schar_rect[0][1]), (schar_rect[1][0], schar_rect[1][1]), schar_rect[2])
    Chineschar_box = cv2.boxPoints(Chineschar_rect)
    Chineschar_box = np.int0(Chineschar_box)

    Xs = [i[0] for i in Chineschar_box]
    Ys = [j[1] for j in Chineschar_box]
    x1 = min(Xs)
    y1 = min(Ys)
    x2 = max(Xs)
    y2 = max(Ys)

    Chineschar = image[y1:y2, x1:x2]                                # 切割图片
    Chineschar = changeshape(Chineschar)

    tmp_chars.insert(0, Chineschar)
    boxs.insert(0, Chineschar_box)

    return tmp_chars, boxs

# 获取颜色
def verifycolor(image):
    lower_blue = np.array([90, 80, 80])             #蓝色区域
    upper_blue = np.array([120, 220, 255])

    lower_yellow = np.array([11, 43, 46])           #黄色区域
    upper_yellow = np.array([34, 255, 255])

    lower_black = np.array([0, 0, 0])               #黑色区域
    upper_black = np.array([180, 255, 46])

    lower_white = np.array([0, 0, 46])              #白色区域
    upper_white = np.array([180, 30, 220])

    blue = judgecolor(image, lower_blue, upper_blue)
    yellow = judgecolor(image, lower_yellow, upper_yellow)
    black = judgecolor(image, lower_black, upper_black)
    white = judgecolor(image, lower_white, upper_white)
    # print blue, yellow, black, white

    if blue == 1:
        return "BLUE"
    if yellow == 1:
        return "YELLOW"
    if black == 1:
        return "BLACK"
    if white == 1:
        return "WHITE"

# 颜色判断
def judgecolor(image, lower, upper):
    thresh = 0.2
    image_mask = cv2.inRange(image, lower, upper)

    rows, cols = image_mask.shape
    percent = float(cv2.countNonZero(image_mask)) / float(rows * cols)
    # imgshow(image_mask, "Blue")

    if percent > thresh:
        return True
    else:
        return False

# 去除边框

# 去除铆钉
def removerivet(image):
    rows, cols = image.shape
    jumpStd = 7
    jumpCounts = []

    for i in range(rows):
        jumpCount = 0

        for j in range(cols - 1):
            if image[i][j] != image[i][j+1]:
                jumpCount += 1

        jumpCounts.append(jumpCount)

    for i in range(rows):
        if jumpCounts[i] <= jumpStd:
            for j in range(cols):
                image[i][j] = 0

    return image

# 判断外接矩形大小
def verifysize(image):
    rows, cols = image.shape
    if cols == 0 or rows == 0:
        return False

    aspect_std = 0.5				    #标准宽高比，45/90
    minHeight = float(10)               #最小高度
    maxHeight = float(35)               #最大高度

    error = 0.7			                #偏差，根据easyPR，error=0.7
    aspect_min = 0.05		            #最小宽高比
    aspect_max = aspect_std + aspect_std * error		        #最大宽高比

    pixels = float(cv2.countNonZero(image))                     #image中像素点
    area = float(rows * cols)				                    #矩形框面积
    percPixels = pixels / area

    aspect = float(cols) / float(rows)  			            #外接矩形宽高比

    if percPixels < 1 and (aspect <= aspect_max and aspect >= aspect_min) and (rows <= maxHeight and rows >= minHeight):
        return True
    else:
        return False

# 修改形状
def changeshape(image):
    rows, cols = image.shape
    cols_side = (rows - cols) / 2

    temp_image = np.zeros((rows, cols_side))
    new_image = np.column_stack((image, temp_image))            # 右侧扩充
    new_image = np.column_stack((temp_image, new_image))        # 左侧扩充
    image = cv2.resize(new_image, (20, 20))

    image = np.array(image, dtype=np.uint8)

    return image

# 保存字符图片
def savechars(tmp_chars, filename):
    if os.path.exists("temp_chars/" + filename[:-4]):
        shutil.rmtree("temp_chars/" + filename[:-4])
    os.mkdir("temp_chars/" + filename[:-4])

    i = 0
    for tmp_char in tmp_chars:
        # imgshow(tmp_char, "Char")
        cv2.imwrite("temp_chars/" + filename[:-4] + "/char_" + str(i) + ".png", tmp_char)
        i += 1

# 字符识别
def segmentchars():
    if os.path.exists("temp_chars"):
        shutil.rmtree("temp_chars")
    os.mkdir("temp_chars")

    # filename = "plate_苏A0CP56.jpg"
    filename = "A03_BT6052_0.jpg"
    plate = cv2.imread("test_plates/" + filename)
    print filename

    tmp_chars, color, image = cutchars(plate)
    savechars(tmp_chars, filename)

    results = RecognizeChars.recognizechars(tmp_chars)

    if color == "BLUE":
        result_color = "蓝色"
    if color == "white":
        result_color = "白色"
    if color == "YELLOW":
        result_color = "黄色"
    if color == "BLACK":
        result_color = "黑色"

    print "颜色：" + result_color
    print  "车牌号:",
    plate_con = ""
    for result in results:
        plate_con += result
    print plate_con, "\n"

    # image = cv2.imread(filename)
    imgshow(image, "Chars")

if __name__ == "__main__":
    segmentchars()