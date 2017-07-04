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
    #
    # plt.title(title)
    # plt.imshow(image, "gray")
    # plt.show()
    # plt.pause(1)
    # plt.close()

    cv2.imshow(title, image)
    cv2.waitKey()

# 字符切割
def cutchars(image):
    # imgshow(image, "Original")
    positions = []
    rows, cols = image.shape[:2]
    # print rows, cols

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # imgshow(image_hsv, "HSV")
    color, lower, upper = verifycolor(image_hsv)                                                      # 颜色判断
    # print color

    image_processed = imageprocess(color, image)                                        # 灰度化二值化
    # imgshow(image_processed, "ProcessedImage")
    image_ROI, image, rows, cols = findROI(image_processed, image, rows, cols)          # 找出车牌区域
    # imgshow(image_ROI, "ROI")

    image_rotated, image = rotation(image_ROI, image, rows, cols, color, lower, upper)                       # 偏斜扭转
    # imgshow(image_rotated, "Rotation")

    image_copy = np.zeros(image_rotated.shape, np.uint8)                                # 复制图片
    image_copy = image_rotated.copy()

    image_contours, contours, hierarchy = cv2.findContours(image_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)           #找出边框
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    # imgshow(image, "Contours")

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

        tmp_char = image_copy[y1:y2, x1:x2]                    #切割图片

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

    # try:
    tmp_chars, boxs, rects = sequencing(tmp_chars, positions, boxs, rects)
    tmp_chars, boxs, rects = rebuildrect(tmp_chars,  boxs, rects, cols)
    tmp_chars, boxs = getChinese(tmp_chars, boxs, rects, image_copy)
    # except Exception, e:
    #     print Exception, ":", e

    for i in range(len(tmp_chars)):
        cv2.drawContours(image, [boxs[i]], 0, (0, 0, 255), 1)
        # imgshow(image, "SubRectangle")

    imgshow(image, "VerifiedRectangle")
    return tmp_chars, color, image

# 灰度化二值化
def imageprocess(color, image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                #灰度化
    # imgshow(image_gray, "Gray")

    if color == "BLUE" or color == "BLACK":                             #二值化
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # imgshow(image_binary, "Binary")
    elif color == "YELLOW" or color == "WHITE":
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # imgshow(image_binary, "Binary")
    else:
        ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # image_binary = removerivet(image_binary)                          #去除铆钉
    # imgshow(image_binary, "RemoveRivet")

    return image_binary

# 获取目标区域
def findROI(image, image_original, rows, cols):
    T = 2
    flag = 0

    for j in range(0, cols):
        count = 0
        for i in range(rows):
            if image[i][j] == 255:
                count += 1
            if count > T:
                ROI_left = j
                flag = 1
                break
        if flag == 1:
            break

    flag = 0
    for j in range(cols - 1, cols / 2, -1):
        count = 0
        for i in range(rows / 2):
            if image[i][j] == 255:
                count += 1
            if count > T:
                ROI_right = j
                flag = 1
                break
        if flag == 1:
            break

    image_new = np.zeros((rows, ROI_right - ROI_left), np.uint8)
    image_original_new = np.zeros((rows, ROI_right - ROI_left, 3), np.uint8)
    for i in range(rows):
        m = 0
        for j in range(ROI_left, ROI_right):
            image_new[i][m] = image[i][j]
            image_original_new[i][m] = image_original[i][j]
            m += 1

    rows, cols = image_new.shape

    return image_new, image_original_new, rows, cols

# 偏斜扭转
def rotation(image, image_original, rows, cols, color, lower, upper):
# ================================================================================================================================================
    image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    image_mask = cv2.inRange(image_hsv, lower, upper)
    imgshow(image, "C")

    edges = cv2.Canny(image_mask, 0, 1000)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)           #???????????????????????????????最后一个参数问题?????????????????????????????

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image_original, (x1, y1), (x2, y2), (0, 0, 255), 2)

    imgshow(image_original, "Line")
# ================================================================================================================================================

    theta = lines[0][0][1] / np.pi * 180
    judge_angle = 90 - theta
    # print judge_angle

    if judge_angle >= -5 and judge_angle <= 5:
        return image, image_original

    else:
        angle = theta - 90

        R = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        image_rotated = cv2.warpAffine(image, R, (cols, rows))
        image_original = cv2.warpAffine(image_original, R, (cols, rows))

        ret, image_rotated = cv2.threshold(image_rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # imgshow(image_rotated, "Rotation")

        image_removerivet = removerivet(image_rotated)                                                                  # 去除铆钉
        # imgshow(image_removerivet, "RemoveRivet")

        image_parallel, image_original, rows, cols = getRect(image_removerivet, image_original, rows, cols)             # 获取平行四边形
        # imgshow(image_parallel, "Parallelogram")

        x = rows * np.tan(np.pi / 2 - lines[0][0][1])

        lu = [x, 0]
        ru = [cols, 0]
        ld = [0, rows]
        pts1 = np.float32([lu, ru, ld])

        lu_new = [0, 0]
        ru_new = [cols - x, 0]
        ld_new = [0, rows]
        pts2 = np.float32([lu_new, ru_new, ld_new])

        A = cv2.getAffineTransform(pts1, pts2)
        image_warp = cv2.warpAffine(image_parallel, A, (cols - int(x), rows))
        image_original = cv2.warpAffine(image_original, A, (cols - int(x), rows))
        # imgshow(image_warp, "WarpAffine")

        return image_warp, image_original

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

    # lower_black = np.array([0, 0, 0])               #黑色区域
    # upper_black = np.array([180, 255, 46])
    #
    # lower_white = np.array([0, 0, 46])              #白色区域
    # upper_white = np.array([180, 30, 220])

    blue = judgecolor(image, lower_blue, upper_blue)
    yellow = judgecolor(image, lower_yellow, upper_yellow)
    # black = judgecolor(image, lower_black, upper_black)
    # white = judgecolor(image, lower_white, upper_white)
    # print blue, yellow, black, white

    if blue == 1:
        return "BLUE", lower_blue, upper_blue
    if yellow == 1:
        return "YELLOW", lower_yellow, upper_yellow
    # if black == 1:
    #     return "BLACK"
    # if white == 1:
    #     return "WHITE"

# 颜色判断
def judgecolor(image, lower, upper):
    thresh = 0.2
    image_mask = cv2.inRange(image, lower, upper)
    # imgshow(image_mask, "Color")

    rows, cols = image_mask.shape
    percent = float(cv2.countNonZero(image_mask)) / float(rows * cols)

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

# 旋转后获取车牌
def getRect(image, image_original, rows, cols):
    sum_rows = []
    for i in range(rows):
        for j in range(cols):
            if image[i][j] == 255:
                sum_rows.append(i)
    upper = min(sum_rows)
    downer = max(sum_rows)

    image_new = np.zeros((downer - upper, cols), dtype=np.uint8)
    image_original_new = np.zeros((downer - upper, cols, 3), dtype=np.uint8)
    temp = 0
    for i in range(upper, downer):
        for j in range(cols):
            image_new[temp][j] = image[i][j]
            image_original_new[temp][j] = image_original[i][j]
        temp += 1

    rows, cols = image_new.shape

    return image_new, image_original_new, rows, cols

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
    if os.path.exists("temp_chars/" + filename):
        shutil.rmtree("temp_chars/" + filename)
    os.mkdir("temp_chars/" + filename)

    i = 0
    for tmp_char in tmp_chars:
        # imgshow(tmp_char, "Char")
        cv2.imwrite("temp_chars/" + filename + "/char_" + str(i) + ".png", tmp_char)
        i += 1

# 字符识别
# def segmentchars(plates, filename):
def segmentchars(plate):
    results = []
    tmp_chars, color, image = cutchars(plate)
    # savechars(tmp_chars, "123")
    result = RecognizeChars.recognizechars(tmp_chars)

    if color == "BLUE":
        result_color = "蓝色"
    if color == "YELLOW":
        result_color = "黄色"
    # if color == "WHITE":
    #     result_color = "白色"
    # if color == "BLACK":
    #     result_color = "黑色"

    # print "颜色：", result_color
    # print "车牌号：", result
    # print ""

    # imgshow(image, "Plate")

    return result

if __name__ == "__main__":
    # f = "/home/computer/PlateRecognize/PR/test_plates"
    # files = os.listdir(f)
    #
    # for file in files:
    #     filename = os.path.join(f, file)
    #     plate = cv2.imread(filename)
    #
    #     try:
    #         result = segmentchars(plate)
    #         print result
    #     except:
    #         continue

    # filename = "test_plates/tagged_苏A0CP56.png"
    # filename = "test_plates/287_0.png"
    filename = "test_plates/44_0.png"
    plate = cv2.imread(filename)

    result = segmentchars(plate)
    print result