#coding=utf8
import JudgePlate
import GetPlate
import SegmentChars
import RecognizeChars
import cv2
import os
import shutil
from matplotlib import pyplot as plt
import time

# 显示图片
def imgshow(image, title):
    # (r, g, b) = cv2.split(image)
    # image = cv2.merge([b, g, r])
    #
    # plt.title(title)
    # plt.axis("off")
    # plt.imshow(image)
    plt.show()
    # plt.pause(1)
    # plt.close()

    cv2.imshow(title, image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def platerecognize(image, filename):
    tmp_boxs, tmp_plates = GetPlate.getplate(image, filename)
    boxs, plates = JudgePlate.judgeplate(tmp_boxs, tmp_plates)
    results_color = []
    results_content = []
    results_plate = []
    i = 0

    for plate in plates:
        result_color, result_content, result_plate = SegmentChars.segmentchars(plate, filename)
        results_color.append(result_color)
        results_content.append(result_content)
        results_plate.append(result_plate)

        cv2.drawContours(image, [boxs[i]], 0, (0, 0, 255), 2)
        i += 1

    return results_color, results_content, results_plate, image

def main():
    if os.path.exists("result/"):
        shutil.rmtree("result/")
    os.mkdir("result/")

    # i = 0
    # time = 5
    # videoCapture = cv2.VideoCapture('IMG_8598.MOV')                         # 获得视频的格式
    # success, frame = videoCapture.read()
    #
    # while success:
    #     success, image = videoCapture.read(i)                                                               # 获取下一帧
    #     if (i%time == 0):
    #         platerecognize(image, i)
    #         # print i
    #
    #     i += 1

    filename = "苏A0CP56.jpg"
    image = cv2.imread(filename)
    filename = filename[:-4]

    if os.path.exists("result/" + filename):
        shutil.rmtree("result/" + filename)
    os.mkdir("result/" + filename)

    results_color, results_content, results_plate, image = platerecognize(image, filename)

    for i in range(len(results_plate)):
        print "颜色：", results_color[i]
        print "车牌号：", results_content[i]
        # cv2.putText(image, "Color：" + results_color[i] +  "\nNumber:" + results_color[i], (100, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255))
        imgshow(image, "ImagePlates")
        imgshow(results_plate[i], "PlateChars")

        cv2.imwrite("result/" + filename + "/result_image_" + str(i) + ".png", image)
        cv2.imwrite("result/" + filename + "/result_plate_" + str(i) + ".png", results_plate[i])

if __name__ == "__main__":
    main()