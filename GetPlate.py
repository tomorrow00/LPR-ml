#coding=utf8
import cv2
import numpy as np
import os
import shutil
from matplotlib import pyplot as plt

#显示图片
def imgshow(image, title):
	# (r, g, b) = cv2.split(image)
	# image = cv2.merge([b, g, r])
    #
	# plt.title(title)
	# plt.imshow(image)
	# # plt.show()
	# plt.pause(2)
	# plt.close()
	
	cv2.imshow(title, image)
	cv2.waitKey(0)

# 获取临时车牌图片
def gettmpplates_sobel(image):
	# image = cv2.imread(filename)
	rows, cols = image.shape[:2]

	image_blur = cv2.GaussianBlur(image, (5, 5), 0)  												# 高斯模糊，5
	# imgshow(image_blur, "Blur")

	image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY) 	 									# 灰度化
	# imgshow(image_gray, "Gray")

	image_sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0)  										# Sobel算子，CV_16S
	image_sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1)

	image_sobelx = np.uint8(np.absolute(image_sobelx))
	image_sobely = np.uint8(np.absolute(image_sobely))
	image_sobel = cv2.bitwise_or(image_sobelx, image_sobely)

	# imgshow(image_sobel, "Sobel")

	ret, image_binary = cv2.threshold(image_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  	# 二值化
	# imgshow(image_binary, "Binary")

	kernel = np.ones((3, 17), np.uint8)  															# 闭操作，width=17，height=3
	image_closing = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
	# imgshow(image_closing, "Closing")

	image_contours, contours, hierarchy = cv2.findContours(image_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)		#绘制边框
	# cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
	# imgshow(image, "Contours")

	region_tmp_plates = []
	boxs = []
	for contour in contours:
		rect = cv2.minAreaRect(contour)																#获取最小外接矩形

		if verifysize(rect) and verifyangle(rect):
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			# cv2.drawContours(image, [box], 0, (0, 255, 255), 2)

			Xs = [i[0] for i in box]
			Ys = [j[1] for j in box]
			x1 = 0 if min(Xs) <= 0 else min(Xs)
			y1 = 0 if min(Ys) <= 0 else min(Ys)
			x2 = cols if max(Xs) >= cols else max(Xs)
			y2 = rows if max(Ys) >= rows else max(Ys)
			angle = rect[2]

			region_tmp_plates.append((x1, y1, x2, y2, angle))
			boxs.append(box)
		else:
			continue

	# imgshow(image, "VerifiedRectangle")
	return region_tmp_plates, boxs

# 颜色定位
def colorlocate(filename):
	image = cv2.imread(filename)
	image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# imgshow(image_hsv, "asdasd")

	lower_blue = np.array([90, 80, 80])  # 蓝色区域
	upper_blue = np.array([120, 220, 255])

	lower_yellow = np.array([11, 43, 46])  # 黄色区域
	upper_yellow = np.array([34, 255, 255])

	# lower_black = np.array([0, 0, 0])  # 黑色区域
	# upper_black = np.array([180, 255, 46])
    #
	# lower_white = np.array([0, 0, 46])  # 白色区域
	# upper_white = np.array([180, 30, 220])

	thresh = 0.2
	image_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
	# imgshow(image_mask, "Blue")
	gettmpplates_color(image_mask, image)

	image_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
	# imgshow(image_mask, "Yellow")

# 获取临时车牌图片
def gettmpplates_color(image_mask, image):
	rows, cols = image.shape[:2]

	kernel = np.ones((21, 17), np.uint8)								# 闭操作，width=17，height=3
	image_closing = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
	imgshow(image_closing, "Closing")

	image_contours, contours, hierarchy = cv2.findContours(image_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)			#绘制边框
	# cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
	# imgshow(image, "Contours")

	region_tmp_plates = []
	boxs = []
	for contour in contours:
		rect = cv2.minAreaRect(contour)								#获取最小外接矩形
		print rect

		# if verifysize(rect) and verifyangle(rect):
		if verifysize_color(rect):
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			# cv2.drawContours(image, [box], 0, (0, 255, 255), 2)

			Xs = [i[0] for i in box]
			Ys = [j[1] for j in box]
			x1 = 0 if min(Xs) <= 0 else min(Xs)
			y1 = 0 if min(Ys) <= 0 else min(Ys)
			x2 = cols if max(Xs) >= cols else max(Xs)
			y2 = rows if max(Ys) >= rows else max(Ys)
			angle = rect[2]

			region_tmp_plates.append((x1, y1, x2, y2, angle))
			boxs.append(box)
		else:
			continue

	# imgshow(image, "VerifiedRectangle")
	return region_tmp_plates, boxs

# 颜色尺寸判断
def verifysize_color(rect):
	# 中国车牌尺寸: 440mm*140mm，宽高比3.142857
	# 实际车牌尺寸: 136*32，宽高比4.25
	aspect_std = 3.75  # 标准宽高比，3.75
	# area_std = 17 * 4  # 标准面积，34*8

	error = 0.1  # 偏差，根据easyPR，error=0.9
	aspect_min = aspect_std - aspect_std * error  # 最小宽高比
	aspect_max = aspect_std + aspect_std * error  # 最大宽高比

	# verifyMin_area = 1  # 偏差，根据easyPR，min=1，max=24
	# verifyMax_area = 400
	# area_min = area_std * verifyMin_area  # 最小面积
	# area_max = area_std * verifyMax_area  # 最大面积

	rect_width = rect[1][0]  # 外接矩形宽
	rect_length = rect[1][1]  # 外接矩形长

	area = rect_width * rect_length  # 外接矩形面积
	if rect_length != 0:  # 外接矩形宽高比
		aspect = rect_width / rect_length
		# aspect = rect_length / rect_width
	else:
		aspect = 0

	print aspect_min, aspect_max, aspect

	# if (area <= area_max and area >= area_min) and (aspect <= aspect_max and aspect >= aspect_min):
	if aspect <= aspect_max and aspect >= aspect_min:
		return True
	else:
		return False

# 尺寸判断
def verifysize(rect):
	# 中国车牌尺寸: 440mm*140mm，宽高比3.142857
	# 实际车牌尺寸: 136*32，宽高比4.25
	aspect_std = 3.75				#标准宽高比，3.75
	area_std = 34 * 8				#标准面积，34*8
	
	error = 0.5			#偏差，根据easyPR，error=0.9
	aspect_min = aspect_std - aspect_std * error		#最小宽高比
	aspect_max = aspect_std + aspect_std * error		#最大宽高比
	
	verifyMin_area = 3			#偏差，根据easyPR，min=1，max=24
	verifyMax_area = 40
	area_min = area_std * verifyMin_area		#最小面积
	area_max = area_std * verifyMax_area		#最大面积
	
	rect_width = rect[1][0]			#外接矩形宽
	rect_length = rect[1][1]		#外接矩形长
	
	area = rect_width * rect_length		#外接矩形面积
	if rect_length != 0:					#外接矩形宽高比
		aspect = rect_width / rect_length
	else:
		aspect = 0

	# print aspect_min, aspect_max, aspect
	
	if (area <= area_max and area >= area_min) and (aspect <= aspect_max and aspect >= aspect_min):
		return True
	else:
		return False
		
# 角度判断
def verifyangle(rect):
	angle = rect[2]
	if angle >= -30 and angle <=30:
		return True
	else:
		return False

# 临时车牌调整
def plateadjustment(image, region_tmp_plates):
	height = 36
	width = 136
	tmp_plates = []

	for region_tmp_plate in region_tmp_plates:
		tmp_plate = image[region_tmp_plate[1]:region_tmp_plate[3], region_tmp_plate[0]:region_tmp_plate[2]]		#切割图片
		# imgshow(tmp_苏A0CP56plate, "TempPlate")
		angle = region_tmp_plate[4]
		
		rows, cols = tmp_plate.shape[:2]		#旋转
		M = cv2.getRotationMatrix2D((cols / 2, rows / 2,), angle, 1)
		tmp_plate_hori = cv2.warpAffine(tmp_plate, M, (cols, rows))
		# imgshow(tmp_plate_hori, "HorizontalTempPlate")
		
		tmp_plate_std = cv2.resize(tmp_plate_hori, (width, height))			#大小调整
		# imgshow(tmp_plate_std, "StdTempPlate")

		tmp_plates.append(tmp_plate_std)

	return tmp_plates

# 存储图片
def saveplate(tmp_plates, filename):
	if os.path.exists("temp_plates/" + filename):
		shutil.rmtree("temp_plates/" + filename)
	os.mkdir("temp_plates/" + filename)

	for i in range(len(tmp_plates)):
		cv2.imwrite("temp_plates/"+ filename +"/plates_" + str(i) + ".png", tmp_plates[i])

#获取候选车牌
def getplate(image, filename):
	region_tmp_plates, boxs = gettmpplates_sobel(image)
	# colorlocate(filename)

	# image = cv2.imread(filename)
	tmp_plates = plateadjustment(image, region_tmp_plates)
	saveplate(tmp_plates, filename)

	return boxs, tmp_plates