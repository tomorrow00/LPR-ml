#coding=utf8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.externals import joblib

# 显示图片
def imgshow(image, title):
	'''(r, g, b) = cv2.split(image)
	image = cv2.merge([b, g, r])
	
	plt.title(title)
	plt.imshow(image)
	plt.show()'''
	
	cv2.imshow(title, image)
	cv2.waitKey(0)

# 图片处理
def imgprocess(image):
	#image_blur = cv2.GaussianBlur(image, (5, 5), 0)		#高斯模糊
	#imgshow(image_blur, "Blur")
	
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		#灰度化
	#imgshow(image_gray, "Gray")
	
	#ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		#二值化
	
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

# 测试
def test(hist):
	clf = joblib.load("model_plates.m")
	result = clf.predict(hist)
	
	return result[0]

def main():
	# files = os.listdir("temp_plates")
	# files = os.listdir("svm/has/test")
	# sum = len(files)

	filenames = []
	filenames.append("plates_22.png")
	# for file in files:
	# 	#filenames.append(os.path.join("temp_plates", file))
	# 	filenames.append(os.path.join("svm/has/test", file))

	hists = []
	correct = 0
	for filename in filenames:
		image = cv2.imread(filename)
		image = imgprocess(image)
		hist = LBPfeature(image)
		hist = np.array(hist, dtype = "float32")
		hist = hist.reshape(1, -1)

		result = test(hist)
		print result

		if result == 1:
			correct += 1

	print correct
	# print sum
	rate = float(float(correct) / float(sum))
	print rate


if __name__ == '__main__':
	main()
