#coding=utf8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.externals import joblib
import datetime

#打开文件
def openfiles(files, filepath):
	filenames = []
	for file in files:
		filenames.append(os.path.join(filepath, file))
	
	return filenames

#显示图片
def imgshow(image, title):
	'''(r, g, b) = cv2.split(image)
	image = cv2.merge([b, g, r])
	
	plt.title(title)
	plt.imshow(image)
	plt.show()'''
	
	cv2.imshow(title, image)
	cv2.waitKey(0)

#图片处理
def imgprocess(image):
	#image_blur = cv2.GaussianBlur(image, (5, 5), 0)		#高斯模糊
	#imgshow(image_blur, "Blur")
	
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		#灰度化
	#imgshow(image_gray, "Gray")
	
	#ret, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)		#二值化
	
	return image_gray

#提取LBP特征
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
	# print max_bins
	hist, edges = np.histogram(image_lbp, bins=max_bins, normed=True)

	return hist

#训练
def train(hists, labels):
	'''svm = cv2.ml.SVM_create()		#建立SVM模型
	svm.setType(cv2.ml.SVM_C_SVC)
	#svm.setKernel(cv2.ml.SVM_RBF)
	#svm.setGamma(0.1)
	#svm.setCoef0(0.1);
	#svm.setC(1);
	svm.setNu(0.1);
	svm.setP(0.1);
	
	svm.train(hists, cv2.ml.ROW_SAMPLE, labels)
	svm.save("model.m")'''
	
	clf = svm.SVC(gamma=0.1, coef0=0.1, kernel='rbf')		#建立SVM模型
	clf.fit(hists, labels)
	joblib.dump(clf, "model_plates.m")

def main():
	labels = []

	has_files = os.listdir("svm/has/train")
	for i in range(len(has_files)):
		labels.append(int(1))

	no_files = os.listdir("svm/no/train")
	for i in range(len(no_files)):
		labels.append(int(0))
	
	filenames = []
	filenames.extend(openfiles(has_files, "svm/has/train"))
	filenames.extend(openfiles(no_files, "svm/no/train"))
	
	hists = []
	num = 1
	output = open("data.txt", "w")
	for filename in filenames:
		image = cv2.imread(filename)
		image = imgprocess(image)
		hist = LBPfeature(image)
		hists.append(hist)

		output.write(str(num))
		output.write("|")
		for i in range(len(hist)):
			output.write(str(hist[i]))
			output.write(" ")

		output.write("\n=======================================================================================================\n")
		num += 1

	output.close()

	hists = np.array(hists, dtype = "float32")
	labels = np.array(labels, dtype = "int32")

	print "Start Training...", "Time:" + str(datetime.datetime.now())
	train(hists, labels)
	print "Training Completed.", "Time:" + str(datetime.datetime.now())

if __name__ == '__main__':
	main()