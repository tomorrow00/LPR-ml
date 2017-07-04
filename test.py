#coding=utf8
import cv2
import numpy as np

image = cv2.imread("tagged_苏A0CP56.png")

temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(temp, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 40)
print lines

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

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("Asdasdasdasd", image)
cv2.waitKey(0)
