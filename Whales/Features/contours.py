import numpy as np
import cv2
from os import path
import os
import matplotlib.pyplot as plt

root_path = "/kaggle/whales/imgs"

def nothing(*arg):
    pass

def createMask((rows, cols), hull):
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.drawContours(mask, [hull], 0, 255, -1)
    return mask

def loadMask(file_path, size):
    return np.fromfile(file_path).reshape(size, size) 

thresh = 4
size = 256

file_name = "w_39.jpg"

sample_image_path = path.join(root_path, file_name)  #threshold 22

#sample_image_path = path.join(processed_path, "5784_right.jpeg") 

srcImage = cv2.resize(cv2.imread(sample_image_path), (size, size))
srcGrey = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
#srcGrey = cv2.resize(cv2.GaussianBlur(srcGrey, (7, 7), 30), (size, size))

src = np.copy(srcImage)
srcG = np.copy(srcGrey)

ret, thresholded = cv2.threshold(srcG, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#polys = cv2.approxPolyDP(contours[0], 5, True)
#for cnt in contours[1:]:
#   polys = np.vstack((polys, cv2.approxPolyDP(cnt, 5, True)))
hull_contours = cv2.convexHull(np.vstack(contours))
hull = np.vstack(hull_contours)
hull_area = cv2.contourArea(hull)
poly = cv2.approxPolyDP(hull, 5, True)

for i in range(0, len(contours)):
    cv2.drawContours(src, contours, i, (255, 0, 0), 1)

cv2.drawContours(src, [hull], 0, (0, 0, 255), 1)
cv2.drawContours(src, [poly], 0, (0, 255, 0), 1)

im_show = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
plt.imshow(im_show)
plt.show()

