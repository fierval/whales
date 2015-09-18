import cv2
import numpy as np
import pandas as pd
from os import path
from kobra.imaging import show_images

root = "/kaggle/whales/imgs"
img_file = "w_39.jpg"

img_ = path.join(root, img_file)

img = cv2.imread(img_)

### preprocess

#pyramid down
img_shift = cv2.pyrDown(img)
img_shift = cv2.pyrDown(img_shift)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)
#img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_CLOSE, kernel)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)

#clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (16, 16))
#hsv = cv2.cvtColor(img_shift, cv2.COLOR_BGR2HSV)
#(h, s, v) = cv2.split(hsv)
#vE = clahe.apply(v)
#hsv = cv2.merge((h, s, vE))
#img_shift = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#show_images([img, img_shift])

# convert to np.float32
Z = img_shift.reshape((-1,3))
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_shift.shape))

show_images([img, res2])