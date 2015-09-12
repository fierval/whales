import cv2
import numpy as np
from os import path
from kobra.imaging import show_images

root = "/kaggle/whales/imgs"
img_file = "w_686.jpg"

img_ = path.join(root, img_file)

img = cv2.imread(img_)

### preprocess

#pyramid down
img_shift = cv2.pyrDown(img)
img_shift = cv2.pyrDown(img_shift)

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