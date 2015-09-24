from IPython.parallel import Client
from kobra.tr_utils import time_now_str, prep_out_path
c = Client()

%%px --local
from matplotlib import pyplot as plt
from os import path
import os
import numpy as np
import cv2
import pandas as pd
from kobra.imaging import *
from kobra.dr.retina import display_contours

# given a list of images, create masks for them
def get_masks(images):
    def mask(im):
        mask = np.zeros_like(im[:,:,0])
        mask[:]=255
        return mask
    return map(lambda i: mask(i), images)

img_path = "/kaggle/whales/color"
out_path = "/kaggle/whales/cropped"
size = (256, 256)
image_names = os.listdir(img_path)

image_paths = map(lambda t: path.join(img_path, t), image_names)

def pre(img):
    img_shift = cv2.pyrDown(img)
    img_shift = cv2.pyrDown(img_shift)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)

    return img_shift

def kmeans(image):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2, center, label.flatten().reshape((image[:,:,0].shape))

def crop(image_name):
    image = cv2.imread(image_name)
    ims_pre = pre(image)
    im_tuple = kmeans(ims_pre)

    out_name = path.split(image_name)[1]
    out_im_name = path.join(out_path, out_name)

    proc_im = im_tuple[0]
    center = im_tuple[1]
    labels = im_tuple[2]

    lab_quants = [labels[labels == i].shape[0] for i in range(0,3)]
    max_ind = np.argmax(lab_quants)
    min_ind = np.argmin(lab_quants)

    middle = filter(lambda r: r != max_ind and r != min_ind, range(0, 3))[0]

    lower = center[max_ind]
    upper = center[middle]
    mask = cv2.inRange(proc_im, lower, upper)

    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get 10 largest areas, except for the last one
    areas = map(cv2.contourArea, cnts)
    contour_areas_sorted = np.argsort(areas)[:-1]

    cnts = [cnts[i] for i in contour_areas_sorted[::-1][:10]]
    if len(cnts) == 0:
        cv2.imwrite(out_im_name, cv2.resize(image, (256, 256)))
    else:
        # bounding rectangle for cropping
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))

        # good for debugging
        #out_rect = cv2.rectangle(proc_im.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        #plt.imshow(out_rect); plt.show()

        # now crop it
        x1, y1, w1, h1 = tuple(map(lambda e: e * 4, (x, y, w, h)))
        toSave = cv2.resize(image[y1:y1+h1, x1:x1+w1, :], (256, 256))

        cv2.imwrite(out_im_name, toSave)

    return out_im_name

prep_out_path(out_path)
dv = Client().load_balanced_view()
fs = dv.map(crop, np.array(image_paths))
print "Started: ", time_now_str()
fs.wait()
print "Finished: ", time_now_str()

