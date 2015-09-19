from matplotlib import pyplot as plt
from os import path
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

img_path = "/kaggle/whales/imgs"
# this is the image into the colors of which we want to map
ref_image_name = "w_188.jpg"
# images picked to illustrate different problems arising during algorithm application
image_names = ["w_31.jpg", "w_25.jpg", "w_190.jpg"]
#image_names = ["w_81.jpg"]

image_paths = map(lambda t: path.join(img_path, t), image_names)
images = np.array(map(lambda p: cv2.imread(p), image_paths))
image_titles = map(lambda i: path.splitext(i)[0], image_names)

ref_image = cv2.imread(path.join(img_path, ref_image_name))

def pre(img):
    img_shift = cv2.pyrDown(img)
    img_shift = cv2.pyrDown(img_shift)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_shift = cv2.morphologyEx(img_shift, cv2.MORPH_OPEN, kernel)

    clahe = cv2.createCLAHE(clipLimit = 2, tileGridSize = (16, 16))
    hsv = cv2.cvtColor(img_shift, cv2.COLOR_BGR2HSV)
    (h, s, v) = cv2.split(hsv)
    vE = clahe.apply(v)
    hsv = cv2.merge((h, s, vE))
    img_shift = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img_shift

# preprocess ref image
ref_im_pre = pre(ref_image)

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

# takes pre-processed images
# prep
def process(ims_pre):
    # get the masks (just stubs)
    masks = np.array(get_masks(ims_pre))

    refMask = get_masks([ref_im_pre])
    refHist = calc_hist([ref_im_pre], refMask)

    histSpec = histogram_specification(refHist, ims_pre, masks)
    kmeans_imgs = map(kmeans, histSpec)

    #show_images(map(lambda x: x[0], kmeans_imgs), scale = 0.9)
    return kmeans_imgs

def crop(images):
    ims_pre = map(pre, images)
    ims = process(ims_pre)

    for j, im_tuple in enumerate(ims):
        proc_im = im_tuple[0]
        center = im_tuple[1]
        labels = im_tuple[2]

        lab_quants = [labels[labels == i].shape[0] for i in range(0,3)]
        max_ind = argmax(lab_quants)
        min_ind = argmin(lab_quants)

        middle = filter(lambda r: r != max_ind and r != min_ind, range(0, 3))[0]

        lower = center[argmax(lab_quants)]
        upper = center[middle]
        mask = cv2.inRange(proc_im, lower, upper)

        _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get 10 largest areas, except for the last one
        areas = map(cv2.contourArea, cnts)
        contour_areas_sorted = argsort(areas)[:-1]

        cnts = [cnts[i] for i in contour_areas_sorted[::-1][:10]]
        # bounding rectangle for cropping
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))

        # good for debugging
        #out_rect = cv2.rectangle(proc_im.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        #plt.imshow(out_rect); plt.show()

        # now crop it
        yield ims_pre[j][y:y+h, x:x+w, :] 