from matplotlib import pyplot as plt
from os import path
import numpy as np
import cv2
import pandas as pd
from kobra.imaging import *
from kobra.dr.retina import display_contours

def saturate (v):
    return map(lambda a: min(max(round(a), 0), 255), v)

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

def kmeans(image):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

def process(image_titles, images):

    ims_pre = map(pre, images)

    # get the masks
    masks = np.array(get_masks(ims_pre))

    ref_im_pre = pre(ref_image)

    refMask = get_masks([ref_im_pre])
    refHist = calc_hist([ref_im_pre], refMask)

    histSpec = histogram_specification(refHist, ims_pre, masks)

    #cdfs = calc_hist(ims_pre, masks)

    #plot_hist(cdfs[0][1], "g")
    kmeans_imgs = map(kmeans, histSpec)

    # mask off the background
    show_images(kmeans_imgs, titles = image_titles, scale = 0.9)

    return kmeans_imgs

def contours(imgs):

