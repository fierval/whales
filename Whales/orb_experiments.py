import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import path
from kobra.imaging import show_images

root = "/kaggle/whales/train"
sample_dir = "whale_03728"
im_file = "w_280.jpg"
im_inp_file = path.join(root, sample_dir, im_file)
img = cv2.imread(im_inp_file)
img0 = img
im_file_1 = "w_280_20_0_8.jpg"
im_inp_file_1 = path.join(root, sample_dir, im_file)
img1 = cv2.imread(im_inp_file)

# Initiate STAR detector
def orb_show(img):
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    img2 = np.zeros_like(img)
    # draw only keypoints location,not size and orientation
    img3 = cv2.drawKeypoints(img, kp, img2, color=(0,255,0), flags=0)
    show_images([img3])

def orb_match(img0, img1):

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img0, None)
    kp2, des2 = orb.detectAndCompute(img1,None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12
                   key_size = 10,     # 20
                   multi_probe_level = 2) #2    search_params = dict(checks=50)   # or pass empty dictionary
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()