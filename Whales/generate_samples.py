import imutils
import numpy as np
import pandas as pd
import os
from os import path
import cv2

root = "/Kaggle/whales/train"

files = os.listdir(root)

dirs = filter(lambda f: path.isdir(path.join(root,f)) ,files)

def generate_samples(dir):
    for dir in dirs:
        for f in os.listdir(path.join(root,dir)):
            fl = path.join(root, dir, f)
            im = cv2.imread(fl)

            for rot in range(20, 180, 20):
                for scale in np.linspace(1.0, 2.5, 3):
                    im_rot = imutils.rotate(im, rot, scale = scale)

                    new_name = path.splitext(fl)[0]
                    new_name = "{0}_{1}_{2}".format(new_name, rot, scale).replace(".", "_") + ".jpg"

                    cv2.imwrite(new_name, im_rot)

                    print "rotated: {0} by {1} scaled by {2}".format(f, rot, scale)

for dir in dirs:
    generate_samples(dir)