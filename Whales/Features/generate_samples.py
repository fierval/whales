from IPython.parallel import Client
c = Client()

%%px --local
import imutils
import numpy as np
import pandas as pd
import os
from os import path
import cv2
from kobra.tr_utils import time_now_str

root = "/Kaggle/whales/train"

files = os.listdir(root)

dirs = filter(lambda f: path.isdir(path.join(root,f)) ,files)

def generate_samples(dir):
    files = os.listdir(path.join(root,dir))
    for f in files:
        fl = path.join(root, dir, f)
        im = cv2.imread(fl)

        for rot in range(20, 360, 20):
            for scale in np.linspace(1, 1, 1):
                im_rot = imutils.rotate(im, rot, scale = scale)

                new_name = path.splitext(fl)[0]
                new_name = "{0}_{1}_{2}".format(new_name, rot, scale).replace(".", "_") + ".jpg"

                cv2.imwrite(new_name, im_rot)

                print "rotated: {0} by {1} scaled by {2}".format(f, rot, scale)
    return dir, len(files)

#for dir in dirs:
#    generate_samples(dir)

dv = Client().load_balanced_view()
fs = dv.map(generate_samples, np.array(dirs))
print "Started: ", time_now_str()
fs.wait()
print "Finished: ", time_now_str()