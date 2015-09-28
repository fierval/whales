from IPython.parallel import Client
c = Client()

%%px --local
import imutils
import numpy as np
import pandas as pd
import os
from os import path
import cv2
from kobra.tr_utils import time_now_str, prep_out_path
import shutil
from scipy import ndimage, misc

root = "/Kaggle/whales/kerano"
out_dir = "/kaggle/whales/augmented"
labels_map = "/Kaggle/whales/labels_map.csv"
labels_file = "/kaggle/whales/train.csv"

dirs = pd.read_csv(labels_map, header=None).icol(0).values
labels = pd.read_csv(labels_file)

def generate_samples(dir):
    files = labels[labels['whaleID'] == dir].icol(0).values
    for f in files:
        fl = path.join(root, f)
        im = ndimage.imread(fl)
        misc.imsave(path.join(out_dir, dir, f), im)
        new_name = path.splitext(path.split(fl)[1])[0]

        for rot in range(20, 360, 20):
            im_rot = ndimage.interpolation.rotate(im, rot, axes=(0,1), reshape=False, mode="nearest", cval=0.)

            save_name = "{0}_{1}".format(new_name, rot).replace(".", "_") + ".jpg"

            misc.imsave(path.join(out_dir, dir, save_name), im_rot)
            print "rotated: {0} by {1}".format(f, rot)
        
        # horizontal flip
        #im_rot = np.fliplr(im)
        #misc.imsave(path.join(out_dir, dir, new_name + "_hflip" + ".jpg"), im_rot)
        #im_rot = np.flipud(im)
        #misc.imsave(path.join(out_dir, dir, new_name + "_vflip" + ".jpg"), im_rot)

    return dir, len(files)

#for dir in dirs:
#    generate_samples(dir)
prep_out_path(out_dir)
for dir in dirs:
    os.makedirs(path.join(out_dir, dir))

dv = Client().load_balanced_view()
fs = dv.map(generate_samples, np.array(dirs))
print "Started: ", time_now_str()
fs.wait()
print "Finished: ", time_now_str()