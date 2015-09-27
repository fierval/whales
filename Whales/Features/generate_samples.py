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
        im = cv2.imread(fl)
        cv2.imwrite(path.join(out_dir, dir, f), im)

        for rot in range(20, 100, 20):
            for scale in np.linspace(1, 1, 1):
                im_rot = imutils.rotate(im, rot, scale = scale)

                new_name = path.splitext(path.split(fl)[1])[0]
                new_name = "{0}_{1}_{2}".format(new_name, rot, scale).replace(".", "_") + ".jpg"

                cv2.imwrite(path.join(out_dir, dir, new_name), im_rot)
                print "rotated: {0} by {1} scaled by {2}".format(f, rot, scale)
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