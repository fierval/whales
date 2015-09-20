import pandas as pd
import numpy as np
import os
from os import path
from kobra.tr_utils import prep_out_path
import shutil
from shutil import copy

labels_file = "/Kaggle/whales/train.csv"    
inp_path = "/Kaggle/whales/c1"
out_path = "/Kaggle/whales/train"
test_path = "/Kaggle/whales/test"

def copy_files_to_label_dirs(inp_path, out_path, labels_file):
    prep_out_path(out_path)
    
    labels = pd.read_csv(labels_file)
    splitter = labels.columns[1]
    
    dirs = np.unique(labels[splitter].as_matrix())
    for dir in dirs:
        p = path.join(out_path, dir)
        os.makedirs(p)

    bad = []
    for f, l in zip(labels[labels.columns[0]], labels[labels.columns[1]]):
        file_name = path.join(out_path, l, f)
        inp_file = path.join(inp_path, f)
        try:
            shutil.copy(inp_file, file_name)
        except IOError:
            print "Cannot copy: {0}".format(f)
            bad += [f]
            continue
        print "copied {0} to {1}".format(inp_file, file_name)

    print bad

def copy_test_files(inp_path, test_path, labels_file):
    prep_out_path(test_path)
    labels = pd.read_csv(labels_file)
    file_names = set(labels[labels.columns[0]].as_matrix())
    all_files = set(os.listdir(inp_path))
    test_files = all_files.difference(file_names)

    for f in test_files:
        copy(path.join(inp_path, f), path.join(test_path, f))

