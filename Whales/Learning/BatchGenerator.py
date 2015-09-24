import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2
from sklearn.cross_validation import train_test_split

train_path = "/Kaggle/whales/kerano"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"

from keras.utils import np_utils

class DataSetLoader(object):
    def __init__(self, train_path, labels_file, labels_map):
        labels = pd.read_csv(labels_file)
        labels_categ_dict = {key: value for key, value in labels.values}

        labls = pd.read_csv(labels_map, header=None)
        ldict = {key: value for key, value in labls.values}

        files = map(lambda f: path.join(train_path, f), os.listdir(train_path))
        self.X_train = np.array(map(cv2.imread, files))
        names = map(lambda f: path.split(f)[1], files)
        self.Y_train = np_utils.to_categorical(map(lambda n: ldict[labels_categ_dict[n]], names))
    
    def get_fraction(self, n = .8):
        X_train, _, Y_train, _ = train_test_split(self.X_train, self.Y_train, train_size = n)
        return X_train, Y_train

class BatchGenerator(object):

    def __init__(self, train_path, labels_map, n = 500):
        self.current = 0
        self.files = []
        self.n = n

        dirs = filter(path.isdir, map(lambda f: path.join(train_path, f), os.listdir(train_path)))
        labls = pd.read_csv(labels_map, header=None)
        self.ldict = {path.join(train_path, key): value for key, value in labls.values}

        for d in dirs:
            dir = path.join(train_path, d)
            files = os.listdir(dir)
            files = map(lambda f: path.join(dir, f), files)
            self.files += files

        self.total = len(self.files)

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        if self.current > self.total:
            raise StopIteration
        else:
            y_train = []
            x_train = []
            i = 0
            while(i < self.n and self.current < self.total):
                im_path = self.files[self.current + i]
                im = cv2.imread(im_path)
                lab_path = path.split(im_path)[0]
                y_train += [self.ldict[lab_path]]
                x_train += [im]
                i += 1
                self.current += 1
            return (x_train, y_train)             