import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2

train_path = "/Kaggle/whales/train"
labels_map = "/Kaggle/whales/labels_map.csv"

class BatchGenerator(object):

    def __init__(self, train_path, labels_map, n = 500):
        self.current = 0
        self.files = []
        self.n = n

        self.dirs = filter(path.isdir, map(lambda f: path.join(self.train_path, f), os.listdir(self.train_path)))
        self.names = map(lambda d: path.split(d)[1], dirs)

        labls = pd.read_csv(self.labels_map, header=None)
        self.ldict = {path.join(train_path, key): value for key, value in labls.values}
        for d in dirs:
            dir = path.join(train_path, d)
            files = os.listdir(dir)
            files = map(lambda f: path.join(dir, f), files)
            self.files += files

    def __iter__(self):
        return self

    def next(self): # Python 3: def __next__(self)
        if self.current > self.high:
            raise StopIteration
        else:
            y_train = []
            x_train = []
            i = 0
            while(i < self.n and self.current < len(self.files)):
                im_path = self.files[self.current + i]
                im = cv2.imread(im_path)
                lab_path = path.split(im_path)[0]
                y_train += [self.ldict[lab_path]]
                x_train += [im]
                i += 1
                self.current += 1
            return (x_train, y_train)             

    def build_data_set(self):
        dirs = filter(path.isdir, map(lambda f: path.join(self.train_path, f), os.listdir(self.train_path)))
        names = map(lambda d: path.split(d)[1], dirs)

        labls = pd.read_csv(self.labels_map, header=None)
        ldict = {key: value for key, value in labls.values}

        Y_train = []
        X_train = []
        for dir, name in zip(dirs, names):

            files = os.listdir(dir)
            for f in files:
                Y_train += [ldict[name]]
                im = cv2.imread(path.join(dir, f))
                X_train += [im]      

