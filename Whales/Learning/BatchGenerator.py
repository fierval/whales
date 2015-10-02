import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2
from sklearn.cross_validation import train_test_split

train_path = "/Kaggle/whales/train384"
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
        np.random.shuffle(files)

        self.X_train = np.array(map(cv2.imread, files))
        # Theano wants depth (# of channels) to be the first dimension
        self.X_train = np.transpose(self.X_train, (0, 3, 1, 2)).astype('f')
        names = map(lambda f: path.split(f)[1], files)
        self.Y_train = np_utils.to_categorical(map(lambda n: ldict[labels_categ_dict[n]], names))
    
    def get_fraction(self, n = .8):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_train, self.Y_train, train_size = n)
        return X_train, Y_train, X_test, Y_test

class BatchGenerator(object):

    def __init__(self, train_path, labels_file, labels_map, n = 500, val_split = 0.2):
        self.current = 0
        self.current_val = 0
        self.files = []
        self.n = n

        labels = pd.read_csv(labels_file)
        self.labels_categ_dict = {key: value for key, value in labels.values}

        dirs = filter(path.isdir, map(lambda f: path.join(train_path, f), os.listdir(train_path)))
        labls = pd.read_csv(labels_map, header=None)
        self.flat_struct = len(dirs) == 0
        self.ldict = {}
        if self.flat_struct:
            self.ldict =  {key: value for key, value in labls.values}
        else:
            self.ldict =  {path.join(train_path, key): value for key, value in labls.values}

        if self.flat_struct:
            self.files = map(lambda f: path.join(train_path, f), os.listdir(train_path))
        else:
            for d in dirs:
                dir = path.join(train_path, d)
                files = os.listdir(dir)
                files = map(lambda f: path.join(dir, f), files)
                self.files += files
        
        np.random.shuffle(self.files)

        if val_split > 0:
            self.files, self.val = train_test_split(self.files, test_size = val_split)
            self.total_val = len(self.val)

        self.total = len(self.files)

    def __iter__(self):
        return self

    def _read_ims(self, paths):
        y_train = []
        x_train = []

        for im_path in paths:
            im = cv2.imread(im_path)
            # TODO: Enable augmented names
            if self.flat_struct:
                name = path.split(im_path)[1]
                y_train = self.ldict[self.labels_categ_dict[name]]
            else:
                lab_path = path.split(im_path)[0]
                y_train += [self.ldict[lab_path]]
            x_train += [im]

        return np.array(x_train).astype('f').transpose(0, 3, 1, 2), np_utils.to_categorical(y_train, nb_classes = len(self.ldict))

    def next(self):
        if self.current >= self.total:
            raise StopIteration
        else:
            paths = self.files[self.current : self.current + self.n]
            self.current += self.n
            return self._read_ims(paths)

    def get_val(self):
        return self._read_ims(self.val)
        