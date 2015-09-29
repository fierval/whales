import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2
from sklearn.cross_validation import train_test_split

train_path = "/Kaggle/whales/augmented"
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
        # Theano wants depth (# of channels) to be the first dimension
        self.X_train = np.transpose(self.X_train, (0, 3, 1, 2)).astype('f')
        names = map(lambda f: path.split(f)[1], files)
        self.Y_train = np_utils.to_categorical(map(lambda n: ldict[labels_categ_dict[n]], names))
    
    def get_fraction(self, n = .8):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X_train, self.Y_train, train_size = n)
        return X_train, Y_train, X_test, Y_test

class BatchGenerator(object):

    def __init__(self, train_path, labels_map, n = 500, val_split = 0.2):
        self.current = 0
        self.current_val = 0
        self.files = []
        self.n = n
        self._validate = False

        dirs = filter(path.isdir, map(lambda f: path.join(train_path, f), os.listdir(train_path)))
        labls = pd.read_csv(labels_map, header=None)
        self.ldict = {path.join(train_path, key): value for key, value in labls.values}

        for d in dirs:
            dir = path.join(train_path, d)
            files = os.listdir(dir)
            files = map(lambda f: path.join(dir, f), files)
            self.files += files
        
        np.random.shuffle(self.files)

        if val_split > 0:
            self.files, self.val = train_test_split(self.files, train_size = 1. - val_split)
            self.total_val = len(self.val)

        self.total = len(self.files)

    def __iter__(self):
        return self

    @property
    def validate(self):
        return self._validate

    @validate.setter
    def validate(self, val):
        self._validate = val

    def next(self): # Python 3: def __next__(self)
        if self.validate:
            return self.next_val()
        else:
            return self.next_train()

    def next_train(self):
        if self.current >= self.total:
            raise StopIteration
        else:
            y_train = []
            x_train = []
            paths = self.files[self.current : self.current + self.n]
            self.current += self.n

            for im_path in paths:
                im = cv2.imread(im_path)
                lab_path = path.split(im_path)[0]
                y_train += [self.ldict[lab_path]]
                x_train += [im]

            return np.array(x_train).astype('f').transpose(0, 3, 1, 2), np_utils.to_categorical(y_train, nb_classes = len(self.ldict))

    def next_val(self):
        if self.current_val >= self.total_val:
            raise StopIteration
        else:
            y_train = []
            x_train = []
            paths = self.val[self.current : self.current + self.n]
            self.current_val += self.n

            for im_path in paths:
                im = cv2.imread(im_path)
                lab_path = path.split(im_path)[0]
                y_train += [self.ldict[lab_path]]
                x_train += [im]

            return np.array(x_train).astype('f').transpose(0, 3, 1, 2), np_utils.to_categorical(y_train, nb_classes = len(self.ldict))  