import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

train_path = "/Kaggle/whales/train"
labels_map = "/Kaggle/whales/labels_map.csv"

def build_data_set(train_path, labels_map):
    dirs = filter(path.isdir, map(lambda f: path.join(train_path, f), os.listdir(train_path)))
    names = map(lambda d: path.split(d)[1], dirs)

    labls = pd.read_csv(labels_map, header=None)
    ldict = {key: value for key, value in labls.values}

    Y_train = []
    X_train = []
    for dir, name in zip(dirs, names):

        files = os.listdir(dir)
        for f in files:
            Y_train += [ldict[name]]
            im = cv2.imread(path.join(dir, f))
            X_train += [im]      

def train(X_train, Y_train):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3)) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64*8*8, 1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, 447))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(X_train, Y_train, batch_size=10, nb_epoch=1)