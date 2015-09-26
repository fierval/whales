﻿import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2
from BatchGenerator import DataSetLoader

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.dot_utils import Grapher
from keras.utils import generic_utils

train_path = "/Kaggle/whales/kerano"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"

#def train(train_path, labels_file, labels_map):
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    horizontal_flip=True)

dsl = DataSetLoader(train_path, labels_file, labels_map)
X_train, Y_train, X_test, Y_test = dsl.get_fraction(.7)
datagen.fit(X_train, augment=True)

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

model.add(Convolution2D(96, 64, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(96, 96, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 96, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(16384, 2000))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2000, 1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1000, 447))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#grapher = Grapher()
#grapher.plot(model, "/temp/graph.png")

#model.fit(X_train, Y_train, batch_size=5, nb_epoch=1)

nb_epoch = 10
# batch train with realtime data augmentation
progbar = generic_utils.Progbar(X_train.shape[0])
for e in range(nb_epoch):
    print('-'*40)
    print('Epoch', e)
    print('-'*40)
    print("Training...")
    # batch train with realtime data augmentation
    for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=10):
        loss = model.train_on_batch(X_batch, Y_batch)
        progbar.add(X_batch.shape[0], values=[("train loss", loss)])
