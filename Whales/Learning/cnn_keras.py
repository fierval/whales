from __future__ import print_function

import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2
from BatchGenerator import DataSetLoader, BatchGenerator

from sklearn.utils import shuffle as sk_shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils.dot_utils import Grapher
from keras.utils import generic_utils

train_path = "/Kaggle/whales/augmented"
val_path = "/kaggle/whales/kerano"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"

#dsl = DataSetLoader(val_path, labels_file, labels_map)
#X_train, Y_train, X_test, Y_test = dsl.get_fraction(.8)

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

model.add(Convolution2D(128, 96, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(128, 128, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(160, 128, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(Convolution2D(160, 160, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(10240, 2000))
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

nb_epoch = 10
batch_size = 2000
nb_samples = 20

batches = BatchGenerator(train_path, labels_map, batch_size)
print("Total dataset: %d" % batches.total)

for e in range(nb_epoch):
    print("epoch %d" % e)
    progbar = generic_utils.Progbar(batches.total)
    for x_train, y_train in batches:
        # shuffle
        
        x_train, y_train = sk_shuffle(x_train, y_train, random_state = 0)
                
        iterations = x_train.shape[0] / nb_samples if x_train.shape[0] % nb_samples == 0 else (x_train.shape[0] + nb_samples) / nb_samples
        
        for i in range(iterations):
            X_batch = x_train[i * nb_samples : (i + 1) * nb_samples]
            Y_batch = y_train[i * nb_samples : (i + 1) * nb_samples]
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values= [("train loss", loss)])

json_string = model.to_json()
open('/users/boris/dropbox/kaggle/whales/models/model_1.json', 'w').write(json_string)
model.save_weights('/users/boris/dropbox/kaggle/whales/models/weights_1.h5', overwrite=True)