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
from keras.layers.normalization import BatchNormalization
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
model.add(BatchNormalization((3, 256, 256)))
model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((32, 262, 262)))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((64, 135, 135)))
model.add(Convolution2D(64, 64, 5, 5)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 64, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((128, 69, 69)))
model.add(Convolution2D(128, 128, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(256, 128, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((256, 36, 36)))
model.add(Convolution2D(256, 256, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(512, 256, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((512, 20, 20)))
model.add(Convolution2D(512, 512, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(1024, 512, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((512, 20, 20)))
model.add(Convolution2D(1024, 1024, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(9216, 4000))
model.add(Activation('relu'))
model.add(BatchNormalization((4000,)))
model.add(Dropout(0.5))

model.add(Dense(4000, 2000))
model.add(Activation('relu'))
model.add(BatchNormalization((2000,)))
model.add(Dropout(0.5))

model.add(Dense(2000, 1000))
model.add(Activation('relu'))
model.add(BatchNormalization((1000,)))
model.add(Dropout(0.5))

model.add(Dense(1000, 447))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#grapher = Grapher()
#grapher.plot(model, "/temp/graph.png")

nb_epoch = 5
batch_size = 3000
nb_samples = 30

from kobra.tr_utils import time_now_str
print("Start time: " + time_now_str())

#x_val, y_val = BatchGenerator(train_path, labels_map, batch_size).get_val()

for e in range(nb_epoch):
    print("Epoch %d" % e)
    batches = BatchGenerator(train_path, labels_map, batch_size)

    progbar = generic_utils.Progbar(batches.total)
    for x_train, y_train in batches:
               
        iterations = x_train.shape[0] / nb_samples if x_train.shape[0] % nb_samples == 0 else (x_train.shape[0] + nb_samples) / nb_samples
        
        for i in range(iterations):
            X_batch = x_train[i * nb_samples : (i + 1) * nb_samples]
            Y_batch = y_train[i * nb_samples : (i + 1) * nb_samples]
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values= [("train loss", loss)])

    loss, accuracy = model.evaluate(x_val, y_val, show_accuracy=True)
    print ("Validation: loss - {0}, accuracy - {1}".format(loss, accuracy))
         
print("End time: " + time_now_str())

json_string = model.to_json()
open('/users/boris/dropbox/kaggle/whales/models/model_1.json', 'w').write(json_string)
model.save_weights('/users/boris/dropbox/kaggle/whales/models/weights_1.h5', overwrite=True)