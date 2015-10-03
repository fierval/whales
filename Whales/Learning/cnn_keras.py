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
from keras.layers.normalization import BatchNormalization, LRN2D
from keras.optimizers import SGD, Adagrad
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils.dot_utils import Grapher
from keras.utils import generic_utils

train_path = "/Kaggle/whales/kerano"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"
labels_file = "/Kaggle/whales/train.csv"

model = Sequential()
#model.add(BatchNormalization((3, 384, 384)))
model.add(Convolution2D(32, 3, 3, 3, border_mode='full', init='glorot_normal')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((32, 390, 390)))
model.add(Convolution2D(32, 32, 7, 7))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 5, 5, border_mode='full', init='glorot_normal')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((64, 199, 199)))
model.add(Convolution2D(64, 64, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(96, 64, 3, 3, border_mode='full', init='glorot_normal')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((96, 69, 69)))
model.add(Convolution2D(96, 96, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 96, 3, 3, border_mode='full', init='glorot_normal')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((128, 36, 36)))
model.add(Convolution2D(128, 128, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(160, 128, 3, 3, border_mode='full', init='glorot_normal')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((256, 20, 20)))
model.add(Convolution2D(160, 160, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(192, 160, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
#model.add(BatchNormalization((512, 12, 12)))
model.add(Convolution2D(192, 192, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(224, 192, 3, 3, border_mode='full')) 
model.add(Activation('relu'))
model.add(BatchNormalization((224, 6, 6)))
model.add(Convolution2D(224, 224, 3, 3)) 
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
#model.add(Dense(25600, 4000))
#model.add(Activation('relu'))
##model.add(BatchNormalization((4000,)))
#model.add(Dropout(0.5))

#model.add(Dense(4000, 2000))
#model.add(Activation('relu'))
#model.add(BatchNormalization((2000,)))
#model.add(Dropout(0.5))

#model.add(Dense(3072, 2000))
#model.add(Activation('relu'))
#model.add(BatchNormalization((2000,)))
#model.add(Dropout(0.5))

#model.add(Dense(2000, 1000))
#model.add(Activation('relu'))
#model.add(BatchNormalization((1000,)))
#model.add(Dropout(0.5))

model.add(Dense(896, 447))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=Adagrad())

#grapher = Grapher()
#grapher.plot(model, "/temp/graph.png")

nb_epoch = 2
batch_size = 300
nb_samples = 30

from kobra.tr_utils import time_now_str
print("Start time: " + time_now_str())

#x_val, y_val = BatchGenerator(train_path, labels_map, batch_size).get_val()

dsl = DataSetLoader(train_path, labels_file, labels_map)
imgen = ImageDataGenerator()
imgen.fit(dsl.X_train)

X_train = dsl.X_train - imgen.mean
X_train = X_train / imgen.std

model.fit(X_train, dsl.Y_train, batch_size=30, nb_epoch=10, validation_split=0.1)

#for e in range(nb_epoch):
#    print("Epoch %d" % e)
#    batches = BatchGenerator(train_path, labels_file, labels_map, batch_size)

#    progbar = generic_utils.Progbar(batches.total)
#    for x_train, y_train in batches:
               
#        iterations = x_train.shape[0] / nb_samples if x_train.shape[0] % nb_samples == 0 else (x_train.shape[0] + nb_samples) / nb_samples
        
#        for i in range(iterations):
#            X_batch = x_train[i * nb_samples : (i + 1) * nb_samples]
#            Y_batch = y_train[i * nb_samples : (i + 1) * nb_samples]
#            loss = model.train_on_batch(X_batch, Y_batch)
#            progbar.add(X_batch.shape[0], values= [("train loss", loss)])

#    loss, accuracy = model.evaluate(x_val, y_val, show_accuracy=True)
#    print ("Validation: loss - {0}, accuracy - {1}".format(loss, accuracy))
         
print("End time: " + time_now_str())

json_string = model.to_json()
open('/users/boris/dropbox/kaggle/whales/models/model_2.json', 'w').write(json_string)
model.save_weights('/users/boris/dropbox/kaggle/whales/models/weights_2.h5', overwrite=True)