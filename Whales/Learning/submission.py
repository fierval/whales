import numpy as np
import pandas as pd
import os
from os import path
import matplotlib.pylab as plt
import cv2

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
#from keras.utils.dot_utils import Grapher
from keras.utils import generic_utils

train_path = "/Kaggle/whales/kerano"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"
models_path = "/kaggle/whales/models"
sample_submission = "/Kaggle/whales/sample_submission.csv"

json_path = path.join(models_path, "model_1.json")
model = model_from_json(open(json_path).read())
model.load_weights(path.join(models_path, "weights_1.h5"))

ss = pd.read_csv(sample_submission)
ims = map(cv2.imread, ss)
images = pd.DataFrame(ss['Images'])

y_train = model.predict(ims)
labels = pd.DataFrame(y_train, columns=ss.columns[1:])
preds = images.join(labels)

preds.to_csv("/kaggle/whales/submissions/sub1.csv")



