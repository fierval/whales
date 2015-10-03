import pandas as pd
from os import path
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

test_path = "/Kaggle/whales/test"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"
models_path = "/users/boris/dropbox/kaggle/whales/models"
sample_submission = "/Kaggle/whales/sample_submission.csv"

json_path = path.join(models_path, "model_2.json")
model = model_from_json(open(json_path).read())
model.load_weights(path.join(models_path, "weights_2.h5"))

ss = pd.read_csv(sample_submission)
images = pd.DataFrame(ss['Image'])
X_test = np.array(map(cv2.imread, np.array(map(lambda im_f: path.join(test_path, im_f), images.values)).flatten())).transpose(0, 3, 1, 2)

imgen = ImageDataGenerator()
imgen.fit(X_test)

X_test = X_test - imgen.mean
X_test = X_test / imgen.std

Y_test = model.predict(X_test, batch_size=30, verbose=1)
labels = pd.DataFrame(Y_test, columns=ss.columns[1:])
preds = images.join(labels)

preds.to_csv("/kaggle/whales/submissions/sub3.csv", index=False)
