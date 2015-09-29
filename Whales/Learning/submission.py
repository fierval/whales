import pandas as pd
from os import path
import cv2
import numpy as np

from keras.models import model_from_json

test_path = "/Kaggle/whales/test"
labels_file = "/Kaggle/whales/train.csv"
labels_map = "/Kaggle/whales/labels_map.csv"
models_path = "/users/boris/dropbox/kaggle/whales/models"
sample_submission = "/Kaggle/whales/sample_submission.csv"

json_path = path.join(models_path, "model_1.json")
model = model_from_json(open(json_path).read())
model.load_weights(path.join(models_path, "weights_1.h5"))

ss = pd.read_csv(sample_submission)
images = pd.DataFrame(ss['Image'])
ims = np.array(map(cv2.imread, np.array(map(lambda im_f: path.join(test_path, im_f), images.values)).flatten())).transpose(0, 3, 1, 2)

y_train = model.predict(ims)
labels = pd.DataFrame(y_train, columns=ss.columns[1:])
preds = images.join(labels)

preds.to_csv("/kaggle/whales/submissions/sub1.csv", index=False)
