import pickle
import sys
import os
import cv2
import numpy as np
from keras.models import load_model

USAGE = """
USAGE:
    ./get_data.sh
    pip install -r requirements.txt
    python train.py # This took ~2 hours on my laptop with a NVIDIA GT 940M GPU.
    python predict.py /path/to/sample/image.png
"""
TRAINED_MODEL_FILE = "trained_model.h5"
with open("dataset/cifar-10-batches-py/batches.meta", 'rb') as f:
    LABELS = pickle.load(f)['label_names']

try:
    filepath = sys.argv[1]
except IndexError:
    print(USAGE)
    exit(0)

if not os.path.isfile(filepath):
    print("Not a valid file.")
    exit(1)

img = cv2.imread(filepath)
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
img = np.expand_dims(img, axis=0)
print(img.shape)
# cv2.imshow("T", img[0])
# cv2.waitKey(0)
img = img.astype('float')
img /= 255
model = load_model(TRAINED_MODEL_FILE)
prediction = model.predict(img, batch_size=1)
prediction = np.argmax(prediction[0])
print("The image is predicted to be:", LABELS[prediction])
