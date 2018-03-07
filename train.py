import pickle
import glob
import numpy as np
import keras
# import cv2


def unpickle(filename, enc='bytes'):
    with open(filename, 'rb') as fo:
        fdict = pickle.load(fo, encoding=enc)
    return fdict


# Load all data.
training_data = []
training_labels = []
testing_labels = []
# for fname in glob.glob("dataset/cifar-10-batches-py/data_batch_*"):
for fname in glob.glob("dataset/cifar-10-batches-py/*_batch*"):
    _trd = unpickle(fname)
    for x in _trd[b'data']:
        training_data.append(x)
    training_labels += _trd[b'labels']
_ted = unpickle("dataset/cifar-10-batches-py/test_batch")
training_data = np.array(training_data).reshape(
    60000, 3, 1024).swapaxes(1, 2).reshape(60000, 32, 32, 3).astype('float')
testing_data = np.array(_ted[b'data']).reshape(
     10000, 3, 1024).swapaxes(1, 2).reshape(10000, 32, 32, 3).astype('float')
training_data /= 255
testing_data /= 255
training_labels = np.array(training_labels)
training_labels = keras.utils.to_categorical(training_labels, 10)
testing_labels = np.array(_ted[b'labels'])
testing_labels = keras.utils.to_categorical(testing_labels, 10)
# label_text = unpickle("dataset/cifar-10-batches-py/batches.meta", 'utf-8')['label_names']

print(training_data.shape)
# print(training_labels[1833])
# cv2.imshow("T", training_data[1833].reshape(32, 32, 3))
# cv2.waitKey(0)

# Training
model = keras.models.Sequential()

# First convulation + max pooling
model.add(keras.layers.Conv2D(
    128, (8, 8), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convulation + max pooling
model.add(keras.layers.Conv2D(256, (16, 16), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D())

# Hidden layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1500, activation='relu'))

# Output
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_data, training_labels, epochs=12, batch_size=100, verbose=1)
model.save("trained_model.h5")

metrics = model.evaluate(testing_data, testing_labels, batch_size=100)
print("Metrics --\nLoss: {0}\nAccuracy: {1}".format(metrics[0], metrics[1]))
