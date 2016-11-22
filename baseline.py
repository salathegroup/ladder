#!/usr/bin/env python

import numpy as np
import deepdish as dd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


DATA_DIR = "/mount/SDB/paper-data/output-aggregated/"
TRAIN_FILE = DATA_DIR + 'train.h5'
TEST_FILE = DATA_DIR + 'validation.h5'

train = dd.io.load(TRAIN_FILE)
test = dd.io.load(TEST_FILE)

def to_one_hot(labels):
    output = []
    _max = labels.max()
    for _item in labels:
        _temp = np.zeros(_max)
        _temp[_item - 1] = 1
        output.append(_temp)

    return np.array(output)

train_x = train["data"]
test_x = test["data"]
train_y = train['coarse_labels']
test_y = test['coarse_labels']

TRAIN_X = []
TRAIN_Y = []

#Collect 10 labelled images per class
n_labels = 10

for _i in range(38):
    count = 0
    for _idx, _item in enumerate(train_y):
        if _item == _i+1:
            TRAIN_X.append(train_x[_idx])
            TRAIN_Y.append(train_y[_idx])
            count += 1
            if count >= n_labels:
                break

TRAIN_X = np.array(TRAIN_X)

TRAIN_Y = to_one_hot(np.array(TRAIN_Y))
test_y = to_one_hot(test_y)

print TRAIN_X.shape
print TRAIN_Y.shape
print test_x.shape
print test_y.shape

# Initialize the MLP
def initialize_nn(frame_size, class_size):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(1000, init='he_normal', activation='tanh', input_dim=frame_size)) # Dense layer
    model.add(Dense(1000, init='he_normal', activation='tanh')) # Another dense layer
    model.add(Dense(class_size, activation='softmax')) # Last dense layer with Softmax

    # model.compile(loss='mean_squared_error', optimizer="adam")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

class_size = 38
model = initialize_nn(TRAIN_X.shape[1], class_size)
EPOCHS = 100
model.fit(TRAIN_X, TRAIN_Y, nb_epoch=EPOCHS, batch_size=16)
score = model.evaluate(test_x, test_y, batch_size=16)
print ""
print score
score = model.evaluate(TRAIN_X, TRAIN_Y, batch_size=16)
print ""
print score
