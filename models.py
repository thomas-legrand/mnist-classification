from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import app
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

FLAT_IMAGE_LENGTH = 784

batch_size = 128
nb_classes = 10
nb_epoch = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, FLAT_IMAGE_LENGTH)
X_test = X_test.reshape(10000, FLAT_IMAGE_LENGTH)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(FLAT_IMAGE_LENGTH,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

model.save(app.MLP_MODEL_FILENAME)
model = load_model(app.MLP_MODEL_FILENAME)
preds = model.predict_classes(X_test, verbose=0)

data = X_test[1].tolist()
predict_request = np.array(data, ndmin=2)