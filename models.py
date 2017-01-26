from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.utils import np_utils
import numpy as np
import pickle
from sklearn import svm, datasets, model_selection


# load digits data
digits = datasets.load_digits()

# train a SVM classifier on all but one data sample
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1])

# serialize the model
pickle.dump(clf, open('svm', 'wb'))

# Read data
train = digits.data
target = digits.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(train, target, test_size=.3)

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, show_accuracy=True, verbose=2)
model.save('ddmlp.h5')
model = load_model('ddmlp.h5')
preds = model.predict_classes(X_test, verbose=0)