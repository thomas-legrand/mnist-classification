import constants
import argparse
import logging
import load_and_process
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

parser = argparse.ArgumentParser(description='Train a classification model.')
parser.add_argument('--filename',
                    default=constants.CNN_MODEL_FILENAME,
                    help='Filename to save the model to')
parser.add_argument('--epoch', type=int,
                    default=constants.NB_EPOCH,
                    help='Number of epoch to train the model for')
parser.add_argument('--batchsize', type=int,
                    default=constants.BATCH_SIZE,
                    help='Size of batches to use for training')


def train_cnn(data, model_name, nb_epoch, batch_size):
    X_train, X_test, Y_train, Y_test = data

    np.random.seed(constants.RANDOM_STATE)  # for reproducibility
    input_shape = (constants.IMG_ROWS, constants.IMG_COLS, 1)

    model = Sequential()
    model.add(Convolution2D(constants.NB_FILTERS, constants.KERNEL_SIZE[0], constants.KERNEL_SIZE[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=constants.POOL_SIZE))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

    filepath = os.path.join(constants.MODELS_DIR, model_name)
    model.save(filepath)
    logging.info("Successfully saved new model at %s", filepath)


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    data = load_and_process.load_and_process_mnist_data()
    train_cnn(data, args.filename, args.epoch, args.batchsize)


if __name__ == '__main__':
    main()