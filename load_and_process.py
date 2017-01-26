import constants
from keras.datasets import mnist
from keras.utils import np_utils


def load_and_process_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # this assumes that we are using a TensorFlow backend
    X_train = X_train.reshape(X_train.shape[0], constants.IMG_ROWS, constants.IMG_COLS, 1)
    X_test = X_test.reshape(X_test.shape[0], constants.IMG_ROWS, constants.IMG_COLS, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, constants.NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, constants.NB_CLASSES)
    return X_train, X_test, Y_train, Y_test