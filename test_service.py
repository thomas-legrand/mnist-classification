import random
import json
from keras.datasets import mnist
import requests
import constants

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(constants.NB_TEST_EXAMPLES, constants.FLAT_IMAGE_LENGTH)
X_test = X_test.astype('float32')
X_test /= 255


def get_service_url(host=constants.DEFAULT_HOST, port=constants.DEFAULT_PORT, path=constants.PATH):
    return "".join(["http://", host, ":", str(port), path])


def make_sample_request():
    sample = random.randint(0, constants.NB_TEST_EXAMPLES)
    data = X_test[sample].tolist()
    a = json.dumps(data)
    r = requests.post(get_service_url(), a)
    print(r)
    print(r.json())
    print(y_test[sample])