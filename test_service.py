import random
import json
from keras.datasets import mnist
import requests
import constants
import argparse
import logging

parser = argparse.ArgumentParser(description='Make a test request.')
parser.add_argument('--host',
                    default=constants.DEFAULT_HOST,
                    help='The host of the classification service')
parser.add_argument('--port', type=int,
                    default=constants.DEFAULT_PORT,
                    help='The port to make requests to the service')


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(constants.NB_TEST_EXAMPLES, constants.FLAT_IMAGE_LENGTH)
X_test = X_test.astype('float32')
X_test /= 255


def get_service_url(host, port, path=constants.PATH):
    return "".join(["http://", host, ":", str(port), path])


def make_sample_request(url):
    sample = random.randint(0, constants.NB_TEST_EXAMPLES)
    data = X_test[sample].tolist()
    a = json.dumps(data)
    logging.info("Making POST request to url: {}".format(url))
    r = requests.post(url, a)
    logging.info("Status code is {}".format(r.status_code))
    logging.info("Classification service predicted: {0}, target: {1}".format(r.json()["classification"], y_test[sample]))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    url = get_service_url(host=args.host, port=args.port)
    make_sample_request(url)

if __name__ == '__main__':
    main()