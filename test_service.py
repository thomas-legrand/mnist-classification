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
X_test = X_test.reshape(X_test.shape[0], constants.IMG_ROWS, constants.IMG_COLS, 1)
X_test = X_test.astype('float32')
X_test /= 255


def get_service_url(host, port, path=constants.PATH):
    """Construct a service URL from host, port and path"""
    return "".join(["http://", host, ":", str(port), path])


def make_sample_request(url, filename):
    """Make a sample request to the service"""
    logging.info("Making POST request to url: %s", url)
    r = requests.post(url, files={'image': open(filename, 'rb')})
    logging.info("Status code is %d", r.status_code)
    logging.info("Classification service predicted: %s", r.json()["classification"])


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    url = get_service_url(host=args.host, port=args.port)
    make_sample_request(url)

if __name__ == '__main__':
    main()