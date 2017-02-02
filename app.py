""" A command line interface to start a Flask service for classifying handwritten digits. """

import argparse
import constants
import cv2
from flask import Flask, jsonify, request, render_template, abort
import io
from keras.models import load_model
import logging
from logging import handlers
import numpy as np
import os
import tensorflow as tf


parser = argparse.ArgumentParser(description='Starts a classification service.')
parser.add_argument('--host',
                    default=constants.DEFAULT_HOST,
                    help='Host for the classification service')
parser.add_argument('--port', type=int,
                    default=constants.DEFAULT_PORT,
                    help='port on which to run the service')
parser.add_argument('--model',
                    default=constants.CNN_MODEL_FILENAME,
                    help='Model filename to use for prediction. Needs to refer to a HDF5 file')

# we need to make graph a global variable
# see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
# similarly we make model global, so we don't have to pass it as argument to the classification function
global graph, model
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

def allowed_file(filename):
    """
    Validate the file extension.
    :param filename: filename to check
    :return: (boolean) whether the file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def validate_img_data(data):
    """
    Validate the processed image data. Checks number of rows and columns.
    :param data: input numpy array
    :return: (boolean) whether the processed image conforms to the expected format
    """
    if not isinstance(data, np.ndarray):
        return False
    try:
        correct_nb_rows = len(data) == constants.IMG_ROWS
        correct_nb_cols = all(len(data[i]) == constants.IMG_COLS for i in range(len(data)))
        return correct_nb_cols and correct_nb_rows
    except TypeError:
        return False


def convert_to_mnist_format(input_file):
    """
    Convert input file to the MNIST format. Uses OpenCV for decoding and transforming the image.
    :param input_file: input file, as received by the server in the request files
    :return: (np.ndarray) processed image as a numpy array
    """
    try:
        in_memory_file = io.BytesIO()
        input_file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        grey_img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        grey_inv_img = cv2.bitwise_not(grey_img)
        (thresh, im_bw) = cv2.threshold(grey_inv_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.resize(im_bw, (28, 28), interpolation=cv2.INTER_AREA)
    except Exception as e:
        app.logger.error("Image processing failed. Exception: %s", e)
        abort(404)
        return
    if not validate_img_data(img):
        app.logger.error("Image format validation failed")
        abort(404)
    return img


@app.route('/mnist/classify', methods=['POST'])
def make_predict_image():
    """
    Hold the main logic of request parsing and response.
    """

    # Get the name of the uploaded file
    if not 'image' in request.files:
        app.logger.error("Please upload an image in the 'image' form member.")
        abort(404)
    input_image_file = request.files['image']

    # Check if the file is one of the allowed types/extensions
    if not input_image_file:
        app.logger.error("File upload error.")
        abort(404)

    if not allowed_file(input_image_file.filename):
        app.logger.error("File upload error. File extension should be png or jpg.")
        abort(404)

    # Convert the file to the mnist format (opencv np array, 28 * 28 pixels)
    img = convert_to_mnist_format(input_image_file)

    try:
        # Attempt to classify the given example
        predict_request = img.reshape(1, constants.IMG_ROWS, constants.IMG_COLS, 1)
        with graph.as_default():
            preds = model.predict_classes(predict_request, verbose=0)

        # jsonify is the safe way to generate a JSON file to return
        # see http://flask.pocoo.org/docs/0.10/security/#json-security
        return jsonify({"classification": str(preds[0])})
    except Exception as e:
        app.logger.error("Model prediction error. Exception: %s", e)
        abort(404)


@app.errorhandler(404)
def page_not_found(error):
    """ Render neat template when page not found """
    return render_template('404.html', planet_ascii_art=constants.PLANET_ASCII_ART), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """ Render neat template when method not allowed """
    return render_template('405.html', planet_ascii_art=constants.PLANET_ASCII_ART), 405


if __name__ == "__main__":
    handler = handlers.RotatingFileHandler('classify.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    args = parser.parse_args()

    try:
        filepath = os.path.join(constants.MODELS_DIR, args.model)
        model = load_model(filepath)
    except OSError as e:
        msg = "Model file could not be found at {0}. Error: {1}".format(args.filename, e)
        logging.error(msg)
        exit(1)
    graph = tf.get_default_graph()
    app.run(host=args.host, port=args.port)
