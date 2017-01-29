import argparse
import constants
import os
from flask import Flask, jsonify, request, render_template, abort
import logging
from logging import handlers
import numpy as np
from keras.models import load_model
import cv2
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
                    help='Model filename to use for prediction')

# we need to make graph a global variable
# see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
# similarly we make model global, so we don't have to pass it as argument to the classification function
global graph, model
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def validate_input_data(data):
    """Validation of the input data. Checks nb of rows and columns"""
    try:
        correct_nb_rows = len(data) == constants.IMG_ROWS
        correct_nb_cols = all(len(data[i]) == constants.IMG_COLS for i in range(len(data)))
        return correct_nb_cols and correct_nb_rows
    except TypeError:
        return False


def convert_to_mnist_format(f):
    raw_img = cv2.imdecode(np.asarray(bytearray(f.stream.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.bitwise_not(raw_img)
    (thresh, im_bw) = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(im_bw, (28, 28), interpolation=cv2.INTER_AREA)
    return resized


@app.route('/mnist/classify', methods=['POST'])
def make_predict_image():
    # Get the name of the uploaded file
    f = request.files['image']
    # Check if the file is one of the allowed types/extensions
    if not f and not allowed_file(f.filename):
        app.logger.error("Something went wrong when trying to classify input data. Aborting with 404 error. "
                         "Exception: %s", e)
        abort(404)

    try:
        img = convert_to_mnist_format(f)
    except Exception as e:
        app.logger.error("Failed conversion to Numpy array")
        app.logger.error(e)
        abort(404)
    if not isinstance(img, np.ndarray) or not validate_input_data(img):
        app.logger.error("Failing to validate input data. Aborting with 404 error.")
        abort(404)

    try:
        predict_request = img.reshape(1, constants.IMG_ROWS, constants.IMG_COLS, 1)
        with graph.as_default():
            preds = model.predict_classes(predict_request, verbose=0)

        # jsonify is the safe way to generate a JSON file to return
        # see http://flask.pocoo.org/docs/0.10/security/#json-security
        return jsonify({"classification": str(preds[0])})
    except Exception as e:
        app.logger.error("Something went wrong when trying to classify input data. Aborting with 404 error. "
                             "Exception: %s", e)
        abort(404)





@app.errorhandler(404)
def page_not_found(error):
    """Render neat template when page not found"""
    return render_template('404.html', planet_ascii_art=constants.PLANET_ASCII_ART), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Render neat template when method not allowed"""
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