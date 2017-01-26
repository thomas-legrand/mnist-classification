import argparse
import constants
import os
from flask import Flask, jsonify, request, render_template, abort
import logging
from logging import handlers
import numpy as np
from keras.models import load_model
import tensorflow as tf


parser = argparse.ArgumentParser(description='Starts a classification service.')
parser.add_argument('--host',
                    default=constants.DEFAULT_HOST,
                    help='Host for the classification service')
parser.add_argument('--port', type=int,
                    default=constants.DEFAULT_PORT,
                    help='port on which to run the service')
parser.add_argument('--filename',
                    default=constants.CNN_MODEL_FILENAME,
                    help='Model filename to use for prediction')
global graph, model
app = Flask(__name__)


def validate_input_data(data):
    try:
        correct_nb_rows = len(data) == constants.IMG_ROWS
        correct_nb_cols = all(len(data[i]) == constants.IMG_COLS for i in range(len(data)))
        return correct_nb_cols and correct_nb_rows
    except TypeError:
        return False


@app.route('/mnist/classify', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)

    # validate that the data is in the expected format.
    if not data or not validate_input_data(data):
        app.logger.error("Failing to validate input data. Aborting with 404 error.")
        abort(404)

    try:
        # convert data to numpy array
        predict_request = np.array(data).reshape(1, constants.IMG_ROWS, constants.IMG_COLS, 1)
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
    return render_template('404.html', planet_ascii_art=constants.PLANET_ASCII_ART), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return render_template('405.html', planet_ascii_art=constants.PLANET_ASCII_ART), 405


if __name__ == "__main__":
    handler = handlers.RotatingFileHandler('classify.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    args = parser.parse_args()
    # get the default graph for future reference.
    # see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
    try:
        filepath = os.path.join(constants.MODELS_DIR, args.filename)
        model = load_model(filepath)
    except OSError as e:
        msg = "Model file could not be found at {0}. Error: {1}".format(args.filename, e)
        logging.error(msg)
        exit(1)
    graph = tf.get_default_graph()
    app.run(host=args.host, port=args.port, debug=True)