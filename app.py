import constants
from flask import Flask, jsonify, request, render_template, abort
import logging
from logging import handlers
import numpy as np
from keras.models import load_model
import tensorflow as tf

# get the default graph for future reference.
# see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
model = load_model(constants.CNN_MODEL_FILENAME)
graph = tf.get_default_graph()
app = Flask(__name__)


def validate_input_data(data):
    try:
        correct_nb_rows = len(data) == constants.IMG_ROWS
        correct_nb_cols = all(len(data[i]) == constants.IMG_COLS for i in range(len(data)))
        all_ints = all(isinstance(pixel, int) for row in data for pixel in row)
        return correct_nb_rows and correct_nb_cols and all_ints
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
        predict_request = np.array(data).reshape(1, constants.FLAT_IMAGE_LENGTH)

        global graph
        with graph.as_default():
            preds = model.predict_classes(predict_request, verbose=0)

        # jsonify is the safe way to generate a JSON file to return
        # see http://flask.pocoo.org/docs/0.10/security/#json-security
        return jsonify({"classification": str(preds[0])})
    except Exception as e:
        app.logger.error("Something went wrong when trying to classify input data. Aborting with 404 error. "
                         "Exception: {0}".format(e))
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
    app.run(port=9000, debug=True)