import numpy as np
from flask import Flask, jsonify, request, render_template, abort
from keras.models import load_model
import tensorflow as tf
import constants

model = load_model(constants.CNN_MODEL_FILENAME)

# get the default graph for future reference.
# see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
graph = tf.get_default_graph()


app = Flask(__name__)


@app.route('/mnist/classify', methods=['POST'])
def make_predict():
    # error checking
    if False:
        abort(404)

    data = request.get_json(force=True)
    # convert data to numpy array
    predict_request = np.array(data).reshape(1, constants.FLAT_IMAGE_LENGTH)

    global graph
    with graph.as_default():
        preds = model.predict_classes(predict_request, verbose=0)

    # jsonify is the safe way to generate a JSON file to return
    # see http://flask.pocoo.org/docs/0.10/security/#json-security
    return jsonify({"classification": str(preds[0])})


@app.errorhandler(404)
def page_not_found():
    return render_template('404.html', planet_ascii_art=constants.PLANET_ASCII_ART), 404


if __name__ == "__main__":
    app.run(port=9000, debug=True)