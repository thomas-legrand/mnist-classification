import numpy as np
from flask import Flask, jsonify, request
from keras.models import load_model
import tensorflow as tf
import models

MLP_MODEL_FILENAME = 'ddmlp.h5'

model = load_model(MLP_MODEL_FILENAME)

# get the default graph for future reference.
# see https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
graph = tf.get_default_graph()


app = Flask(__name__)


@app.route('/mnist/classify', methods=['POST'])
def make_predict():
    # error checking
    data = request.get_json(force=True)
    # convert data to numpy array
    # data to predict
    predict_request = np.array(data)
    global graph
    with graph.as_default():
        preds = model.predict_classes(predict_request.reshape(1, models.FLAT_IMAGE_LENGTH), verbose=0)
    output = {"ddmlp": str(preds[0])}
    # jsonify is the safe way to generate a JSON file to return
    # see http://flask.pocoo.org/docs/0.10/security/#json-security
    return jsonify(output)


if __name__ == "__main__":
    app.run(port=9000, debug=True)