import numpy as np
from flask import Flask, jsonify, request
import pickle
from keras.models import load_model
import tensorflow as tf

clf2 = pickle.load(open('svm', 'rb'))
model = load_model('ddmlp.h5')
graph = tf.get_default_graph() # https://github.com/fchollet/keras/issues/2397#issuecomment-254919212
app = Flask(__name__)


@app.route('/classify', methods=['POST'])
def make_predict():
    # error checking
    data = request.get_json(force=True)
    # convert data to numpy array
    # data to predict
    predict_request = np.array(data)

    # predict
    y_hat = clf2.predict(predict_request)
    global graph
    with graph.as_default():
        preds = model.predict_classes(predict_request, verbose=0)
    output = {"svm": str(y_hat[0]),
              "ddmlp": str(preds[0])}
    return jsonify(results=output)


if __name__ == "__main__":
    app.run(port=9000, debug=True)