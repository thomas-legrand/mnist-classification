import json
from keras.datasets import mnist
import requests
import models

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(10000, models.FLAT_IMAGE_LENGTH)
X_test = X_test.astype('float32')
X_test /= 255
data = X_test[1].tolist()
url = "http://localhost:9000/mnist/classify"


a = json.dumps(data)

r = requests.post(url, a)
print(r.json())