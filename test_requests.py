import requests
import json
from sklearn import datasets
digits = datasets.load_digits()
data = json.dumps(digits.data[-1:].tolist())
data.shape

url = "http://localhost:9000/classify"


a = json.dumps(data)

r = requests.post(url, a)
print(r.json())