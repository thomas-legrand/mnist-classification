# Take-home challenge

## Instructions

Create a service using a python web framework such as flask or django that takes an mnist image as a post request and 
returns a JSON blob with the classification of that image. http://yann.lecun.com/exdb/mnist/
```
POST /mnist/classify
    Returns the class of the image. Invalid input should return a 404.
```

## Installation

These installation steps assume that you are working on an Ubuntu 64 bits machine.
If working on a Windows or Mac machine, please look for the corresponding installation steps online.
An easy way to get such a setup is to launch an Amazon EC2 instance with this 
[tutorial](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance_linux).

Prerequisites: we suppose that `git` and `python3` are already installed.

* Download the package lists from the repositories

`sudo apt-get update`

* install `virtualenv` and start a virtual environment

`sudo apt-get install python-virtualenv`

* Install `pip3`, the Python 3 package manager

`sudo apt install python3-pip`

* Clone the project

`git clone git@github.com:thomas-legrand/mnist-classification.git`

* Create a virtual environment and activate it

```
cd mnist-classification
virtualenv flask-aws
source flask-aws/bin/activate
```

* With `pip3`, install all the necessary packages

`pip3 install -r requirements.txt`

Congratulations, you are all set!

## Usage

### Starting the service

Starting the service is as simple as running: `python3 app.py --host 0.0.0.0 --port 5000`.
The service host is now referred as the server.

Make sure that the server is set up to accept inbound HTTP traffic from any IP. 
On Amazon EC2, you can do this by updating the security group assigned to your VM.
If running the service locally, please use `--host 127.0.0.1`.

### Making a request to the server

The service you just set up accepts only `jpg` or `png` images through POST requests.

To request a classification from the server, you need an input image, representing a (preferably handwritten) digit.
Feel fre to use your own images or the ones in the two small test sets provided (`test-data-0` and `test-data-1`).
 
Requests can be performed in two ways:

1. directly from a python console:  

```
import requests
requests.post('http://127.0.0.1:5000/mnist/classify', files = {'image': open('test-digits-0/200.jpg', 'rb')}).json()
```
2. using a custom script provided: `test_service.py`:

```python3 test_service.py --host 127.0.0.1 --port 5000 --image test-digits-0/200.jpg```

The `host` IP should be replaced by the Public IP of your server the service.

### Underlying classification model

A classification model is provided in the `models/` folder.
It is a CNN, trained on 10 epoch with a default batch size.
It was trained using the command below and showed 98.69% accuracy on the test set.

```python3 train_model.py --filename new_cnn_model.h5 --epoch 10```

Feel free to use it for "of the shelf" classifications.

The model can be re-trained in the exact same way. 
Make sure you provide the filename to save it.

```python3 train_model.py --filename new_cnn_model.h5 --epoch 20 --batchsize 256```

## Implementation

### Service

The service was implemented using [Flask](http://flask.pocoo.org/), a lightweight Python web framework.


Features:
 - Fast loading of a trained CNN model, using the Keras function `load_model`
 - Checking provided file extension
 - Handling of image processing with `OpenCV`
 - Format validation of data to be fed to the model 
 - Special error handling was implemented for 404 (page not found) and 405 (method not allowed), since only POST 
 requests are allowed
on the API endpoint (`mnist/classify`), with an ASCII art piece representing the Earth.
 

### Data

1. Training data

The MNIST dataset, provided with the `keras` package was used to train our classifer.
60,000 training examples were used to fit the model and 10,000 to assess/report its performance.
Training and testing examples are tuples of images pixels (28 * 28 array of pixels, int values ranging from 0 to 255 
where 0 corresponds to white.)

2. Test data

We provide two small sets of testing data, in the form of raw images.
Please note that to better test real life applications of the algorithm, I chose a set of images other than MNIST, 
representing better what people might use the service for.
- `test-digits-0`: digit images extracted from the web. Mix of handwritten and machine generated digits.
- `test-digits-1`: handwritten digit images created by myself

While the algorithm performs quite poorly on the first set, it does a decent job with the second one.


### Image processing

When provided with a raw `jpg` or `png` image, we used an `OpenCV` API for Python to transform it to the correct format.
Steps:

1. Decode image from the file byte stream, as a grayscale image

2. Invert the image, to conform to MNIST standards (0 = white, 255 = black)

3. Since the image is bimodal, use Otsu's binarization to choose the adequate threshold to binarize the image. 

4. Finally resize the image to 28 * 28 pixels.

### Model training and saving

We train a Convolutional Neural Network using Keras.
We use a simpler version of the CNN provided by F. Chollet on the 
[Keras Github repository](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py).

It is a sequential model, containing a 2D Convolutional layer, followed by three Dense layers of decreasing sizes 
(256, 128 and 10 neurons).
We include some Dropout layer to limit overfitting and finally use a softmax activation for the last layer that we can 
then use as the classification.



## Repository organization

- `README.md`: this README file
- `app.py`: launch a Flask app
- `train_model.py`: train a CNN on the MNIST data
- `load_and_process.py`: load the MNIST dataset and pre-process it
- `test_service.py`: issue test requests to a given endpoint
- `test-data-0/`: first set of test images, extracted from the web
- `test-data-1/`: second set of test images, handwritten by myself
- `templates/`: error pages templates
- `models/`: directory to save folders
- `requirements.txt`: requirements for the package


## Future work

I have not had the time to work on the following features, which would be nice to have.
They range from simple to advanced.

 - provide a training endpoint, where users can train the model
 - unit testing (for Flask see http://flask.pocoo.org/docs/0.12/quickstart/#accessing-request-data)
 - provide multiple classfication models
 - we are using Flask builtin web server. This is good enough for testing but should not be used in production.