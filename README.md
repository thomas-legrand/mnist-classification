# Take-home challenge

## Instructions

Create a service using a python web framework such as flask or django that takes an mnist image as a post request and returns a JSON blob with the classification of that image. http://yann.lecun.com/exdb/mnist/
```
POST /mnist/classify
    Returns the class of the image. Invalid input should return a 404.
```

## Installation

These installation steps assume that you are working on an Ubuntu 64 bits machine.
If working on Windows or Mac, please look for the corresponding installation steps online.
An easy way to get such a setup is to launch an Amazon EC2 instance with this [tutorial](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance_linux).

Prerequisites: git and python3

* Download the package lists from the repositories

`sudo apt-get update`

* install `virtualenv` and start a virtual environment

`sudo apt-get install python-virtualenv`

* Install `pip3`, the Python 3 package manager

`sudo apt install python3-pip`

* Clone the project

`git clone git@github.com:thomas-legrand/planet.git`

* Create a virtual environment and activate it

`virtualenv flask-aws`
`source flask-aws/bin/activate`

* With `pip3`, install all the necessary packages

`pip3 install -t requirements.txt`

Congratulations, you are all set!

## Usage

### Starting the service

Starting the service is as simply as `python3 app.py --host 0.0.0.0--port 9000`.

Make sure that the machine is set up to accept inbound HTTP traffic from any IP. 
On Amazon EC2, you can do this by updating the security group assigned to your VM.

### Making a test request

You can make a test request to the service using the command line tool `test_service.py`:

```python3 test_service.py --host 35.166.130.221 --port 9000```

The `host` IP should be replaced by the Public IP of the machine/VM running the service.

### Underlying classification model

A default classification model is provided in the `models/` folder.
It is a simple CNN, trained on 10 epoch with a default batch size.
It was trained using the command below and showed 98.69% accuracy on the test set.

```python3 train_model.py --filename cnn_model.h5 --epoch 10```

The model can be re-trained in the exact same way. 
Make sure you provide a filename to save it.

## Future work

I have not had the time to work on the following features, which would be nice to have.
They range from simple to advanced.

 - take any image file as input (jpg, png, etc...) through the POST request
 - provide a training endpoint, where users can train the model
 - unit testing (for Flask see http://flask.pocoo.org/docs/0.12/quickstart/#accessing-request-data)
 - provide multiple classfication models
 - we are using Flask builtin web server. This is good enough for testing but could not be used in production.