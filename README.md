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


* Download the package lists from the repositories

`sudo apt-get update`

* install `virtualenv` and start a virtual environment

`sudo apt-get install python-virtualenv`

* Install `pip3`, the Python 3 package manager

`sudo apt install python3-pip`

* Clone the project

`git clone git@github.com:thomas-legrand/planet.git`

* With `pip3`, install all the necessary packages

`pip3 install -t requirements.txt`

Congratulations, you are all set!

## Usage

`python3 test_service.py --host 127.0.0.1 --port 9000`

## Future work

take input as jpg file

training endpoint

unit testing (For unit testing see: http://flask.pocoo.org/docs/0.12/quickstart/ section accessing request data)

multiple models

scaling issues (mention basic web server) (Deploying? on Googe App Engine)