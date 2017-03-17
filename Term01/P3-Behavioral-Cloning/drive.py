import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#Pre-processing Images

# Re-size images down to a quarter of original size, to speed up training
def resize(img):
    img = img.resize((80, 40), Image.ANTIALIAS)
    return img

#Cutting the image to the section, that holds the road information
def cut_top_portion_of_images(image):
    array_Image = np.array(image)
    array_Cut = array_Image[15:]
    return array_Cut

#Converting the RGB Image to an HLS Image
def convert_to_HLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

#Normalizing the input Image
def normalize(image_data):
    max = 255. #np.max(img)
    return (((image_data) / max) - 0.5)


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]

    image = Image.open(BytesIO(base64.b64decode(imgString)))

    image_array = resize(image)
    
    image_array = np.asarray(image_array)
    print(image_array.shape)

    #Preprocess data in the same way as TrainingData
    image_array = cut_top_portion_of_images(image_array)
    image_array = convert_to_HLS(image_array)
    image_array = normalize(image_array)

    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))
        #model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
