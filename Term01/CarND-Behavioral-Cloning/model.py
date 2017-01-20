import os
import sys
import csv
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from scipy.misc import toimage


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

#Reading the driving log to match stearing information to Images
with open('./driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    driving_list = list(reader)
    
X_train = []
y_train = []


#Preprocess all Images with cut/convert to HLS/Normalize
for i, row in enumerate(driving_list):
    if i == 0:
        continue
    groups = row[0].split('/')
    image = Image.open("./IMG/" + groups[-1])
    image = resize(image)
    image = cut_top_portion_of_images(image)
    image = convert_to_HLS(image)
    image = normalize(image)
    X_train.append(image)
    y_train.append(row[3])

X_train = np.array(X_train)

#shuffle and split Training Data into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2)

print("Loaded training and validation data")

batch_size = 100
nb_epoch = 15
pool_size = (2, 2)

X_train = X_train.astype('float32')
X_test = X_val.astype('float32')

print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

input_shape = X_train.shape[1:]

model = Sequential()

# Normalize data
model.add(BatchNormalization(input_shape=input_shape))

# Convolutional Layer 1 and Dropout
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 3
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 4
model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# Fully Connected Layer 1 and Dropout
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(64))
model.add(Activation('relu'))

# FC Layer 3
model.add(Dense(32))
model.add(Activation('relu'))

# Final FC Layer - just one output - steering angle
model.add(Dense(1))

# Compiling and training the model
model.compile(metrics=['mean_squared_error'], optimizer='Nadam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json.dump(model_json, json_file)

model.save_weights('./model.h5')

# Show summary of model
model.summary()
