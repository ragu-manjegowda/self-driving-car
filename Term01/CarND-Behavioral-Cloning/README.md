# Behaviour Cloning

This is the third project in Udacity Self-driving car nano degree program.

Aim of the project is to capture the training data set by running the Simulator in training mode. Later train it such way it steers by itself in Autonomous mode.

<p align="center"><img src="Simulator.png" width="700" height="400" /></p>

I have trained the car to run autonomously in first track (track on the left).

## Download simulator from here:

>[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587527cb_udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-development-linux-desktop-64-bit-5.zip) 

>[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/587525b2_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-mac-desktop-universal-5.zip)

>[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58752736_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4.zip)

## Usage:

> ### To Train the model  

```shell
$ python model.py
```

> ### To drive in autonomous mode  

```shell
$ python drive.py model.json
```

Now launch the simulator in autonomous mode, drive.py provides steering angle based on the image captured by the car. Throttle is provided by the simulator.

## Explaining the scripts:

There are **2** major scripts: **model.py**, and **drive.py**. **model.ipynb** is same as model.py in jupyter notebook.

> ### model.py  

   This python script imports the raw image data that I captured by running the simulator in training mode and preprocesses them. I only use the picture from the center camera, then resize it. Then I cut upper part of the picture, for this, I cut the first 15 rows of the picture so that only road is seen. Then I convert the picture to HSL, since colors are not required for detecting lanes (recall project 1). At last, I normalize the picture.

   The resulting **array of pictures** is saved as **features** and the **steering angels** as **labels**. Then, I split the data into **train** and **validation** using test train split function with 80% train data.  
   Next the model gets defined and trained by the data using model that I built. After training, the model and weights are saved as **model.json** and **model.h5** respectively.

> ### drive.py  

   This is the python script that receives the images from simulator and predicts the steering angle using the deep learning model saved as model.json, and feeds it to simulator.  
   For treating the incoming pictures the same way as the model was trained. I have done the same preprocessing inside the drive.py as well.  
   Then based on the images captured from **center camera**, **model.json** is used used to predict the steering angle. Model uses weights stored in **model.h5** for prediction.  

## Generating Training Data

I was using only keyboard inputs which makes the data a little bit bias. But I used the smoothening of the network to get rid of this steering wheel jumps. For generating Training Data I drove the race circle 2 times in both directions and added some fallback situations when getting to close to the edges.  

## Preprocessing

The Images get loaded from the local drive and preprocessed by the following functions:

### This is how image looks when loaded before any preprocessing.

<p align="center"><img src="01.png" width="160" height="80" /></p>

### 1. Re-size the image (I don't want to feed large image to my model)

```python
# Re-size images down to a quarter of original size, to speed up training
def resize(img):
    img = img.resize((80, 40), Image.ANTIALIAS)
    return img
```

<p align="center"><img src="02.png" width="160" height="80" /></p>

### 2. Cut the top portion of the image so that we get only road (To how many rows to be cut was figured out after many trials)

```python
def cut_top_portion_of_images(image):
    array_Image = np.array(image)
    array_Cut = array_Image[15:]
    return array_Cut
```

<p align="center"><img src="03.png" width="160" height="80" /></p>

### 3. Convert to HLS, I don't want color to introduce error in the model, also I want to retain S layer so RGB to HLS instead of Grayscale.

```python
#Converting the RGB Image to an HLS Image
def convert_to_HLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls
```

<p align="center"><img src="04.png" width="160" height="80" /></p>

### 3. Normalize image to reduce error rate. (Not normalizing between 0 and 1 to reduce 0 steer angles)

```python
#Normalizing the input Image
def normalize(image_data):
    max = 255. #np.max(img)
    return (((image_data) / max) - 0.5)
```
<p align="center"><img src="05.png" width="160" height="80" /></p>

## Training Model

   For Training I used **80%** of the the images to train the model. And **20%** for validation purposes. Test was always performed in the Simulator. My Batch-Size is 100 images. The network ran 15 epochs. And finally it gave me a result of **one** steering wheel value.

   Those paramters were tuned on performance of the validation set during training. I found for a small batch size and a bigger epoch size my network performed better.

   For the number of channels in the convolutional layers, I found, that using less than 50 channels in my training data set forces the model to underfit. Using more than 100 overfits the model. For Training I used an ADAM Optimizer. 
   
   Regular normalization was not giving me good results so I used Keras batch normalization to my network architecture. This is the first layer of my sequential model (before an convolutional layers).
   
   From here, I deepened the network a bit to include more layers. Finally I ended up with four convolutional layers followed by four fully connected layers. Note that the final fully connected layer needs to have an output of one, since it needs just one steering angle output. This is the architecture that I found to arrive at best training and validation scores.
   
   The convolutional layers have filters totalling 64, 32, 16, and 8 in descending layers. The fully connected layers have output sizes of 128, 64, 32, and 1. I used mean squared error, as the problem of steering angle is regression problem.


### Following the architecture of my model:

      Layer (type)                     Output Shape          Param #     Connected to                     
      ====================================================================================================
      batchnormalization_1 (BatchNorma (None, 25, 80, 3)     12          batchnormalization_input_1[0][0] 
      ____________________________________________________________________________________________________
      convolution2d_1 (Convolution2D)  (None, 23, 78, 64)    1792        batchnormalization_1[0][0]       
      ____________________________________________________________________________________________________
      activation_1 (Activation)        (None, 23, 78, 64)    0           convolution2d_1[0][0]            
      ____________________________________________________________________________________________________
      dropout_1 (Dropout)              (None, 23, 78, 64)    0           activation_1[0][0]               
      ____________________________________________________________________________________________________
      convolution2d_2 (Convolution2D)  (None, 21, 76, 32)    18464       dropout_1[0][0]                  
      ____________________________________________________________________________________________________
      activation_2 (Activation)        (None, 21, 76, 32)    0           convolution2d_2[0][0]            
      ____________________________________________________________________________________________________
      convolution2d_3 (Convolution2D)  (None, 19, 74, 16)    4624        activation_2[0][0]               
      ____________________________________________________________________________________________________
      activation_3 (Activation)        (None, 19, 74, 16)    0           convolution2d_3[0][0]            
      ____________________________________________________________________________________________________
      convolution2d_4 (Convolution2D)  (None, 17, 72, 8)     1160        activation_3[0][0]               
      ____________________________________________________________________________________________________
      activation_4 (Activation)        (None, 17, 72, 8)     0           convolution2d_4[0][0]            
      ____________________________________________________________________________________________________
      maxpooling2d_1 (MaxPooling2D)    (None, 8, 36, 8)      0           activation_4[0][0]               
      ____________________________________________________________________________________________________
      flatten_1 (Flatten)              (None, 2304)          0           maxpooling2d_1[0][0]             
      ____________________________________________________________________________________________________
      dropout_2 (Dropout)              (None, 2304)          0           flatten_1[0][0]                  
      ____________________________________________________________________________________________________
      dense_1 (Dense)                  (None, 128)           295040      dropout_2[0][0]                  
      ____________________________________________________________________________________________________
      activation_5 (Activation)        (None, 128)           0           dense_1[0][0]                    
      ____________________________________________________________________________________________________
      dropout_3 (Dropout)              (None, 128)           0           activation_5[0][0]               
      ____________________________________________________________________________________________________
      dense_2 (Dense)                  (None, 64)            8256        dropout_3[0][0]                  
      ____________________________________________________________________________________________________
      activation_6 (Activation)        (None, 64)            0           dense_2[0][0]                    
      ____________________________________________________________________________________________________
      dense_3 (Dense)                  (None, 32)            2080        activation_6[0][0]               
      ____________________________________________________________________________________________________
      activation_7 (Activation)        (None, 32)            0           dense_3[0][0]                    
      ____________________________________________________________________________________________________
      dense_4 (Dense)                  (None, 1)             33          activation_7[0][0]               
      ====================================================================================================
