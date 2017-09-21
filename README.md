# [In-Depth on Udacity’s Self-Driving Car Curriculum](https://www.udacity.com/drive)

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=f9z-eKD2xjk
" target="_blank"><img src="http://img.youtube.com/vi/f9z-eKD2xjk/0.jpg" 
alt="Click to view Introduction Video" width="480" height="360" border="10" /></a></p>

## Term 1: Computer Vision and Deep Learning


### Introduction

> **1. Meet the instructors**  
Learn about the systems that comprise a self-driving car, and the structure of the program.

> **_2. Project: Detect Lane Lines_**   
**Detect highway lane lines from a video stream. Use OpenCV image analysis techniques to identify lines, including Hough transforms and   Canny edge detection.**

### Deep Learning

> **1. Machine Learning:**  
Review fundamentals of machine learning, including regression and classification.

> **2. Neural Networks:**  
Learn about perceptrons, activation functions, and basic neural networks. Implement your own neural network in Python.

> **3. Logistic Classifier:**  
Study how to train a logistic classifier, using machine learning. Implement a logistic classifier in TensorFlow.

> **4. Optimization:**  
Investigate techniques for optimizing classifier performance, including validation and test sets, gradient descent, momentum, and learning rates.

> **5. Rectified Linear Units:**  
Evaluate activation functions and how they affect performance.

> **6. Regularization:**  
Learn techniques, including dropout, to avoid overfitting a network to the training data.

> **7. Convolutional Neural Networks:**  
Study the building blocks of convolutional neural networks, including filters, stride, and pooling.

> **_8.Project: Traffic Sign Classification_**  
**Implement and train a convolutional neural network to classify traffic signs. Use validation sets, pooling, and dropout to choose a network architecture and improve performance.**

> **9. Keras:**  
Build a multi-layer convolutional network in Keras. Compare the simplicity of Keras to the flexibility of TensorFlow.

> **10. Transfer Learning:**  
Finetune pre-trained networks to solve your own problems. Study cannonical networks such as AlexNet, VGG, GoogLeNet, and ResNet.

> **_11. Project: Behavioral Cloning_**  
**Architect and train a deep neural network to drive a car in a simulator. Collect your own training data and use it to clone your own driving behavior on a test track.**


### Computer Vision

> **1. Cameras:**  
  Learn the physics of cameras, and how to calibrate, undistort, and transform image perspectives.

> **2. Lane Finding:**  
Study advanced techniques for lane detection with curved roads, adverse weather, and varied lighting.

> **_3. Project: Advanced Lane Detection_**  
**Detect lane lines in a variety of conditions, including changing road surfaces, curved roads, and variable lighting. Use OpenCV to implement camera calibration and transforms, as well as filters, polynomial fits, and splines.**

> **4. Support Vector Machines:**  
Implement support vector machines and apply them to image classification.

> **5. Decision Trees:**  
Implement decision trees and apply them to image classification.

> **6. Histogram of Oriented Gradients:**  
Implement histogram of oriented gradients and apply it to image classification.

> **7. Deep Neural Networks:**  
Compare the classification performance of support vector machines, decision trees, histogram of oriented gradients, and deep neural networks.

> **8. Vehicle Tracking:**  
Review how to apply image classification techniques to vehicle tracking, along with basic filters to integrate vehicle position over time.

>**_9. Project: Vehicle Tracking_**  
**Track vehicles in camera images using image classifiers such as SVMs, decision trees, HOG, and DNNs. Apply filters to fuse position data.**


## Term 2: Sensor Fusion, Localization, and Control


### Sensor Fusion
<p align="center"><a target="_blank"><img src="https://cdn-images-1.medium.com/max/800/1*cucK1uncJXoiVYAEvJQd1Q.png" 
width="480" height="360" border="10" /></a></p>

> **1. Sensors:**  
The first lesson of the Sensor Fusion Module covers the physics of two of the most import sensors on an autonomous vehicle — radar and lidar.

> **2. Kalman Filters:**  
Kalman filters are the key mathematical tool for fusing together data. Implement these filters in Python to combine measurements from a single sensor over time.

> **3. C++ Primer:**  
Review the key C++ concepts for implementing the Term 2 projects.

> **_4. Project: Extended Kalman Filters in C++:_**  
**Extended Kalman filters are used by autonomous vehicle engineers to combine measurements from multiple sensors into a non-linear model. Building an EKF is an impressive skill to show an employer.**

> **5. Unscented Kalman Filter:**  
The Unscented Kalman filter is a mathematically-sophisticated approach for combining sensor data. The UKF performs better than the EKF in many situations. This is the type of project sensor fusion engineers have to build for real self-driving cars.

> **_6. Project: Pedestrian Tracking:_**  
**Fuse noisy lidar and radar data together to track a pedestrian.**

### Localization
<p align="center"><a target="_blank"><img src="https://cdn-images-1.medium.com/max/800/1*I1Tly02tvTOUWrs2COkBFw.png" 
width="480" height="360" border="10" /></a></p>

> **1. Motion:**  
Study how motion and probability affect your belief about where you are in the world.

> **2. Markov Localization:**  
Use a Bayesian filter to localize the vehicle in a simplified environment.

> **3. Egomotion:**  
Learn basic models for vehicle movements, including the bicycle model. Estimate the position of the car over time given different sensor data.

> **4. Particle Filter:**  
Use a probabilistic sampling technique known as a particle filter to localize the vehicle in a complex environment.

> **5. High-Performance Particle Filter:**  
Implement a particle filter in C++.

> **_6. Project: Kidnapped Vehicle_**  
**Implement a particle filter to take real-world data and localize a lost vehicle.**

### Control
<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=YMzXAWNDyJg
" target="_blank"><img src="http://img.youtube.com/vi/YMzXAWNDyJg/0.jpg" 
alt="Click to view Introduction Video" width="480" height="360" border="10" /></a></p>

> **1. Control:**  
Learn how control systems actuate a vehicle to move it on a path.

> **2. PID Control:**  
Implement the classic closed-loop controller — a proportional-integral-derivative control system.

> **3. Linear Quadratic Regulator:**  
Implement a more sophisticated control algorithm for stabilizing the vehicle in a noisy environment.

> **_4. Project: Lane-Keeping_**  
**Implement a controller to keep a simulated vehicle in its lane. For an extra challenge, use computer vision techniques to identify the lane lines and estimate the cross-track error.**


## Term 3: Path Planning, Semantic Segmentation, and Systems


### Path Planning

<p align="center"><a target="_blank"><img src="http://img.youtube.com/vi/WzsWJxUbaiY/0.jpg" 
width="480" height="360" border="10" /></a></p>

> **1. Search:**  
This lesson discuss about discrete path planning and algorithms for solving the path planning problem.

> **2. Prediction:**  
Use data from sensor fusion to generate predictions about the likely behavior of the moving objects.

> **3. Behavior Planning:**  
Hight level behavior planning in a self driving car.

> **4. Trajectory Generation:**  
Use C++ and Eigen linear algebra library to build candidate trajectories for the vehicle to follow.

> **_5. Project: Path Planning:_**  
**Drive a car down a highway with other cars using own path planner.**

### Semantic Segmentation

<p align="center"><a target="_blank"><img src="http://img.youtube.com/vi/PNzQ4PNZSzc/0.jpg" 
width="480" height="360" border="10" /></a></p>

> **1. Advanced Deep Learning:**  
This lesson discussed semantic segmentation and inference optimization.

> **2. Fully Convolutional Networks:**  
This lesson deals with motivation for fully convolutional network and how they are structured.

> **3. Scene Understanding:**  
Introduction to scene understanding and role of FCNs play.

> **4. Inference Performance:**  
Familiarize with various optimizations in an effort to squeeze every last bit of performance at inference.

> **_5. Project: Semantic Segmentation:_**  
**Train segmentation networks, which paint each pixel of the image with different color based on its class. Use segmented images to find free space on road.**

### Systems

<p align="center"><a target="_blank"><img src="http://img.youtube.com/vi/ESIz-aSjklY/0.jpg" 
width="480" height="360" border="10" /></a></p>

> **1. Autonomous Vehicle Architecture:**  
Learn about architecture of Carla, Udacity's autonomous vehicle.

> **2. Introduction to ROS:**  
Architectural overview of Robotic Operating System framework.

> **3. Packages and Catkin Workspaces:**  
Introduction to ROS workspace structure, essential command line utilities and how to manage software packages within a project.

> **4. Writing ROS Nodes:**  
ROS nodes are key abstraction that allows a robot system to be built modularly.

> **_5. Project: System Integration Project:_**  
**Write ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following!**
