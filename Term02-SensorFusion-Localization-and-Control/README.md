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

