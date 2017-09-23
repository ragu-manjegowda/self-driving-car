# P1-CarND-Path-Planning-Project
Udacity Self-Driving Car Nanodegree - Term3, Project 1 

## The code compiles correctly.
Code compiles without any errors or warnings! 

## The car is able to drive at least 4.32 miles without incident.
No incidents are reported at top right screen of the simulator that shows the current/best miles driven without incident. Incidents include exceeding acceleration/jerk/speed, collision, and driving outside of the lanes. Each incident case is also listed below in more detail.

## The car drives according to the speed limit.
Car drives within the speed limit (50mph), I have set the max speed of the model to 49.9mph. Also the car isn't driving much slower than speed limit unless obstructed by traffic. Line 329 in main.cpp defines max speed limit of 49.9.

## Max Acceleration and Jerk are not Exceeded.
Car uses the acceleration and deacceleration are done in the steps of 0.224 which is within the defined limit of total acceleration of 10 m/s^2 and a jerk of 10 m/s^3. Line 347 and Line 351 in main.cpp defines steps to increase and decrease speed.

## Car does not have collisions.
Car slows down when it encounters any vehicle ahead of its path within 30m distance. Also car will not change lane when it encounters vehicle within 30m distance ahead and 5m behind it in the intended lane within. Line 304 defines safe distance between vehicle.

## The car stays in its lane, except for the time between changing lanes.
The car doesn't spend more than a 3 second length out side the lane lanes during changing lanes, and every other time the car stays inside one of the 3 lanes on the right hand side of the road. Code starting line 393 defines lane number that is used to calculate d of the vehicle to stay in lane.

## The car is able to change lanes
The car is able to smoothly change lanes when it makes sense to do so, such as when behind a slower moving car and an adjacent lane is clear of other traffic. Code starting from line 307 defines lane changing conditions.

## Reflection on how to generate paths.
The code model for generating paths is described in detail. Please refer "Model Documentation".