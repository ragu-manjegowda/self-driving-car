# PID Controller

## Driving Simulator car with PID Controller

### 1. The PID procedure follows what was taught in the lessons.
Implemented the sections of maon.cpp and PID.cpp based on the lessons taught in class.

### 2. Describe the effect each of the P, I, D components had in your implementation.
P controller alone has the oscillation effect, with just the P controller car keeps on oscillating between the tracks and never become stable.
I was also able to see the overshooting where car goes off the track.

With just PD controller I was able to see the bias resulting in fail to reduce cross talk error.

The PID controller performed well as expected, where the car takes some time to stabilize (after few oscillations) and then becomes stable driving through the track. 

### 3. Describe how the final hyperparameters were chosen. 
I chosed the hyperparameters by trial and error. Parameters those used in the class (0.2, 0.004, 3.0) worked well for me. I tried changing these values which resulted in either big oscillations or the car moving out of track. 

### 4. The vehicle must successfully drive a lap around the track.
My PID controller keeps the car within the drivable portion of the track. None of thre tires goes off the road. 
    
    
