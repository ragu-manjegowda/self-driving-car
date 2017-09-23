# P1-CarND-Path-Planning-Project
Udacity Self-Driving Car Nanodegree - Term3, Project 1 

## Path Planing Implementation

### Introduction
The goal is to find a safe, comfortable and efficient path through a set of dynamic maneuverable objects in the simulator. This includes path searching, prediction, behavior planning and trajectory generation. In this project, a path planning model on enclosed highway is implemented. Since the highway map is already known, the path searching and prediction are easy as the car should just follow the highway based on the given way points. Trajectory generation and behavior planning are dealt in this project. 

### Trajectory generation
Since this project is for highway path planning, polynomial trajectory generation will be a good choice. To make polynomial calculation, spline is leveraged in this model, as suggest in the project page. 3 future waypoints (seperated 30m) are predicted and 2 previous points are used to provide a rough path. And each point co-ordinates are converted from global co-ordinates to car local co-ordinates so that the spline always starts at (0, 0) as shown below.

```

// if previous size is consumed use current position as reference
if(prev_size < 2)
{
  // calculate previous path point using yaw
  double prev_car_x = car_x - cos(car_yaw);
  double prev_car_y = car_y - sin(car_yaw);

  ptsx.push_back(prev_car_x);
  ptsx.push_back(car_x);

  ptsy.push_back(prev_car_y);
  ptsy.push_back(car_y);
}
// if previous path points exist use last two points again
else
{
  ref_x = previous_path_x[prev_size - 1];
  ref_y = previous_path_y[prev_size - 1];

  double ref_prev_x = previous_path_x[prev_size - 2];
  double ref_prev_y = previous_path_y[prev_size - 2];

  ref_yaw = atan2(ref_y - ref_prev_y, ref_x - ref_prev_x);

  ptsx.push_back(ref_prev_x);
  ptsx.push_back(ref_x);

  ptsy.push_back(ref_prev_y);
  ptsy.push_back(ref_y);
}


// Add three 30m spaced points in future
vector<double> next_wp0 = getXY((car_s + 30), (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp1 = getXY((car_s + 60), (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp2 = getXY((car_s + 90), (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

ptsx.push_back(next_wp0[0]);
ptsx.push_back(next_wp1[0]);
ptsx.push_back(next_wp2[0]);

ptsy.push_back(next_wp0[1]);
ptsy.push_back(next_wp1[1]);
ptsy.push_back(next_wp2[1]);

for(int i = 0; i < ptsx.size() ; i++)
{
  // shift car reference angle to 0 degrees
  double shift_x = ptsx[i] - ref_x;
  double shift_y = ptsy[i] - ref_y;

  ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
  ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
}

// create a spline
tk::spline s;

// set (x, y) points to pline
s.set_points(ptsx, ptsy);

// define vector for waypoints
vector<double> next_x_vals;
vector<double> next_y_vals;

// use all previous points
for(int i = 0; i < previous_path_x.size(); i++)
{
  next_x_vals.push_back(previous_path_x[i]);
  next_y_vals.push_back(previous_path_y[i]);
}

//calculate points using spline to travel at desired velocity
double target_x = 30.0; // try to get 30th point (approx 30m)
double target_y = s(target_x);
double target_dist = sqrt((target_x * target_x) + (target_y * target_y));

double x_add_on = 0; // start at 0

// fill up the rest of 50 points
for(int i = 0; i < 50 - previous_path_x.size(); i++)
{
  // convert ref velocity to m/s, divide by 2.24
  double N = (target_dist / ( 0.02 * (ref_vel / 2.24)));
  double x_pt = x_add_on + (target_x / N);
  double y_pt = s(x_pt);

  x_add_on = x_pt;

  double x_ref = x_pt;
  double y_ref = y_pt;

  // rotate back to normal
  x_pt = ((x_ref * cos(ref_yaw)) - (y_ref * sin(ref_yaw)));
  y_pt = ((x_ref * sin(ref_yaw)) + (y_ref * cos(ref_yaw)));

  x_pt += ref_x;
  y_pt += ref_y;

  next_x_vals.push_back(x_pt);
  next_y_vals.push_back(y_pt);
}

```


### Behavior planning
In high way driving, car can have following possible states: keep lane, left lane change, right lane change. It is pretty straight forward when we are in the left mostr or right most lane as the only option is to drive to middle lane when it is safe. But when car is in middle lane, periodically I check if it is safe to change to either left or right lane. When it is safe I simply change to desired lane. If there is no car 30m ahead of my position and 5m behind me in the desired lane, then I consider it as safe condition an activate lane change mechanism.

I. Change lane: 

Change lane code is triggered when we have car 30m ahead of us.

```

for(int i = 0; i < sensor_fusion.size(); i++)
{
  // check lanes of nearby cars
  float d = sensor_fusion[i][6];

  // if there is a car ahead of us
  if(d < (2+4*lane+2) && d > (2+4*lane-2))
  {
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double check_speed = sqrt(vx*vx + vy*vy);
    double check_car_s = sensor_fusion[i][5];

    check_car_s += (double)prev_size*0.02*check_speed;

    // check if the car is within 30 meters from our car
    if(check_car_s > car_s && ((check_car_s - car_s) < 30))
```


II. Change lane if it is safe to change:

Based on the above behavioural discussion below is my implmentation.

```

for(int i = 0; i < sensor_fusion.size(); i++)
{
  // check lanes of nearby cars
  float d = sensor_fusion[i][6];

  // if there is a car ahead of us
  if(d < (2+4*lane+2) && d > (2+4*lane-2))
  {
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double check_speed = sqrt(vx*vx + vy*vy);
    double check_car_s = sensor_fusion[i][5];

    check_car_s += (double)prev_size*0.02*check_speed;

    // check if the car is within 30 meters from our car
    if(check_car_s > car_s && ((check_car_s - car_s) < 30))
    {
      too_close = true;
      if(lane == 0)
      {
        if(safeLaneChange(1, sensor_fusion, car_s, prev_size))
        {
          lane = 1;
        }
      }
      else if(lane == 1)
      {
        if(laneFlag)
        {
          if(safeLaneChange(0, sensor_fusion, car_s, prev_size))
          {
            lane = 0;
          }
        }
        else
        {
          if(safeLaneChange(2, sensor_fusion, car_s, prev_size))
          {
            lane = 2;
          }
        }
      }
      else
      {
        if(safeLaneChange(1, sensor_fusion, car_s, prev_size))
        {
          lane = 1;
        }
      }

      laneFlag = !laneFlag;

    }
  }
}

```         

III. safeLaneChange function's implementation.

```
typedef nlohmann::basic_json<std::map, std::vector,                 
      std::__1::basic_string<char>, bool, long long, unsigned long long, double, std::allocator, nlohmann::adl_serializer> datatype;

bool safeLaneChange(int lane, datatype sensor_fusion, double car_s, int prev_size)
{

  //auto sensor_fusion = (auto) sensor_fusion2;
  for(int i = 0; i < sensor_fusion.size(); i++)
  {
    // check lanes of nearby cars
    float d = sensor_fusion[i][6];

    // if there is a car ahead of us
    if(d < (2+4*lane+2) && d > (2+4*lane-2))
    {
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = sqrt(vx*vx + vy*vy);
      double check_car_s = sensor_fusion[i][5];

      check_car_s += (double)prev_size*0.02*check_speed;

      // check if the car is within 30 meters from our car
      if( (car_s - check_car_s) < 5 && ((check_car_s - car_s) < 30))
        return false;
    }
  }
    return true;
}

```

### Summary
In summary, I followed the classroom suggestions and designed the path plannig model accordingly. Spline was much helpful in getting the intermediate points between 30m waypoints in future. Also Aarons sugession to increment and decrement speed in steps helped in fixing the jerk.