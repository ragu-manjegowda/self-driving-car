**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/detected_examples.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I first wrote a class, which stores all learning parameters, pickles and trains the classifier.
You will find it in `./detection-functions/Vehicle_Classififcation.py`

The Code for Feature Extraction is provided in `./detection_functions/festure_extraction.py` 
Here you find the feature extraction pipeline for Hog Features, Spatial Features and Color Histograms.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters such as `orientations`, `pixels_per_cell`, and `cells_per_block`.

I also did some research on which color space is the best for vehicle detection, therefore I plotted some vehicle end no_vehicle images and tried out, which parameters are best for gradient and best for color recognition.
I Chose the LUV space, because the L chanel does provide very good information for the HOG Gradient approach.

#### 2. Explain how you settled on your final choice of HOG parameters.

First I just chose a random set of different parameters and looked for best detections.
After getting the vehicle detection run after short time, I started tweeking those parameters for performance.
So I chose to have less features, so the predict will work much faster later on.
I looked for the best prediction concerning the best performance.

So for me it turned out to be follwing parameter:

```
    def __init__(self):
        self.color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = '0'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 64 # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off

        self.classifier = None
        self.X_scaler = None

```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I am using a Support Vector Machine with a Linear Kernel. it gets trained in the function train_calssifier in `./detection-functions/Vehicle_Classififcation.py`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In file `./detection_functions/sliding_window` I wrote an algorithm, that performs a sliding window based on the horizon position.
So the quadratic windows get sized by the position in the image relativly to the horizon.

The metric for sliding on the y axis up and down is based on the window size in current position. 
After performing a horizontal slide every 20% of cluster-tile size, I move upwards 20% of cluster-tile size and recalc the new size of the cluster-tile.

This algorithm reduces the amount of windows to proof and gives hough performance feedback.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As described before I used the `./detection-functions/Vehicle_Classification.py` class to extract the features. To optimize the performance I looked for a reasonable good classification using as less paramters as possible.

The image gets divided in horizontal stripes based on the horizon and then I predict for every stripe.
That gives me a very good opportunity to prepare every stripe onces for HOG, Spatial and Color feature extraction. So I dont have to calculate HOG and resize the image in every iteration, but only once when I switch to the stripe.
This gives me a very good performance. 

I dont care to much for false positives later on, because my vehicle detection pipeline is pretty robust against so called Ghost Cars. But even doing so, the number of false positives is pretty low with the chosen parameters.
Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_calc.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

All of the following code you will find in the `Vehicle_Collection()` class.
Here I implment algorithm to reduce all hot windows to Vehicle positions.

Also you will find a `Vehicle()` class, where I store all information about the detected Vehicle. This could be easily added by vehicle relative speed and further information.

I first look for all hot_windows per horizontal stripe. You will find the code in `./detection_functions/detection_pipline.py`
Then I take those hot windows per horizontal line and combine them using a heatmap and thresholding process.

When this is finished, I reduced the overlaying hot_windows per stripe to possible car locations in the horizon positions.
Afterwoods I try to match every stripe detections to a current detected vehicle. Therefore I look for detected cars, which overlay in the horizon stripe.

If this is not possible, I add a new vehicle by using the Vehicle Class.

Here is the code:

```
    def analyze_current_stripe(self, process_image):
        for stripe in self.hot_windows:
            heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
            heatmap = add_heat(heatmap, stripe)
            heat_thresh = apply_threshold(heatmap, 2)
            labels = label(heat_thresh)
            self.window_stripe_collection = self.window_stripe_collection + (return_labeled_windows(labels))

        hot_windows = self.window_stripe_collection
        self.sort_vehicle_collection()
        for vehicle in self.detected_vehicles:
            if vehicle.window:
                for hot_window_index in range(len(hot_windows))[::-1]:
                    if (vehicle.window[0][0] <= hot_windows[hot_window_index][1][0]) & (vehicle.window[1][0] >= hot_windows[hot_window_index][0][0]):
                        if((vehicle.window[1][0] - vehicle.window[0][0])*1.5 > (hot_windows[hot_window_index][1][0]-hot_windows[hot_window_index][0][0])):
                            vehicle.vehicle_Window_List.append(hot_windows[hot_window_index])
                            hot_windows.pop(hot_window_index)

        if hot_windows:
            new_vehicle_list = []
            for window in hot_windows:
                window_is_new_vehicle = True
                for vehicle in new_vehicle_list:
                    if vehicle.window:
                        if (vehicle.window[0][0] <= window[1][0]) & (vehicle.window[1][0] >= window[0][0]):
                            if ((vehicle.window[1][0] - vehicle.window[0][0]) * 2 > (window[1][0] - window[0][0])):
                                vehicle.vehicle_Window_List.append(window)
                                window_is_new_vehicle = False
                            break
                if window_is_new_vehicle:
                    print('new vehicle')
                    new_vehicle_list.append(Vehicle(window))
```

Now I have a collection of detected vehicles with all there corresponding windows.
I store 40 detected windows for every car and at least 30% of 40 (12) to claim a car as not Ghost Car.
Since most of the cars get detected in more than one stripe at a time, the algorithm is very efficient to detect cars in with 3-4 frames and in popping out Ghost cars.

For each frame every car looses some of his detected windows, to make sure, only new detections are in the set.

To have a valid Vehicle Size I perform a heatmap on every set of detected Vehicle windows and resize the vehicle class size by the new thresholded window.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Using a SVC not linear had any better classification results, but it turned out to be slow in performance.

I am happy with my approach and especially the horizon performance of sliding window does a very good job.
But I had a hard work to estimate a really tight window around the cars, without loosing performance.
