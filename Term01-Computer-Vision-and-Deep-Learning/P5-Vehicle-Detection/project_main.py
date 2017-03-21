import numpy as np
import cv2
import pickle

from detection_functions.Vehicle_Classification import *
from detection_functions.Vehicle import *

from moviepy.editor import VideoFileClip

MTX=None
DIST=None

VERBOSE=False

def process_image(raw_image, correct_distortion=False):
    img_shape = raw_image.shape

    if correct_distortion:
    # Correct Distortion with calculated Camera Calibration (If not present calibrate)
        global MTX, DIST, VERBOSE
        mtx, dist, process_image = correct_distortion(raw_image, mtx=MTX, dist=DIST, verbose=VERBOSE)
        if (MTX == None) | (DIST == None):
            MTX = mtx
            DIST = dist
    else:
        process_image = raw_image

    if vehicle_collection.image_initialized == False:
        vehicle_collection.initalize_image(img_shape=process_image.shape, y_start_stop=[440, 720], xy_window=(360, 360), xy_overlap=(0.9, 0.8))
    vehicle_collection.find_hot_windows(process_image, vehicle_classification)
    vehicle_collection.analyze_current_stripe(process_image)

    draw_image = vehicle_collection.identify_vehicles(process_image)

    return draw_image

if __name__ == "__main__":
    VERBOSE = True
    LEARN_NEW_CLASSIFIER = False

    vehicle_classification = Vehicle_Classification()
    vehicle_classification.train_classifier(LEARN_NEW_CLASSIFIER)

    vehicle_collection = Vehicle_Collection()

    heatmap_frame_collection = None

    #image = cv2.imread('./test_images/test6.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #processed_image = process_image(image)
    #processed_image = cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR)
    #cv2.imshow('Resulting Image', processed_image)

    #cv2.imwrite('../output_images/test2_applied_lane_lines.jpg', combo)

    video_output = './project_video_calc.mp4'
    clip1 = VideoFileClip('./project_video.mp4')
    #clip1 = VideoFileClip('./test_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')

    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)


