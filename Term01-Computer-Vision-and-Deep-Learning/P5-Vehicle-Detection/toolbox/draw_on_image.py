import cv2
import numpy as np

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=2):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0., color=(255,0,0)):
    """
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """

    color_image = np.zeros((initial_img.shape[0], initial_img.shape[1], 3), np.uint8)
    color_image[((img >= 1))] = color

    return cv2.addWeighted(initial_img, α, color_image, β, λ)
