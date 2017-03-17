import cv2
import numpy as np

import logging

def create_blank(height, width, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def add_image_at_position(base_img, add_img, position_percentage=(0,1,0,1)):
    # If Image On_Color_Image Convert to 3 Channels
    add_img_shape = len(add_img.shape)
    #print(add_img_shape)
    #ToDo: Find Correct criteria for grayscale Image
    if add_img_shape != 3:
        add_img = cv2.cvtColor(add_img, cv2.COLOR_GRAY2BGR)

    base_img_shape = base_img.shape
    position_px = (int(position_percentage[0]*base_img_shape[0]),
                   int(position_percentage[1]*base_img_shape[0]),
                   int(position_percentage[2]*base_img_shape[1]),
                   int(position_percentage[3]*base_img_shape[1]))
    new_img_shape = ((position_px[3]-position_px[2]), (position_px[1]-position_px[0]))
    new_img = cv2.resize(add_img, new_img_shape)
    base_img[position_px[0]:position_px[1], position_px[2]:position_px[3]] = new_img

    return base_img

def two_images(img1, img2):
    img_shape = img1.shape
    new_image = create_blank(img_shape[0], img_shape[1] * 2)

    new_image = add_image_at_position(new_image, img1, (0, 1, 0, .5))
    new_image = add_image_at_position(new_image, img2, (0, 1, .5, 1))

    return new_image

def image_cluster(img_list=[], img_text=False, new_img_shape=None, cluster_shape=None, font_size=.5, y_position=40):
    if cluster_shape == None:
        val_col = int(np.ceil(np.sqrt(len(img_list))))
        val_row = int(np.ceil(len(img_list)/val_col))
        cluster_shape = (val_row, val_col)
    if new_img_shape == None:
        new_img_shape = img_list[0].shape
        new_img_shape = tuple([int(new_img_shape[i]*cluster_shape[i]/max(cluster_shape)) for i in range(len(cluster_shape))])
    #Todo: Calculate New Image Shape if One Value is given

    size_of_cluster = (cluster_shape[0] * cluster_shape[1])
    if size_of_cluster < len(img_list):
        logging.info('Cluster to small for all Images')

    new_image = create_blank(new_img_shape[0], new_img_shape[1])
    cluster_index = 0
    for row in range(cluster_shape[0]):
        for col in range(cluster_shape[1]):
            new_image = add_image_at_position(new_image, img_list[cluster_index], (row/cluster_shape[0], (row+1)/cluster_shape[0], col/cluster_shape[1], (col+1)/cluster_shape[1]))

            if img_text:
                if cluster_index < len(img_text):
                    if img_text[cluster_index] != None:
                        text_position = (col*int(new_img_shape[1]/cluster_shape[1])+20, row*int(new_img_shape[0]/cluster_shape[0])+y_position)
                        cv2.putText(new_image, img_text[cluster_index], text_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)

            cluster_index += 1
            if (cluster_index >= len(img_list)):
                break
        if (cluster_index >= len(img_list)):
            break

    return new_image

if __name__ == "__main__":
    # Read in the image

    img1 = cv2.imread('../output_images/test1_detected.jpg')
    img2 = cv2.imread('../output_images/test3_detected.jpg')
    img3 = cv2.imread('../output_images/test4_detected.jpg')
    img4 = cv2.imread('../output_images/test5_detected.jpg')

    #new_image = two_images(img1,img2)

    new_image = image_cluster([img1, img2, img3, img4], img_text=['Image 1','Image 2', 'Image 3','Image 4'])

    #new_image = cv2.resize(new_image, (256 * 5, 72 * 5))
    #new_image = cv2.resize(new_image, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('Two Images', new_image)
    cv2.waitKey(0)

    cv2.imwrite('../output_images/detected_examples.jpg', new_image)



