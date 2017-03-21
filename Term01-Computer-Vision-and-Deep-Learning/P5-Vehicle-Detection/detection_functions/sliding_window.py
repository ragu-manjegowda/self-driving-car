from toolbox.draw_on_image import *

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img_shape, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img_shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img_shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y

    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int((xspan-xy_window[0]) / nx_pix_per_step)+1
    ny_windows = np.int((yspan-xy_window[1]) / ny_pix_per_step)+1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            if endx > img_shape[1]:
                delta_x = endx - img_shape[1]
                endx = endx - delta_x
                startx = startx - delta_x
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def return_labeled_windows(labels):
    bbox_collection = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox)-1, np.min(nonzeroy)-1), (np.max(nonzerox)+1, np.max(nonzeroy)+1))

        bbox_collection.append(bbox)

    return bbox_collection

def perspective_width(new_y,y_start_stop=[420, 720],bottom_width=360, top_width=32):
    new_width = int(((1. - ((y_start_stop[1] - new_y) / (y_start_stop[1] - y_start_stop[0]))) * (bottom_width - top_width)) + top_width)
    return int(new_width)

def slide_precheck(img_shape, y_start_stop=[440, 720], xy_window_start=(440, 440), xy_overlap = (0.0, 0.8)):
    shifted_collection = []
    #dead_step_y = int(xy_window_start[1] * (1.0 - xy_overlap[1]))
    for shift in range(10):
        y_position = y_start_stop[1]
        x_position = 0
        xy_window = xy_window_start

        windows_collection = []

        while (xy_window[0] >= 64):
            dead_step_x = int(xy_window[0] * (1.0 - xy_overlap[0]) * shift / 10)
            dead_x_position = x_position + dead_step_x
            y_step = int(xy_window[0] * (1.0 - xy_overlap[1]))

            windows = slide_window(img_shape, x_start_stop=[dead_x_position, img_shape[1]-x_position],
                                               y_start_stop=[y_position - xy_window[0], y_position],
                                               xy_window=xy_window, xy_overlap=xy_overlap)

            y_position = y_position - y_step
            width = perspective_width(y_position,y_start_stop=y_start_stop,bottom_width=xy_window_start[1], top_width=32)
            xy_window = (width, width)



            x_width = perspective_width(y_position,y_start_stop=y_start_stop,bottom_width=img_shape[1]*15, top_width=32*15)

            if x_width > (img_shape[1]):
                x_position = 0
            else:
                x_position = int((img_shape[1] - x_width)/2)

            windows_collection.append(windows)

        shifted_collection.append(windows_collection)
    return shifted_collection


if __name__ == "__main__":

    image = cv2.imread('../test_images/test1.jpg')
    img_shape = image.shape
    shifted_collection = slide_precheck(img_shape)

    for shift in shifted_collection:
        print(len(shift))
        for stripe in shift:
            image = draw_boxes(image, stripe)
        cv2.imshow('shifted_collection', image)

        cv2.imwrite('../output_images/sliding_windows.jpg', image)
        cv2.waitKey(2000)
