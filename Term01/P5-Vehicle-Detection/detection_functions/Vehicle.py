from detection_functions.sliding_window import *
from detection_functions.detection_pipeline import *

from scipy.ndimage.measurements import label

class Vehicle():
    def __init__(self, vehicleWindow):
        self.window = vehicleWindow
        self.vehicle_Window_List = [vehicleWindow]
        self.window_size = 0

    def return_window_size_x(self):
        self.window_size = self.window[1][0] - self.window[0][0]
        return self.window_size

class Vehicle_Collection():
    def __init__(self):
        self.image_initialized = False
        self.img_shape = None
        self.precheck_windows = None
        self.hot_windows = None

        self.detected_vehicles = []

        self.height_shift = 0
        self.circle_shift = 0

        self.heatmap_array = None
        self.abs_heatmap = None

        self.window_stripe_collection = []

    def initalize_image(self, img_shape=(720,1280), y_start_stop=[440, 720], xy_window=(440, 440), xy_overlap = (0.5, 0.5)):
        self.image_initialized = True
        self.img_shape = img_shape
        self.precheck_windows = slide_precheck(img_shape, y_start_stop=y_start_stop, xy_window_start=xy_window, xy_overlap=xy_overlap)

    def find_hot_windows(self, process_image, vehicle_classification):
        self.circle_shift = (self.circle_shift +1)%10
        self.height_shift = (self.height_shift + 1) %len(self.precheck_windows[0])
        self.hot_windows = search_windows(process_image, self.precheck_windows[self.circle_shift], vehicle_classification.classifier, vehicle_classification.X_scaler, color_space=vehicle_classification.color_space,
                                 spatial_size=vehicle_classification.spatial_size, hist_bins=vehicle_classification.hist_bins,
                                 orient=vehicle_classification.orient, pix_per_cell=vehicle_classification.pix_per_cell,
                                 cell_per_block=vehicle_classification.cell_per_block,
                                 hog_channel=vehicle_classification.hog_channel, spatial_feat=vehicle_classification.spatial_feat,
                                 hist_feat=vehicle_classification.hist_feat, hog_feat=vehicle_classification.hog_feat)

    def sort_vehicle_collection(self):
        for vehicle in self.detected_vehicles:
            vehicle.return_window_size_x()
        self.detected_vehicles = sorted(self.detected_vehicles, key=lambda vehicle: vehicle.window_size, reverse=True)

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
                    #print('new vehicle')
                    new_vehicle_list.append(Vehicle(window))

            self.detected_vehicles.extend(new_vehicle_list)

    def identify_vehicles(self, process_image):
        draw_image = np.copy(process_image)
        ghost_cars = []
        for vehicle_index in range(len(self.detected_vehicles)):
            for i in range(1):
                if not self.detected_vehicles[vehicle_index].vehicle_Window_List:
                    break
                self.detected_vehicles[vehicle_index].vehicle_Window_List.pop(0)
            if not self.detected_vehicles[vehicle_index].vehicle_Window_List:
                ghost_cars.append(vehicle_index)
        for ghost in ghost_cars[::-1]:
            #print(ghost,ghost_cars,self.detected_vehicles)
            self.detected_vehicles.pop(ghost)

        for vehicle_index in range(len(self.detected_vehicles)):

            while len(self.detected_vehicles[vehicle_index].vehicle_Window_List) > 40:
                self.detected_vehicles[vehicle_index].vehicle_Window_List.pop(0)
            if len(self.detected_vehicles[vehicle_index].vehicle_Window_List) > 12:

                heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
                heatmap = add_heat(heatmap, self.detected_vehicles[vehicle_index].vehicle_Window_List)

                heat_thresh = apply_threshold(heatmap, 5)
                labels = label(heat_thresh)

                label_windows = return_labeled_windows(labels)
                if label_windows:
                    self.detected_vehicles[vehicle_index].window = label_windows[0]
                if self.detected_vehicles[vehicle_index].return_window_size_x() > 20:
                    draw_image = cv2.rectangle(draw_image, tuple(self.detected_vehicles[vehicle_index].window[0]), tuple(self.detected_vehicles[vehicle_index].window[1]), (0,255,0), 2)
                    #draw_image = draw_labeled_bboxes(draw_image, labels)


            #draw_image = cv2.rectangle(draw_image, tuple(self.detected_vehicles[vehicle_index].window[0]),tuple(self.detected_vehicles[vehicle_index].window[1]), (0, 0, 255), 1)

        #cv2.imshow('test', draw_image)
        #cv2.waitKey(1)

        return draw_image




### NOT IN USE AT THE TIME
    def analyze_stripes(self, process_image):
        #total_detections=[]
        draw_image = np.copy(process_image)
        if self.heatmap_array == None:
            self.heatmap_array = []

        self.window_stripe_collection = []

        for stripe_index in range(len(self.hot_windows)):
            stripe = self.hot_windows[stripe_index]
            if len(stripe) > 0:
                heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
                heatmap = add_heat(heatmap, stripe)

                self.heatmap_array = [None] * len(self.hot_windows)

                if self.heatmap_array[stripe_index] == None:
                    self.heatmap_array[stripe_index] = np.array(heatmap, ndmin=3)
                elif len(self.heatmap_array[stripe_index]) < 5:
                    self.heatmap_array[stripe_index] = np.append(self.heatmap_array[stripe_index], np.array(heatmap, ndmin=3), axis=0)
                else:
                    self.heatmap_array[stripe_index] = np.roll(self.heatmap_array[stripe_index], -1, axis=0)
                    self.heatmap_array[stripe_index][-1, :] = np.array(heatmap, ndmin=2)

                heatmap = np.mean(self.heatmap_array[stripe_index], axis=0)

                heat_thresh = apply_threshold(heatmap, 1)

                labels = label(heat_thresh)

                self.window_stripe_collection = self.window_stripe_collection + (return_labeled_windows(labels))

        #if self.abs_heatmap == None:
        #self.abs_heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
        #labels_1 = label(add_heat(self.abs_heatmap, self.window_stripe_collection))

        #self.abs_heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
        #self.abs_heatmap = add_heat_labels(self.abs_heatmap, self.window_stripe_collection, labels_1)

        #self.abs_heatmap = apply_threshold(self.abs_heatmap, 0.275)
        #draw_image = weighted_img(self.abs_heatmap, draw_image, α=1., β=.8, λ=0., color=(255, 0, 0))

        labels = label(self.abs_heatmap)

        #draw_image = draw_labeled_bboxes(draw_image, labels)
        #draw_image = draw_boxes(draw_image, self.window_stripe_collection)

        #cv2.imshow('Labels', draw_image)
        #cv2.waitKey(1)

        return draw_image


            #for car_number in range(1, labels[1] + 1):
                # Find pixels with each car_number label value
            #    nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
            #    nonzeroy = np.array(nonzero[0])
            #    nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
            #    bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            #    total_detections.append(bbox)

        #heatmap = np.zeros_like(process_image[:, :, 0]).astype(np.float)
        #heatmap = add_heat(heatmap, total_detections)
        #heatmap = cv2.GaussianBlur(heatmap,(65,65),0)

        #heatmap = heatmap * 10

        #if self.heatmap_frame_collection == None:
        #    self.heatmap_frame_collection = np.array(heatmap, ndmin=3)
        #elif self.heatmap_frame_collection.shape[0] < 20:
        #    self.heatmap_frame_collection = np.append(self.heatmap_frame_collection, np.array(heatmap, ndmin=3), axis=0)
        #else:
        #    self.heatmap_frame_collection = np.roll(self.heatmap_frame_collection, -1, axis=0)
        #    self.heatmap_frame_collection[-1, :] = np.array(heatmap, ndmin=2)

        #heatmap = np.mean(self.heatmap_frame_collection, axis=0)


        process_image = weighted_img(heatmap, process_image, α=0.4, β=1., λ=0., color=(255,0,0))

